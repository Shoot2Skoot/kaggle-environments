import json
import logging
import random
from os import path
from typing import Callable, Dict

from .game.actions import (
    Action,
    BidAction,
    ChatAction,
    DiscardPolicyAction,
    ExecutionAction,
    InvestigateAction,
    NominateChancellorAction,
    NoOpAction,
    PolicyPeekAckAction,
    SpecialElectionAction,
    VetoConsentAction,
    VoteAction,
    create_action,
)
from .game.consts import DetailedPhase, EnvInfoKeys, Power, RoleConst, Team, Vote
from .game.engine import Moderator
from .game.protocols.factory import DEFAULT_DISCUSSION_PROTOCOL_NAME, create_protocol
from .game.records import (
    SecretHitlerObservationModel,
    get_raw_observation,
    set_raw_observation,
)
from .game.rng import FIRST_PRESIDENT_OFFSET, deterministic_index
from .game.roles import assign_role_counts, create_players_from_agents_config, hitler_knows_fascists
from .game.states import EventName, GameState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Built-in agents (legal-by-construction; read the observation)
# ------------------------------------------------------------------ #
def _common_args(raw):
    return {"round_number": raw.round_number, "phase": raw.game_state_phase, "actor_id": raw.player_id}


def _act(raw, pick):
    """Build an action for the current phase. ``pick`` chooses from a sequence (random/first)."""
    phase = DetailedPhase(raw.detailed_phase)
    args = _common_args(raw)

    if phase == DetailedPhase.ELECTION_NOMINATION_AWAIT:
        targets = raw.eligible_chancellor_ids or [p for p in raw.alive_players if p != raw.player_id]
        return NominateChancellorAction(**args, target_id=pick(targets), reasoning="auto")
    if phase == DetailedPhase.ELECTION_BIDDING_AWAIT:
        return BidAction(**args, amount=pick([0, 1, 2, 3]), reasoning="auto")
    if phase == DetailedPhase.ELECTION_CHAT_AWAIT:
        return ChatAction(**args, message=pick(["I support this.", "I am suspicious.", "Pass."]), reasoning="auto")
    if phase == DetailedPhase.ELECTION_VOTE_AWAIT:
        return VoteAction(**args, vote=pick([Vote.JA, Vote.NEIN]), reasoning="auto")
    if phase == DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT:
        hand = raw.policy_hand or []
        return DiscardPolicyAction(**args, discard=pick(hand) if hand else None, reasoning="auto")
    if phase == DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT:
        hand = raw.policy_hand or []
        return DiscardPolicyAction(**args, discard=pick(hand) if hand else None, veto=False, reasoning="auto")
    if phase == DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT:
        return VetoConsentAction(**args, consent=pick([True, False]), reasoning="auto")
    if phase == DetailedPhase.EXECUTIVE_POWER_AWAIT:
        power = raw.pending_power
        targets = raw.power_valid_targets or [p for p in raw.alive_players if p != raw.player_id]
        if power == Power.INVESTIGATE:
            return InvestigateAction(**args, target_id=pick(targets), reasoning="auto")
        if power == Power.SPECIAL_ELECTION:
            return SpecialElectionAction(**args, target_id=pick(targets), reasoning="auto")
        if power == Power.EXECUTION:
            return ExecutionAction(**args, target_id=pick(targets), reasoning="auto")
        return PolicyPeekAckAction(**args, reasoning="auto")
    return NoOpAction(**args, reasoning="auto")


def random_agent(obs):
    raw = get_raw_observation(obs)
    return _act(raw, lambda seq: random.choice(list(seq))).serialize()


def deterministic_agent(obs):
    raw = get_raw_observation(obs)
    return _act(raw, lambda seq: list(seq)[0]).serialize()


# ------------------------------------------------------------------ #
# Observation construction
# ------------------------------------------------------------------ #
def _known_allies(game_state: GameState, player) -> Dict[str, str]:
    if player.role.team != Team.FASCIST:
        return {}
    if player.role.name == RoleConst.HITLER and not hitler_knows_fascists(game_state.num_players):
        return {}
    return {p.id: p.role.name for p in game_state.players if p.role.team == Team.FASCIST and p.id != player.id}


def _build_observation(env, moderator: Moderator, game_state: GameState, player) -> SecretHitlerObservationModel:
    phase = game_state.detailed_phase
    is_president = player.id == game_state.president_id
    is_chancellor = player.id == game_state.chancellor_id
    board = game_state.board

    pending_power = None
    power_valid_targets = []
    if is_president and phase == DetailedPhase.EXECUTIVE_POWER_AWAIT:
        pending_power = moderator.pending_power
        power_valid_targets = moderator.current_power_targets()

    veto_available = False
    if is_chancellor and phase == DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT:
        veto_available = moderator.veto_available

    eligible = []
    if is_president and phase == DetailedPhase.ELECTION_NOMINATION_AWAIT:
        eligible = moderator.eligible_chancellors()

    new_views = player.consume_messages()
    return SecretHitlerObservationModel(
        player_id=player.id,
        role=player.role.name,
        team=player.role.team,
        is_alive=player.alive,
        round_number=game_state.round_count,
        detailed_phase=phase,
        game_state_phase=game_state.phase,
        all_player_ids=game_state.all_player_ids,
        player_thumbnails=env.player_thumbnails,
        alive_players=game_state.alive_player_ids(),
        liberal_policies=board.liberal_track,
        fascist_policies=board.fascist_track,
        election_tracker=board.election_tracker,
        veto_unlocked=board.veto_unlocked,
        draw_pile_count=game_state.deck.draw_count,
        discard_pile_count=game_state.deck.discard_count,
        president_id=game_state.president_id,
        chancellor_id=game_state.chancellor_id,
        nominee_chancellor_id=game_state.nominee_chancellor_id,
        last_president_id=game_state.last_president_id,
        last_chancellor_id=game_state.last_chancellor_id,
        eligible_chancellor_ids=eligible,
        known_allies=_known_allies(game_state, player),
        policy_hand=player.get_role_state("policy_hand"),
        pending_power=pending_power,
        power_valid_targets=power_valid_targets,
        veto_available=veto_available,
        investigation_results=dict(player.get_role_state("investigations") or {}),
        new_visible_announcements=[v.description for v in new_views],
        new_player_event_views=new_views,
    )


# ------------------------------------------------------------------ #
# Interpreter
# ------------------------------------------------------------------ #
def initialize_moderator(state, env):
    num_players = len(state)
    agents_from_config = list(env.configuration.agents)

    if len(agents_from_config) < num_players:
        raise ValueError(
            f"Configuration has {len(agents_from_config)} agents, but {num_players} kaggle agents are present."
        )
    agents_from_config = [dict(cfg) for cfg in agents_from_config[:num_players]]

    # Roles must form the canonical multiset for this player count. If the configured roles do
    # not (e.g. the throwaway initial reset slices a larger config), deal canonical roles in
    # seat order; randomize_roles still shuffles them afterwards.
    expected = assign_role_counts(num_players)
    actual = {RoleConst.LIBERAL: 0, RoleConst.FASCIST: 0, RoleConst.HITLER: 0}
    for cfg in agents_from_config:
        actual[RoleConst(cfg.get("role", RoleConst.LIBERAL))] += 1
    if actual != expected:
        canonical_roles = (
            [RoleConst.LIBERAL.value] * expected[RoleConst.LIBERAL]
            + [RoleConst.FASCIST.value] * expected[RoleConst.FASCIST]
            + [RoleConst.HITLER.value] * expected[RoleConst.HITLER]
        )
        for cfg, role in zip(agents_from_config, canonical_roles):
            cfg["role"] = role

    players = create_players_from_agents_config(
        agents_from_config,
        randomize_roles=env.configuration.randomize_roles,
        randomize_ids=env.configuration.randomize_ids,
        seed=env.configuration.seed,
    )

    game_state = GameState(players=players)
    from .game.deck import PolicyDeck

    game_state.deck = PolicyDeck(seed=env.configuration.seed)
    env.game_state = game_state

    env.player_ids_map = {i: p.id for i, p in enumerate(players)}
    env.player_id_str_list = [p.id for p in players]
    env.player_thumbnails = {p.id: (p.agent.thumbnail or "") for p in players}

    discussion = create_protocol(
        env.configuration.get("discussion_protocol", {}), default_name=DEFAULT_DISCUSSION_PROTOCOL_NAME
    )

    moderator = Moderator(state=game_state, discussion=discussion)
    # Seed the first president seat deterministically.
    moderator._marker = deterministic_index(num_players, env.configuration.seed + FIRST_PRESIDENT_OFFSET)

    env.moderator = moderator
    env.info[EnvInfoKeys.MODERATOR_OBS] = []
    env.agents = agents


def parse_player_actions(state, moderator, game_state) -> Dict[str, Action]:
    parsed: Dict[str, Action] = {}
    active = set(moderator.get_active_player_ids())
    for sub_state, player in zip(state, game_state.players):
        if player.id in active and sub_state.status == "ACTIVE":
            serialized = sub_state.action
            if serialized:
                try:
                    parsed[player.id] = create_action(serialized)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to parse action for {player.id}: {exc}")
    return parsed


def record_game_end(state, env, game_state, agent_error):
    current_info = {}
    game_end_entry = next(iter(game_state.get_event_by_name(EventName.GAME_END)), None)
    if game_end_entry and game_end_entry.data:
        current_info.update(game_end_entry.data.model_dump())
    current_info["terminated_with_agent_error"] = agent_error
    env.info[EnvInfoKeys.GAME_END] = current_info

    if game_end_entry and game_end_entry.data and not agent_error:
        scores = game_end_entry.data.scores
        for i, player_id in enumerate(env.player_id_str_list):
            state[i].reward = scores.get(player_id, 0)
    else:
        # No winner (timeout) or agent error: all-zero, zero-sum-preserving.
        for i in range(len(state)):
            state[i].reward = 0


def update_agent_messages(state, env, moderator, game_state, is_game_done, active_ids, agent_error):
    for i, sub_state in enumerate(state):
        player_id = env.player_ids_map[i]
        player = game_state.get_player_by_id(player_id)

        # Preserve framework error statuses so core can null the offending agent's reward.
        if sub_state.status in ("ERROR", "INVALID", "TIMEOUT"):
            set_raw_observation(sub_state, _build_observation(env, moderator, game_state, player))
            continue

        if player_id not in active_ids and not is_game_done:
            sub_state.status = "INACTIVE"
            # Still drain the player's queue so events aren't lost across ticks they're inactive.
            obs = _build_observation(env, moderator, game_state, player)
            set_raw_observation(sub_state, obs)
            continue

        obs = _build_observation(env, moderator, game_state, player)
        set_raw_observation(sub_state, obs)

        if is_game_done or agent_error:
            sub_state.status = "DONE"
        elif player_id in active_ids:
            sub_state.status = "ACTIVE"
        else:
            sub_state.status = "INACTIVE"


def interpreter(state, env):
    agent_error = False
    for status_code in ["TIMEOUT", "ERROR", "INVALID"]:
        if any(s["status"] == status_code for s in state):
            agent_error = True

    if not hasattr(env, "moderator") or env.done or len(env.game_state.players) != len(state):
        initialize_moderator(state, env)

    moderator: Moderator = env.moderator
    game_state: GameState = env.game_state

    # Safety: force a terminal resolution if we are about to hit the step cap with no winner.
    episode_steps = env.configuration.episodeSteps
    if not moderator.is_game_over() and len(env.steps) >= episode_steps - 1:
        moderator.force_timeout_finalize()
    else:
        parsed = parse_player_actions(state, moderator, game_state)
        moderator.advance(parsed)

    is_game_done = moderator.is_game_over() or agent_error
    if is_game_done:
        record_game_end(state, env, game_state, agent_error)

    active_ids = set(moderator.get_active_player_ids())

    global_messages = game_state.consume_messages()
    env.info[EnvInfoKeys.MODERATOR_OBS].append([rec.serialize() for rec in global_messages])

    update_agent_messages(state, env, moderator, game_state, is_game_done, active_ids, agent_error)
    return state


# ------------------------------------------------------------------ #
# Renderers
# ------------------------------------------------------------------ #
def renderer(state, env):
    if not hasattr(env, "game_state"):
        return "Game not initialized."
    game_state: GameState = env.game_state
    board = game_state.board
    lines = [
        f"=== Secret Hitler | Round {game_state.round_count} | {game_state.detailed_phase} ===",
        f"Liberal policies: {board.liberal_track}/5    Fascist policies: {board.fascist_track}/6"
        f"    Election tracker: {board.election_tracker}/3",
        f"President: {game_state.president_id}    Chancellor: {game_state.chancellor_id}",
        f"Alive: {game_state.alive_player_ids()}",
    ]
    for day in sorted(game_state.history):
        for entry in game_state.history[day]:
            if entry.public and entry.visible_in_ui:
                lines.append(f"[R{entry.day}] {entry.description}")
    return "\n".join(lines)


def html_renderer():
    jspath = path.join(path.dirname(__file__), "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


# ------------------------------------------------------------------ #
# Agent registry
# ------------------------------------------------------------------ #
agents = {
    "random": random_agent,
    "deterministic": deterministic_agent,
}


def register_agents(agent_dict: Dict[str, Callable]):
    agents.update(agent_dict)


jsonpath = path.abspath(path.join(path.dirname(__file__), "secret_hitler.json"))
with open(jsonpath) as handle:
    specification = json.load(handle)
