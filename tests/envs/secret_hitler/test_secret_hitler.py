"""Tests for the Secret Hitler environment and engine."""

import random

import pytest

from kaggle_environments import make
from kaggle_environments.envs.secret_hitler.game import actions as A
from kaggle_environments.envs.secret_hitler.game.board import Board, power_for
from kaggle_environments.envs.secret_hitler.game.consts import (
    DetailedPhase,
    EventName,
    PolicyColor,
    Power,
    RoleConst,
    Team,
    Vote,
)
from kaggle_environments.envs.secret_hitler.game.deck import PolicyDeck
from kaggle_environments.envs.secret_hitler.game.engine import Moderator
from kaggle_environments.envs.secret_hitler.game.protocols.chat import RoundRobinDiscussion
from kaggle_environments.envs.secret_hitler.game.roles import (
    assign_role_counts,
    create_players_from_agents_config,
    hitler_knows_fascists,
)
from kaggle_environments.envs.secret_hitler.game.states import GameState


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def agents_config(n):
    counts = assign_role_counts(n)
    roles = (
        [RoleConst.LIBERAL.value] * counts[RoleConst.LIBERAL]
        + [RoleConst.FASCIST.value] * counts[RoleConst.FASCIST]
        + [RoleConst.HITLER.value] * counts[RoleConst.HITLER]
    )
    return [{"id": f"p{i}", "role": roles[i], "agent_id": "random"} for i in range(n)]


def build_moderator(n, seed=1, marker=0, rounds=1):
    players = create_players_from_agents_config(agents_config(n))
    state = GameState(players=players)
    state.deck = PolicyDeck(seed=seed)
    mod = Moderator(state=state, discussion=RoundRobinDiscussion(rounds=rounds))
    mod._marker = marker
    mod.advance({})  # run setup -> first nomination
    return mod


def role_of(mod, pid):
    return mod.state.get_player_by_id(pid).role.name


def find_pid(mod, role):
    return next(p.id for p in mod.state.players if p.role.name == role)


def _scripted_action(mod, pid, *, vote=Vote.JA, discard_pref=None, nominate=None, veto=False, consent=False):
    """Build a legal action for ``pid`` given the current phase, with optional overrides."""
    state = mod.state
    phase = DetailedPhase(state.detailed_phase)
    args = dict(round_number=state.round_count, phase=state.phase, actor_id=pid)
    if phase == DetailedPhase.ELECTION_NOMINATION_AWAIT:
        eligible = mod.eligible_chancellors()
        target = nominate if (nominate in eligible) else eligible[0]
        return A.NominateChancellorAction(**args, target_id=target)
    if phase in (DetailedPhase.ELECTION_CHAT_AWAIT, DetailedPhase.LEGISLATIVE_DEBRIEF_AWAIT):
        return A.ChatAction(**args, message="x")
    if phase == DetailedPhase.ELECTION_VOTE_AWAIT:
        return A.VoteAction(**args, vote=vote)
    if phase in (DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT, DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT):
        hand = list(state.get_player_by_id(pid).get_role_state("policy_hand") or [])
        if veto and phase == DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT:
            return A.DiscardPolicyAction(**args, veto=True)
        choice = discard_pref if discard_pref in hand else hand[0]
        return A.DiscardPolicyAction(**args, discard=choice)
    if phase == DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT:
        return A.VetoConsentAction(**args, consent=consent)
    if phase == DetailedPhase.EXECUTIVE_POWER_AWAIT:
        power = mod.pending_power
        targets = mod.current_power_targets()
        if power == Power.INVESTIGATE:
            return A.InvestigateAction(**args, target_id=targets[0])
        if power == Power.SPECIAL_ELECTION:
            return A.SpecialElectionAction(**args, target_id=targets[0])
        if power == Power.EXECUTION:
            return A.ExecutionAction(**args, target_id=targets[0])
        return A.PolicyPeekAckAction(**args)
    return A.NoOpAction(**args)


def drive(mod, decide, max_steps=5000):
    """Advance the moderator to completion. ``decide(mod, pid) -> Action`` supplies actions."""
    steps = 0
    while not mod.is_game_over():
        steps += 1
        assert steps < max_steps, "engine did not terminate"
        active = mod.get_active_player_ids()
        assert active, f"no active players in {mod.state.detailed_phase}"
        actions = {pid: decide(mod, pid) for pid in active}
        mod.advance(actions)


# ------------------------------------------------------------------ #
# Unit tests
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "n,libs,fasc",
    [(5, 3, 1), (6, 4, 1), (7, 4, 2), (8, 5, 2), (9, 5, 3), (10, 6, 3)],
)
def test_role_counts(n, libs, fasc):
    counts = assign_role_counts(n)
    assert counts[RoleConst.LIBERAL] == libs
    assert counts[RoleConst.FASCIST] == fasc
    assert counts[RoleConst.HITLER] == 1


def test_powers_table():
    assert power_for(5, 3) == Power.POLICY_PEEK
    assert power_for(6, 4) == Power.EXECUTION
    assert power_for(5, 1) is None and power_for(5, 2) is None
    assert power_for(7, 2) == Power.INVESTIGATE
    assert power_for(8, 3) == Power.SPECIAL_ELECTION
    assert power_for(7, 4) == Power.EXECUTION and power_for(7, 5) == Power.EXECUTION
    assert power_for(9, 1) == Power.INVESTIGATE and power_for(10, 2) == Power.INVESTIGATE
    assert power_for(9, 3) == Power.SPECIAL_ELECTION
    # Policy Peek is unique to 5-6 player games.
    assert Power.POLICY_PEEK not in {power_for(7, k) for k in range(1, 6)}


def test_deck_composition_and_reshuffle():
    deck = PolicyDeck(seed=5)
    assert deck.draw_count == 17
    all_drawn = []
    while deck.draw_count >= 3:
        all_drawn.extend(deck.draw(3))
    assert all_drawn.count(PolicyColor.LIBERAL) <= 6
    # Reshuffle is triggered inside draw() when fewer than 3 remain.
    before = deck.draw_count
    deck.discard([PolicyColor.LIBERAL, PolicyColor.FASCIST, PolicyColor.FASCIST])
    drawn = deck.draw(3)  # forces reshuffle of discard pile
    assert len(drawn) == 3
    assert before < 3


def test_hitler_knows_fascists_by_count():
    assert hitler_knows_fascists(5) and hitler_knows_fascists(6)
    assert not hitler_knows_fascists(7) and not hitler_knows_fascists(10)


def test_board_enact_and_wins():
    board = Board()
    for _ in range(5):
        board.enact(PolicyColor.LIBERAL)
    assert board.liberal_policy_win()
    board2 = Board()
    for _ in range(6):
        board2.enact(PolicyColor.FASCIST)
    assert board2.fascist_policy_win()
    assert board2.veto_unlocked


def test_eligible_chancellors_term_limits():
    mod = build_moderator(7)
    pres = mod.state.president_id
    # No prior government: everyone except the president is eligible (no self-nomination).
    elig = mod.eligible_chancellors()
    assert pres not in elig
    assert set(elig) == set(mod.state.alive_player_ids()) - {pres}

    # With a prior elected government, last president and chancellor are term-limited (>5 alive).
    others = [p for p in mod.state.alive_player_ids() if p != pres]
    mod.state.last_president_id = others[0]
    mod.state.last_chancellor_id = others[1]
    elig = mod.eligible_chancellors()
    assert others[0] not in elig and others[1] not in elig


def test_eligible_chancellors_five_alive_only_chancellor_limited():
    mod = build_moderator(5)
    pres = mod.state.president_id
    others = [p for p in mod.state.alive_player_ids() if p != pres]
    mod.state.last_president_id = others[0]
    mod.state.last_chancellor_id = others[1]
    elig = mod.eligible_chancellors()
    # With <=5 alive, only the last chancellor is term-limited.
    assert others[1] not in elig
    assert others[0] in elig


# ------------------------------------------------------------------ #
# Full-game tests
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("n", [5, 7, 9, 10])
def test_game_completes(n):
    env = make("secret_hitler", debug=True, configuration={"agents": agents_config(n)})
    env.run(["deterministic"] * n)
    j = env.toJSON()
    assert j["statuses"] == ["DONE"] * n
    rewards = j["rewards"]
    assert all(r in (-1, 0, 1) for r in rewards)
    assert (1 in rewards and -1 in rewards) or all(r == 0 for r in rewards)


def test_random_games_terminate():
    for trial in range(8):
        random.seed(trial)
        n = random.choice([5, 7, 9, 10])
        env = make("secret_hitler", configuration={"agents": agents_config(n), "seed": trial})
        env.run(["random"] * n)
        assert set(env.toJSON()["statuses"]) == {"DONE"}


def test_determinism_with_seed():
    def run():
        env = make(
            "secret_hitler",
            configuration={"agents": agents_config(7), "randomize_roles": True, "seed": 42},
        )
        env.run(["deterministic"] * 7)
        roles = [s.observation.raw_observation["role"] for s in env.steps[-1]]
        return roles, env.toJSON()["rewards"]

    assert run() == run()


def test_renderer_nonempty():
    env = make("secret_hitler", configuration={"agents": agents_config(5)})
    env.run(["deterministic"] * 5)
    out = env.render(mode="ansi")
    assert isinstance(out, str) and len(out) > 0
    assert "Secret Hitler" in out


def _erroring_agent(obs):
    raise RuntimeError("boom")


def test_agent_error_aborts_game():
    env = make("secret_hitler", configuration={"agents": agents_config(5)})
    env.run([_erroring_agent] * 5)
    j = env.toJSON()
    # Any ERROR/INVALID/TIMEOUT aborts the whole game (werewolf-style); offender reward nulled.
    assert "ERROR" in j["statuses"]
    assert any(r is None for r in j["rewards"])
    assert all(s in ("DONE", "ERROR") for s in j["statuses"])


def test_malformed_action_is_tolerated():
    # A non-conforming action must not crash the engine; it is dropped and the engine defaults.
    def junk(obs):
        return {"action_type": "Nonsense", "kwargs": {}}

    env = make("secret_hitler", debug=True, configuration={"agents": agents_config(5)})
    env.run([junk] * 5)
    assert set(env.toJSON()["statuses"]) == {"DONE"}


# ------------------------------------------------------------------ #
# Scenario (engine-driven) tests
# ------------------------------------------------------------------ #
def test_hitler_chancellor_win():
    # Push 3 fascist policies, then elect Hitler as Chancellor -> Fascists win at election pass.
    mod = build_moderator(7, seed=3)
    mod.state.board.fascist_track = 3
    hitler = find_pid(mod, RoleConst.HITLER)

    def decide(mod, pid):
        # Always nominate Hitler (if eligible) and always vote Ja.
        return _scripted_action(mod, pid, vote=Vote.JA, nominate=hitler)

    # Ensure Hitler is eligible: not the sitting president and no term-limit on him.
    if mod.state.president_id == hitler:
        # advance president once by failing nothing — instead just pick a state where pres != hitler
        mod._marker = (mod._marker + 1) % mod.state.num_players
        mod.state.president_id = mod.state.all_player_ids[mod._marker]
        mod.state.nominee_chancellor_id = None

    drive(mod, decide, max_steps=200)
    assert mod.winner() == Team.FASCIST
    end = mod.state.get_event_by_name(EventName.GAME_END)[0]
    assert "Hitler" in end.data.reason


def test_hitler_executed_liberal_win():
    # Drive to an execution power and execute Hitler.
    mod = build_moderator(7, seed=2)
    hitler = find_pid(mod, RoleConst.HITLER)
    mod.state.board.fascist_track = 3  # next fascist policy (4th) grants Execution in 7p

    def decide(mod, pid):
        phase = DetailedPhase(mod.state.detailed_phase)
        if phase == DetailedPhase.EXECUTIVE_POWER_AWAIT and mod.pending_power == Power.EXECUTION:
            args = dict(round_number=mod.state.round_count, phase=mod.state.phase, actor_id=pid)
            if hitler in mod.current_power_targets():
                return A.ExecutionAction(**args, target_id=hitler)
        # Enact fascist policies: discard liberal when possible.
        return _scripted_action(mod, pid, vote=Vote.JA, discard_pref=PolicyColor.LIBERAL)

    drive(mod, decide, max_steps=400)
    assert mod.winner() == Team.LIBERAL
    assert not mod.state.get_player_by_id(hitler).alive


def test_chaos_after_three_failed_elections():
    mod = build_moderator(5, seed=4)

    def decide(mod, pid):
        return _scripted_action(mod, pid, vote=Vote.NEIN)  # everyone votes Nein -> elections fail

    # Pre-seed a prior government so we can confirm term limits get forgotten by chaos.
    mod.state.last_president_id = mod.state.alive_player_ids()[1]
    mod.state.last_chancellor_id = mod.state.alive_player_ids()[2]
    total_before = mod.state.board.liberal_track + mod.state.board.fascist_track

    # Step through exactly until a chaos enactment occurs.
    steps = 0
    while mod.state.board.election_tracker < 2 and not mod.is_game_over() and steps < 200:
        steps += 1
        active = mod.get_active_player_ids()
        mod.advance({pid: decide(mod, pid) for pid in active})
    # One more failed election triggers chaos.
    while not mod.is_game_over() and steps < 400:
        steps += 1
        active = mod.get_active_player_ids()
        mod.advance({pid: decide(mod, pid) for pid in active})
        chaos = mod.state.get_event_by_name(EventName.CHAOS_ENACTED)
        if chaos:
            break
    assert mod.state.get_event_by_name(EventName.CHAOS_ENACTED), "chaos never triggered"
    total_after = mod.state.board.liberal_track + mod.state.board.fascist_track
    assert total_after == total_before + 1  # the top tile was auto-enacted
    assert mod.state.board.election_tracker == 0  # tracker reset
    assert mod.state.last_president_id is None and mod.state.last_chancellor_id is None  # term limits forgotten


def test_discussion_only_in_discussion_phases():
    # Discussion (speech) events occur only in the two round-robin phases: the pre-vote
    # discussion (ELECTION_CHAT_AWAIT) and the post-policy debrief (LEGISLATIVE_DEBRIEF_AWAIT).
    mod = build_moderator(7, seed=6, rounds=2)
    drive(mod, lambda m, pid: _scripted_action(m, pid, vote=Vote.JA, discard_pref=PolicyColor.FASCIST))
    for events in mod.state.history.values():
        for e in events:
            if e.event_name == EventName.DISCUSSION:
                assert e.detailed_phase in (
                    DetailedPhase.ELECTION_CHAT_AWAIT,
                    DetailedPhase.LEGISLATIVE_DEBRIEF_AWAIT,
                ), e.detailed_phase


def test_no_reinvestigation():
    mod = build_moderator(9, seed=8)  # 9p: 1st & 2nd fascist policy grant Investigate
    investigated = []

    def decide(mod, pid):
        phase = DetailedPhase(mod.state.detailed_phase)
        if phase == DetailedPhase.EXECUTIVE_POWER_AWAIT and mod.pending_power == Power.INVESTIGATE:
            targets = mod.current_power_targets()
            # No already-investigated player may be a valid target.
            assert all(t not in mod.state.investigated_ids for t in targets)
            args = dict(round_number=mod.state.round_count, phase=mod.state.phase, actor_id=pid)
            investigated.append(targets[0])
            return A.InvestigateAction(**args, target_id=targets[0])
        return _scripted_action(mod, pid, vote=Vote.JA, discard_pref=PolicyColor.LIBERAL)

    drive(mod, decide, max_steps=600)
    # If two investigations happened they must be distinct players.
    assert len(investigated) == len(set(investigated))


def test_win_check_precedes_power():
    # The 6th fascist policy lands on the 5th-fascist execution square; the game must end and
    # NOT trigger an execution.
    mod = build_moderator(7, seed=9)
    mod.state.board.fascist_track = 5  # next fascist enacted is the 6th -> fascist policy win

    def decide(mod, pid):
        return _scripted_action(mod, pid, vote=Vote.JA, discard_pref=PolicyColor.LIBERAL)

    drive(mod, decide, max_steps=200)
    assert mod.winner() == Team.FASCIST
    assert not mod.state.get_event_by_name(EventName.EXECUTION)  # no execution fired


def test_no_hidden_info_leak_in_observations():
    from kaggle_environments.envs.secret_hitler.secret_hitler import _known_allies

    env = make("secret_hitler", configuration={"agents": agents_config(7), "seed": 11})
    env.run(["deterministic"] * 7)
    gs = env.game_state
    # A liberal must have no known allies, and must never receive another player's role event.
    for player in gs.players:
        allies = _known_allies(gs, player)
        if player.role.name == RoleConst.LIBERAL:
            assert allies == {}
        for ally_id, ally_role in allies.items():
            # Allies are only ever fascist-team members.
            assert gs.get_player_by_id(ally_id).role.team == Team.FASCIST
    # Hitler in a 7-player game knows nobody.
    hitler = next(p for p in gs.players if p.role.name == RoleConst.HITLER)
    assert _known_allies(gs, hitler) == {}
