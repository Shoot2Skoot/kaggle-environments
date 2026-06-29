import json
import logging
from typing import Dict, List, Optional, Type

from .actions import (
    Action,
    BidAction,
    ChatAction,
    DiscardPolicyAction,
    ExecutionAction,
    InvestigateAction,
    NominateChancellorAction,
    PolicyPeekAckAction,
    SpecialElectionAction,
    VetoConsentAction,
    VoteAction,
)
from .base import BaseModerator, PlayerID
from .board import power_for
from .consts import (
    DetailedPhase,
    EventName,
    Phase,
    PolicyColor,
    Power,
    RoleConst,
    Team,
    Vote,
)
from .protocols.base import DiscussionProtocol
from .records import (
    BidDataEntry,
    BidResultDataEntry,
    BoardSnapshotDataEntry,
    ChaosDataEntry,
    ChatDataEntry,
    DiscardActionDataEntry,
    ElectionTrackerDataEntry,
    ExecutionResultDataEntry,
    GameEndResultsDataEntry,
    GameStartDataEntry,
    GameStartRoleDataEntry,
    InvestigationResultDataEntry,
    NominationDataEntry,
    PolicyEnactedDataEntry,
    PowerUsedDataEntry,
    RequestBidDataEntry,
    RequestChatDataEntry,
    RequestDiscardDataEntry,
    RequestNominationDataEntry,
    RequestPowerDataEntry,
    RequestVetoConsentDataEntry,
    RequestVoteDataEntry,
    VetoResultDataEntry,
    VoteDataEntry,
    VoteResultDataEntry,
)
from .roles import assign_role_counts, hitler_knows_fascists
from .states import GameState

logger = logging.getLogger(__name__)

_HAND_KEY = "policy_hand"
_INVESTIGATIONS_KEY = "investigations"

_MAX_ADVANCE_ITERATIONS = 10_000  # watchdog against an FSM livelock (plan invariant #1)


class ActionQueue:
    """Tracks which players the moderator is currently awaiting an action from."""

    def __init__(self):
        self._queue: Dict[str, List[PlayerID]] = {}

    def clear(self):
        self._queue = {}

    def append(self, action_cls: Type[Action], player_id: PlayerID):
        key = action_cls.__name__
        self._queue.setdefault(key, [])
        if player_id not in self._queue[key]:
            self._queue[key].append(player_id)

    def get_active_player_ids(self) -> List[PlayerID]:
        players = []
        for ids in self._queue.values():
            for pid in ids:
                if pid not in players:
                    players.append(pid)
        return players


class Moderator(BaseModerator):
    def __init__(self, state: GameState, discussion: DiscussionProtocol):
        self._state = state
        self.discussion = discussion
        self._queue = ActionQueue()

        self._winner: Optional[Team] = None
        self._win_reason: str = ""
        self._finalized = False

        # Per-round transient state.
        self._marker: int = 0
        self._pending_override: Optional[PlayerID] = None
        self._pending_power: Optional[Power] = None
        self._enacted_color: Optional[PolicyColor] = None
        self._veto_available: bool = True
        self._votes: Dict[PlayerID, Vote] = {}

        self._handlers = {
            DetailedPhase.GAME_START: self._h_game_start,
            DetailedPhase.ELECTION_NOMINATION_AWAIT: self._h_nomination,
            DetailedPhase.ELECTION_BIDDING_AWAIT: self._h_bidding,
            DetailedPhase.ELECTION_CHAT_AWAIT: self._h_chat,
            DetailedPhase.ELECTION_VOTE_AWAIT: self._h_vote,
            DetailedPhase.ELECTION_CONCLUDE: self._h_election_conclude,
            DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT: self._h_president_discard,
            DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT: self._h_chancellor_discard,
            DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT: self._h_veto_consent,
            DetailedPhase.LEGISLATIVE_CONCLUDE: self._h_legislative_conclude,
            DetailedPhase.EXECUTIVE_POWER_AWAIT: self._h_power,
            DetailedPhase.EXECUTIVE_CONCLUDE: self._h_executive_conclude,
        }

    # ------------------------------------------------------------------ #
    # Framework interface
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> GameState:
        return self._state

    def request_action(self, action_cls, player_id, prompt, data=None, event_name=EventName.MODERATOR_ANNOUNCEMENT):
        self._queue.append(action_cls, player_id)
        self._state.push_event(
            description=prompt,
            event_name=event_name,
            public=False,
            visible_to=[player_id],
            data=data,
        )

    def get_active_player_ids(self) -> List[PlayerID]:
        return self._queue.get_active_player_ids()

    @property
    def pending_power(self) -> Optional[Power]:
        return self._pending_power

    @property
    def veto_available(self) -> bool:
        return self._veto_available

    def current_power_targets(self) -> List[PlayerID]:
        if self._pending_power is None:
            return []
        return self._power_request(self._pending_power)[1]

    def winner(self) -> Optional[Team]:
        return self._winner

    def is_game_over(self) -> bool:
        return self._winner is not None or self._state.detailed_phase == DetailedPhase.GAME_OVER

    def advance(self, player_actions: Dict[PlayerID, Action]):
        if self._state.detailed_phase == DetailedPhase.GAME_OVER:
            return
        self._call_handler(player_actions)
        iterations = 0
        while not self.get_active_player_ids() and not self.is_game_over():
            iterations += 1
            if iterations > _MAX_ADVANCE_ITERATIONS:
                raise RuntimeError(
                    f"FSM livelock in phase {self._state.detailed_phase}: no active players and no winner."
                )
            self._call_handler({})
        if self.is_game_over() and not self._finalized:
            self._finalize_game()

    def _call_handler(self, player_actions: Dict[PlayerID, Action]):
        handler = self._handlers[self._state.detailed_phase]
        next_phase = handler(player_actions)
        self._state.detailed_phase = next_phase
        self._state.phase = next_phase.category

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _index_of(self, pid: PlayerID) -> int:
        return self._state.all_player_ids.index(pid)

    def _next_alive_index(self, idx: int) -> int:
        n = self._state.num_players
        for step in range(1, n + 1):
            cand = (idx + step) % n
            if self._state.players[cand].alive:
                return cand
        return idx

    def _president(self):
        return self._state.get_player_by_id(self._state.president_id)

    def _chancellor(self):
        return self._state.get_player_by_id(self._state.chancellor_id)

    def _pending(self) -> bool:
        return bool(self._queue.get_active_player_ids())

    def eligible_chancellors(self) -> List[PlayerID]:
        alive = self._state.alive_player_ids()
        president = self._state.president_id
        candidates = [p for p in alive if p != president]  # no self-nomination
        term_limited = set()
        if self._state.last_chancellor_id:
            term_limited.add(self._state.last_chancellor_id)
        if self._state.num_alive > 5 and self._state.last_president_id:
            term_limited.add(self._state.last_president_id)
        eligible = [p for p in candidates if p not in term_limited]
        if not eligible:  # defensive: never deadlock
            eligible = candidates
        return eligible

    def _board_snapshot(self):
        board = self._state.board
        data = BoardSnapshotDataEntry(
            liberal_policies=board.liberal_track,
            fascist_policies=board.fascist_track,
            election_tracker=board.election_tracker,
            veto_unlocked=board.veto_unlocked,
            president_id=self._state.president_id,
            chancellor_id=self._state.chancellor_id,
            draw_pile_count=self._state.deck.draw_count,
            discard_pile_count=self._state.deck.discard_count,
            alive_player_ids=self._state.alive_player_ids(),
        )
        # public=False, visible_to=[] => reaches the global queue (MODERATOR_OBS / spectator) but
        # pollutes no player's prompt queue.
        self._state.push_event(
            description="Board state",
            event_name=EventName.BOARD_SNAPSHOT,
            public=False,
            visible_to=[],
            data=data,
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _h_game_start(self, _actions):
        state = self._state
        num_players = state.num_players
        role_counts = assign_role_counts(num_players)
        team_counts = {
            Team.LIBERAL: role_counts[RoleConst.LIBERAL],
            Team.FASCIST: role_counts[RoleConst.FASCIST] + role_counts[RoleConst.HITLER],
        }
        state.push_event(
            description=(
                f"A game of Secret Hitler begins with {num_players} players: "
                f"{role_counts[RoleConst.LIBERAL]} Liberals, {role_counts[RoleConst.FASCIST]} "
                f"Fascists, and 1 Hitler."
            ),
            event_name=EventName.GAME_START,
            public=True,
            data=GameStartDataEntry(
                player_ids=state.all_player_ids,
                number_of_players=num_players,
                role_counts=role_counts,
                team_member_counts=team_counts,
                discussion_protocol_name=self.discussion.display_name,
                discussion_protocol_rule=self.discussion.rule,
            ),
        )

        # Private role + allies knowledge.
        fascists = state.get_players_by_role(RoleConst.FASCIST)
        hitler = state.get_players_by_role(RoleConst.HITLER)
        fascist_team_ids = {p.id: p.role.name for p in fascists + hitler}
        for player in state.players:
            known_allies: Dict[PlayerID, RoleConst] = {}
            if player.role.team == Team.FASCIST:
                if player.role.name == RoleConst.HITLER:
                    if hitler_knows_fascists(num_players):
                        known_allies = {pid: r for pid, r in fascist_team_ids.items() if pid != player.id}
                else:  # regular Fascist sees all teammates + Hitler
                    known_allies = {pid: r for pid, r in fascist_team_ids.items() if pid != player.id}
            state.push_event(
                description=f"You are {player.role.name}. {player.role.descriptions}",
                event_name=EventName.GAME_START_ROLE,
                public=False,
                visible_to=[player.id],
                visible_in_ui=False,
                data=GameStartRoleDataEntry(
                    player_id=player.id,
                    team=player.role.team,
                    role=player.role.name,
                    known_allies=known_allies,
                    rule_of_role=player.role.descriptions,
                ),
            )

        # First President: seeded deterministic seat.
        state.president_id = state.all_player_ids[self._marker]
        state.round_count = 1
        self._board_snapshot()
        return DetailedPhase.ELECTION_NOMINATION_AWAIT

    # ------------------------------------------------------------------ #
    # Election
    # ------------------------------------------------------------------ #
    def _h_nomination(self, actions):
        state = self._state
        if not self._pending():
            eligible = self.eligible_chancellors()
            self.request_action(
                NominateChancellorAction,
                state.president_id,
                prompt=(f"President {state.president_id}, nominate a Chancellor. Eligible candidates: {eligible}."),
                data=RequestNominationDataEntry(
                    valid_targets=eligible,
                    action_json_schema=json.dumps(NominateChancellorAction.schema_for_player()),
                ),
                event_name=EventName.NOMINATION_REQUEST,
            )
            return DetailedPhase.ELECTION_NOMINATION_AWAIT

        self._queue.clear()
        eligible = self.eligible_chancellors()
        action = actions.get(state.president_id)
        nominee = action.target_id if isinstance(action, NominateChancellorAction) else None
        if nominee not in eligible:
            nominee = eligible[0]
            logger.warning(
                f"President {state.president_id} nominated an ineligible chancellor; defaulting to {nominee}."
            )
        state.nominee_chancellor_id = nominee
        state.push_event(
            description=f"President {state.president_id} nominates {nominee} as Chancellor.",
            event_name=EventName.NOMINATION,
            public=True,
            source=state.president_id,
            data=NominationDataEntry(
                actor_id=state.president_id,
                target_id=nominee,
                reasoning=getattr(action, "reasoning", None),
                action=action,
            ),
        )
        self.discussion.begin(state.alive_player_ids())
        return DetailedPhase.ELECTION_BIDDING_AWAIT

    def _h_bidding(self, actions):
        state = self._state
        if not self._pending():
            if self.discussion.is_finished():
                return DetailedPhase.ELECTION_VOTE_AWAIT
            bidders = self.discussion.bidders(state.alive_player_ids())
            if not bidders:
                return DetailedPhase.ELECTION_VOTE_AWAIT
            for pid in bidders:
                self.request_action(
                    BidAction,
                    pid,
                    prompt=f"Bid 0-{self.discussion.max_bid} for the floor to speak about the proposed government.",
                    data=RequestBidDataEntry(
                        max_bid=self.discussion.max_bid,
                        action_json_schema=json.dumps(BidAction.schema_for_player()),
                    ),
                    event_name=EventName.BID_REQUEST,
                )
            return DetailedPhase.ELECTION_BIDDING_AWAIT

        self._queue.clear()
        bids: Dict[PlayerID, int] = {}
        for pid in state.alive_player_ids():
            action = actions.get(pid)
            amount = action.amount if isinstance(action, BidAction) else 0
            amount = max(0, min(amount, self.discussion.max_bid))
            bids[pid] = amount
            if isinstance(action, BidAction):
                state.push_event(
                    description=f"{pid} placed a bid.",
                    event_name=EventName.BID_ACTION,
                    public=False,
                    visible_to=[pid],
                    source=pid,
                    data=BidDataEntry(actor_id=pid, bid_amount=amount, reasoning=action.reasoning, action=action),
                )
        speakers = self.discussion.resolve_bids(bids)
        state.push_event(
            description=f"Bidding result: speakers={speakers}.",
            event_name=EventName.BID_RESULT,
            public=True,
            data=BidResultDataEntry(winner_player_ids=speakers, bid_overview=bids),
        )
        self._next_speakers = speakers
        if not speakers:
            return DetailedPhase.ELECTION_VOTE_AWAIT
        return DetailedPhase.ELECTION_CHAT_AWAIT

    def _h_chat(self, actions):
        state = self._state
        if not self._pending():
            speakers = getattr(self, "_next_speakers", [])
            if not speakers:
                return DetailedPhase.ELECTION_VOTE_AWAIT
            for pid in speakers:
                self.request_action(
                    ChatAction,
                    pid,
                    prompt="You won the floor. Address the table about the proposed government.",
                    data=RequestChatDataEntry(action_json_schema=json.dumps(ChatAction.schema_for_player())),
                    event_name=EventName.CHAT_REQUEST,
                )
            return DetailedPhase.ELECTION_CHAT_AWAIT

        self._queue.clear()
        speakers = getattr(self, "_next_speakers", [])
        for pid in speakers:
            action = actions.get(pid)
            message = action.message if isinstance(action, ChatAction) else ""
            state.push_event(
                description=f"{pid}: {message}",
                event_name=EventName.DISCUSSION,
                public=True,
                source=pid,
                data=ChatDataEntry(
                    actor_id=pid, message=message, reasoning=getattr(action, "reasoning", None), action=action
                ),
            )
        self.discussion.record_speech(speakers)
        if self.discussion.is_finished():
            return DetailedPhase.ELECTION_VOTE_AWAIT
        return DetailedPhase.ELECTION_BIDDING_AWAIT

    def _h_vote(self, actions):
        state = self._state
        if not self._pending():
            for pid in state.alive_player_ids():
                self.request_action(
                    VoteAction,
                    pid,
                    prompt=(
                        f"Vote Ja or Nein on the government: President {state.president_id}, "
                        f"Chancellor {state.nominee_chancellor_id}."
                    ),
                    data=RequestVoteDataEntry(
                        president_id=state.president_id,
                        chancellor_nominee_id=state.nominee_chancellor_id,
                        action_json_schema=json.dumps(VoteAction.schema_for_player()),
                    ),
                    event_name=EventName.VOTE_REQUEST,
                )
            return DetailedPhase.ELECTION_VOTE_AWAIT

        self._queue.clear()
        self._votes = {}
        for pid in state.alive_player_ids():
            action = actions.get(pid)
            vote = action.vote if isinstance(action, VoteAction) else Vote.NEIN
            self._votes[pid] = vote
            state.push_event(
                description=f"{pid} voted {vote}.",
                event_name=EventName.VOTE_ACTION,
                public=True,
                source=pid,
                data=VoteDataEntry(
                    actor_id=pid, vote=vote, reasoning=getattr(action, "reasoning", None), action=action
                ),
            )
        return DetailedPhase.ELECTION_CONCLUDE

    def _h_election_conclude(self, _actions):
        state = self._state
        ja = [pid for pid, v in self._votes.items() if v == Vote.JA]
        nein = [pid for pid, v in self._votes.items() if v == Vote.NEIN]
        passed = len(ja) > len(nein)  # tie fails
        nominee = state.nominee_chancellor_id
        state.push_event(
            description=f"Election {'passed' if passed else 'failed'}. Ja: {ja}. Nein: {nein}.",
            event_name=EventName.ELECTION_RESULT,
            public=True,
            data=VoteResultDataEntry(
                ja_voters=ja,
                nein_voters=nein,
                passed=passed,
                president_id=state.president_id,
                chancellor_id=nominee,
            ),
        )
        if passed:
            state.chancellor_id = nominee
            # Hitler-elected-Chancellor win, checked at election pass, gated on fascist_track >= 3.
            chancellor = self._chancellor()
            if state.board.fascist_track >= 3 and chancellor and chancellor.role.name == RoleConst.HITLER:
                self._latch_winner(Team.FASCIST, "Hitler was elected Chancellor with 3+ Fascist policies enacted.")
                return DetailedPhase.GAME_OVER
            state.board.reset_tracker()
            state.push_event(
                description=f"Government elected: President {state.president_id}, Chancellor {nominee}.",
                event_name=EventName.GOVERNMENT_ELECTED,
                public=True,
            )
            return DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT

        state.nominee_chancellor_id = None
        chaos = self._advance_tracker_and_maybe_chaos("Election failed.")
        if self.is_game_over():
            return DetailedPhase.GAME_OVER
        _ = chaos
        return self._start_new_round()

    # ------------------------------------------------------------------ #
    # Legislative
    # ------------------------------------------------------------------ #
    def _h_president_discard(self, actions):
        state = self._state
        president = self._president()
        if not self._pending():
            hand = state.deck.draw(3)
            president.set_role_state(_HAND_KEY, list(hand))
            self.request_action(
                DiscardPolicyAction,
                president.id,
                prompt=f"You drew 3 policies: {[c.value for c in hand]}. Discard one; pass the other two to the Chancellor.",
                data=RequestDiscardDataEntry(
                    policy_hand=hand,
                    veto_available=False,
                    action_json_schema=json.dumps(DiscardPolicyAction.schema_for_player()),
                ),
                event_name=EventName.DISCARD_REQUEST,
            )
            return DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT

        self._queue.clear()
        hand: List[PolicyColor] = list(president.get_role_state(_HAND_KEY) or [])
        action = actions.get(president.id)
        discard = action.discard if isinstance(action, DiscardPolicyAction) else None
        discard = self._validate_discard(discard, hand)
        hand.remove(discard)
        state.deck.discard([discard])
        self._chancellor().set_role_state(_HAND_KEY, list(hand))
        president.set_role_state(_HAND_KEY, None)
        state.push_event(
            description=f"President {president.id} discarded a policy and passed two to the Chancellor.",
            event_name=EventName.DISCARD_ACTION,
            public=True,
            source=president.id,
            data=DiscardActionDataEntry(
                actor_id=president.id, discard=discard, reasoning=getattr(action, "reasoning", None), action=action
            ),
        )
        self._veto_available = state.board.veto_unlocked
        return DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT

    def _h_chancellor_discard(self, actions):
        state = self._state
        chancellor = self._chancellor()
        if not self._pending():
            hand: List[PolicyColor] = list(chancellor.get_role_state(_HAND_KEY) or [])
            self.request_action(
                DiscardPolicyAction,
                chancellor.id,
                prompt=(
                    f"You received 2 policies: {[c.value for c in hand]}. Discard one and enact the other."
                    + (" You may instead propose a veto." if self._veto_available else "")
                ),
                data=RequestDiscardDataEntry(
                    policy_hand=hand,
                    veto_available=self._veto_available,
                    action_json_schema=json.dumps(DiscardPolicyAction.schema_for_chancellor()),
                ),
                event_name=EventName.DISCARD_REQUEST,
            )
            return DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT

        self._queue.clear()
        hand = list(chancellor.get_role_state(_HAND_KEY) or [])
        action = actions.get(chancellor.id)
        if self._veto_available and isinstance(action, DiscardPolicyAction) and action.veto:
            state.push_event(
                description=f"Chancellor {chancellor.id} proposes to veto this agenda.",
                event_name=EventName.VETO_PROPOSED,
                public=True,
                source=chancellor.id,
                data=DiscardActionDataEntry(
                    actor_id=chancellor.id,
                    veto_proposed=True,
                    reasoning=getattr(action, "reasoning", None),
                    action=action,
                ),
            )
            return DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT

        discard = action.discard if isinstance(action, DiscardPolicyAction) else None
        discard = self._validate_discard(discard, hand)
        hand.remove(discard)
        state.deck.discard([discard])
        enacted = hand[0]
        chancellor.set_role_state(_HAND_KEY, None)
        self._enacted_color = enacted
        state.push_event(
            description=f"Chancellor {chancellor.id} discarded a policy and enacted a policy.",
            event_name=EventName.DISCARD_ACTION,
            public=True,
            source=chancellor.id,
            data=DiscardActionDataEntry(
                actor_id=chancellor.id, discard=discard, reasoning=getattr(action, "reasoning", None), action=action
            ),
        )
        return DetailedPhase.LEGISLATIVE_CONCLUDE

    def _h_veto_consent(self, actions):
        state = self._state
        president = self._president()
        if not self._pending():
            self.request_action(
                VetoConsentAction,
                president.id,
                prompt=f"Chancellor {state.chancellor_id} proposes a veto. Do you consent? (consent: true/false)",
                data=RequestVetoConsentDataEntry(
                    chancellor_id=state.chancellor_id,
                    action_json_schema=json.dumps(VetoConsentAction.schema_for_player()),
                ),
                event_name=EventName.VETO_CONSENT_REQUEST,
            )
            return DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT

        self._queue.clear()
        action = actions.get(president.id)
        consent = bool(action.consent) if isinstance(action, VetoConsentAction) else False
        state.push_event(
            description=f"President {president.id} {'consented to' if consent else 'refused'} the veto.",
            event_name=EventName.VETO_RESULT,
            public=True,
            source=president.id,
            data=VetoResultDataEntry(consented=consent),
        )
        if consent:
            chancellor = self._chancellor()
            leftover = list(chancellor.get_role_state(_HAND_KEY) or [])
            state.deck.discard(leftover)
            chancellor.set_role_state(_HAND_KEY, None)
            self._advance_tracker_and_maybe_chaos("Veto enacted (no policy).")
            if self.is_game_over():
                return DetailedPhase.GAME_OVER
            return self._start_new_round()

        # Refusal: chancellor must now enact, with no veto option.
        self._veto_available = False
        return DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT

    def _h_legislative_conclude(self, _actions):
        state = self._state
        color = self._enacted_color
        state.board.enact(color)
        state.board.reset_tracker()
        state.last_president_id = state.president_id
        state.last_chancellor_id = state.chancellor_id
        state.push_event(
            description=f"A {color} policy was enacted.",
            event_name=EventName.POLICY_ENACTED,
            public=True,
            data=PolicyEnactedDataEntry(
                policy_color=color,
                liberal_policies=state.board.liberal_track,
                fascist_policies=state.board.fascist_track,
                president_id=state.president_id,
                chancellor_id=state.chancellor_id,
            ),
        )
        # Win-check precedes power-grant.
        if self._check_policy_win():
            return DetailedPhase.GAME_OVER
        if color == PolicyColor.FASCIST:
            power = power_for(state.num_players, state.board.fascist_track)
            if power is not None:
                self._pending_power = power
                return DetailedPhase.EXECUTIVE_POWER_AWAIT
        return self._start_new_round()

    # ------------------------------------------------------------------ #
    # Executive
    # ------------------------------------------------------------------ #
    def _h_power(self, actions):
        president = self._president()
        power = self._pending_power
        if not self._pending():
            action_cls, targets = self._power_request(power)
            self.request_action(
                action_cls,
                president.id,
                prompt=self._power_prompt(power, targets),
                data=RequestPowerDataEntry(
                    power=power,
                    valid_targets=targets,
                    action_json_schema=json.dumps(action_cls.schema_for_player()),
                ),
                event_name=EventName.POWER_REQUEST,
            )
            return DetailedPhase.EXECUTIVE_POWER_AWAIT

        self._queue.clear()
        action = actions.get(president.id)
        self._apply_power(power, president, action)
        self._pending_power = None
        if self.is_game_over():
            return DetailedPhase.GAME_OVER
        return DetailedPhase.EXECUTIVE_CONCLUDE

    def _h_executive_conclude(self, _actions):
        if self.is_game_over():
            return DetailedPhase.GAME_OVER
        return self._start_new_round()

    def _power_request(self, power: Power):
        state = self._state
        president = self._state.president_id
        alive = [p for p in state.alive_player_ids() if p != president]
        if power == Power.INVESTIGATE:
            targets = [p for p in alive if p not in state.investigated_ids]
            return InvestigateAction, targets
        if power == Power.SPECIAL_ELECTION:
            return SpecialElectionAction, alive
        if power == Power.EXECUTION:
            return ExecutionAction, alive
        return PolicyPeekAckAction, []

    def _power_prompt(self, power: Power, targets) -> str:
        if power == Power.POLICY_PEEK:
            top = self._state.deck.peek(3)
            return f"Policy Peek: the top three policies are {[c.value for c in top]}. Acknowledge to continue."
        return f"Use your {power} power. Valid targets: {targets}."

    def _apply_power(self, power: Power, president, action):
        state = self._state
        if power == Power.POLICY_PEEK:
            top = state.deck.peek(3)
            state.push_event(
                description=f"You peeked at the top three policies: {[c.value for c in top]}.",
                event_name=EventName.POLICY_PEEK,
                public=False,
                visible_to=[president.id],
                source=president.id,
                visible_in_ui=False,
            )
            state.push_event(
                description=f"President {president.id} used Policy Peek.",
                event_name=EventName.POWER_USED,
                public=True,
                source=president.id,
                data=PowerUsedDataEntry(
                    actor_id=president.id, power=power, reasoning=getattr(action, "reasoning", None)
                ),
            )
            return

        target = getattr(action, "target_id", None)
        valid = self._power_request(power)[1]
        if target not in valid:
            target = valid[0] if valid else None
            logger.warning(f"President {president.id} chose an invalid {power} target; defaulting to {target}.")

        if power == Power.INVESTIGATE:
            target_player = state.get_player_by_id(target)
            membership = Team.FASCIST if target_player.role.team == Team.FASCIST else Team.LIBERAL
            state.mark_investigated(target)
            investigations = dict(president.get_role_state(_INVESTIGATIONS_KEY) or {})
            investigations[target] = membership
            president.set_role_state(_INVESTIGATIONS_KEY, investigations)
            state.push_event(
                description=f"You investigated {target}: their party membership is {membership}.",
                event_name=EventName.INVESTIGATE_RESULT,
                public=False,
                visible_to=[president.id],
                source=president.id,
                visible_in_ui=False,
                data=InvestigationResultDataEntry(actor_id=president.id, target_id=target, membership=membership),
            )
            state.push_event(
                description=f"President {president.id} investigated {target}'s party membership.",
                event_name=EventName.POWER_USED,
                public=True,
                source=president.id,
                data=PowerUsedDataEntry(
                    actor_id=president.id, power=power, target_id=target, reasoning=getattr(action, "reasoning", None)
                ),
            )
        elif power == Power.SPECIAL_ELECTION:
            self._marker = self._index_of(president.id)
            self._pending_override = target
            state.push_event(
                description=f"President {president.id} calls a Special Election: {target} will be the next President.",
                event_name=EventName.POWER_USED,
                public=True,
                source=president.id,
                data=PowerUsedDataEntry(
                    actor_id=president.id, power=power, target_id=target, reasoning=getattr(action, "reasoning", None)
                ),
            )
        elif power == Power.EXECUTION:
            state.eliminate_player(target)
            state.push_event(
                description=f"President {president.id} executed {target}.",
                event_name=EventName.EXECUTION,
                public=True,
                source=president.id,
                data=ExecutionResultDataEntry(executed_player_id=target),
            )
            state.push_event(
                description=f"President {president.id} used Execution.",
                event_name=EventName.POWER_USED,
                public=True,
                source=president.id,
                data=PowerUsedDataEntry(
                    actor_id=president.id, power=power, target_id=target, reasoning=getattr(action, "reasoning", None)
                ),
            )
            target_player = state.get_player_by_id(target)
            if target_player and target_player.role.name == RoleConst.HITLER:
                self._latch_winner(Team.LIBERAL, "Hitler was executed.")

    # ------------------------------------------------------------------ #
    # Shared resolution helpers
    # ------------------------------------------------------------------ #
    def _validate_discard(self, discard, hand: List[PolicyColor]) -> PolicyColor:
        if discard in hand:
            return discard
        logger.warning(f"Invalid discard {discard} for hand {hand}; defaulting to first tile.")
        return hand[0]

    def _advance_tracker_and_maybe_chaos(self, reason: str) -> bool:
        state = self._state
        chaos = state.board.advance_tracker()
        state.push_event(
            description=f"{reason} Election tracker is now {state.board.election_tracker}.",
            event_name=EventName.ELECTION_TRACKER,
            public=True,
            data=ElectionTrackerDataEntry(election_tracker=state.board.election_tracker, reason=reason),
        )
        if chaos:
            self._enact_chaos()
        return chaos

    def _enact_chaos(self):
        state = self._state
        tile = state.deck.draw(1)[0]
        state.board.enact(tile)
        state.board.reset_tracker()
        # Chaos forgets all term limits.
        state.last_president_id = None
        state.last_chancellor_id = None
        state.push_event(
            description=f"The country is in chaos! The top policy ({tile}) is enacted automatically.",
            event_name=EventName.CHAOS_ENACTED,
            public=True,
            data=ChaosDataEntry(policy_color=tile),
        )
        state.push_event(
            description=f"A {tile} policy was enacted by chaos.",
            event_name=EventName.POLICY_ENACTED,
            public=True,
            data=PolicyEnactedDataEntry(
                policy_color=tile,
                liberal_policies=state.board.liberal_track,
                fascist_policies=state.board.fascist_track,
                by_chaos=True,
            ),
        )
        self._check_policy_win()  # chaos can win; grants no power.

    def _check_policy_win(self) -> bool:
        board = self._state.board
        if board.liberal_policy_win():
            self._latch_winner(Team.LIBERAL, "Five Liberal policies were enacted.")
            return True
        if board.fascist_policy_win():
            self._latch_winner(Team.FASCIST, "Six Fascist policies were enacted.")
            return True
        return False

    def _latch_winner(self, team: Team, reason: str):
        if self._winner is None:
            self._winner = team
            self._win_reason = reason

    def _start_new_round(self) -> DetailedPhase:
        state = self._state
        self._advance_president()
        state.round_count += 1
        state.chancellor_id = None
        state.nominee_chancellor_id = None
        self._enacted_color = None
        self._veto_available = True
        self._board_snapshot()
        return DetailedPhase.ELECTION_NOMINATION_AWAIT

    def _advance_president(self):
        state = self._state
        if self._pending_override is not None:
            state.president_id = self._pending_override
            self._pending_override = None
            return
        self._marker = self._next_alive_index(self._marker)
        state.president_id = state.all_player_ids[self._marker]

    # ------------------------------------------------------------------ #
    # Finalize
    # ------------------------------------------------------------------ #
    def _finalize_game(self):
        state = self._state
        winner_team = self._winner
        winner_ids, loser_ids, scores = [], [], {}
        for player in state.players:
            if winner_team is not None and player.role.team == winner_team:
                winner_ids.append(player.id)
                scores[player.id] = 1
            elif winner_team is not None:
                loser_ids.append(player.id)
                scores[player.id] = -1
            else:
                scores[player.id] = 0
        state.push_event(
            description=f"Game over. Winner: {winner_team}. {self._win_reason}",
            event_name=EventName.GAME_END,
            public=True,
            data=GameEndResultsDataEntry(
                winner_team=winner_team,
                winner_ids=winner_ids,
                loser_ids=loser_ids,
                scores=scores,
                reason=self._win_reason,
                last_round=state.round_count,
                all_players_and_role={p.id: p.role.name for p in state.players},
            ),
        )
        self._board_snapshot()
        state.detailed_phase = DetailedPhase.GAME_OVER
        state.phase = Phase.GAME_OVER
        self._finalized = True

    def force_timeout_finalize(self):
        """Called by the interpreter if the episode hits its step cap without a winner."""
        if self._finalized:
            return
        self._winner = None
        self._win_reason = "Episode step limit reached with no winner."
        self._finalize_game()
