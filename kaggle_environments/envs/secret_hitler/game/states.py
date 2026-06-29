import logging
from collections import defaultdict, deque
from functools import cached_property
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

from pydantic import ConfigDict, Field, PrivateAttr, computed_field

from .base import BaseState, PlayerID
from .board import Board
from .consts import MODERATOR_ID, DetailedPhase, EventName, Phase, PhaseDivider, RoleConst, Team
from .deck import PolicyDeck
from .records import (
    DataAccessLevel,
    DataEntry,
    Event,
    PhaseDividerDataEntry,
    PlayerEventView,
)
from .roles import Player

logger = logging.getLogger(__name__)


class GameState(BaseState):
    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    players: List[Player]
    phase: Phase = Phase.ELECTION
    detailed_phase: DetailedPhase = DetailedPhase.GAME_START
    round_count: int = 0
    history: Dict[int, List[Event]] = Field(default_factory=dict)

    # Government pointers.
    president_id: Optional[PlayerID] = None
    chancellor_id: Optional[PlayerID] = None
    nominee_chancellor_id: Optional[PlayerID] = None
    last_president_id: Optional[PlayerID] = None
    last_chancellor_id: Optional[PlayerID] = None
    special_election_return_id: Optional[PlayerID] = None

    board: Board = Field(default_factory=Board)
    deck: PolicyDeck = None

    _id_to_player: Dict[PlayerID, Player] = PrivateAttr(default_factory=dict)
    _event_by_type: Dict[EventName, List[Event]] = PrivateAttr(default_factory=lambda: defaultdict(list))
    _event_queue: Deque[Event] = PrivateAttr(default_factory=deque)
    _investigated_ids: List[PlayerID] = PrivateAttr(default_factory=list)

    @computed_field
    @cached_property
    def all_player_ids(self) -> List[str]:
        return [player.id for player in self.players]

    def model_post_init(self, context: Any, /) -> None:
        self._id_to_player = {p.id: p for p in self.players}

    # --- accessors ---
    def get_player_by_id(self, pid: PlayerID) -> Optional[Player]:
        return self._id_to_player.get(pid)

    def get_players_by_role(self, role: RoleConst) -> List[Player]:
        return [p for p in self.players if p.role.name == role]

    def get_players_by_team(self, team: Team) -> List[Player]:
        return [p for p in self.players if p.role.team == team]

    def alive_players(self) -> List[Player]:
        return [p for p in self.players if p.alive]

    def alive_player_ids(self) -> List[PlayerID]:
        return [p.id for p in self.players if p.alive]

    def eliminated_players(self) -> List[Player]:
        return [p for p in self.players if not p.alive]

    def is_alive(self, player_id: PlayerID) -> bool:
        player = self.get_player_by_id(player_id)
        return bool(player and player.alive)

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def num_alive(self) -> int:
        return len(self.alive_players())

    def get_event_by_name(self, event_name: EventName) -> List[Event]:
        return self._event_by_type[event_name]

    def mark_investigated(self, pid: PlayerID) -> None:
        self._investigated_ids.append(pid)

    @property
    def investigated_ids(self) -> List[PlayerID]:
        return list(self._investigated_ids)

    # --- events ---
    def push_event(
        self,
        description: str,
        event_name: EventName,
        public: bool,
        visible_to: Optional[List[PlayerID]] = None,
        data: Optional[Union[DataEntry, Dict[str, Any]]] = None,
        source=MODERATOR_ID,
        visible_in_ui: bool = True,
    ):
        visible_to = visible_to or []
        day_key = self.round_count
        self.history.setdefault(day_key, [])
        entry = Event(
            day=day_key,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=event_name,
            description=description,
            public=public,
            visible_to=visible_to,
            data=data,
            source=source,
            visible_in_ui=visible_in_ui,
        )
        self.history[day_key].append(entry)
        self._event_by_type[event_name].append(entry)
        self._event_queue.append(entry)

        public_view = entry.view_by_access(user_level=DataAccessLevel.PUBLIC)
        personal_view = entry.view_by_access(user_level=DataAccessLevel.PERSONAL)

        if public:
            for player in self.players:
                player.update(personal_view if player.id == source else public_view)
        else:
            for player_id in visible_to:
                player = self.get_player_by_id(player_id)
                if player:
                    player.update(personal_view if player.id == source else public_view)

    def add_phase_divider(self, divider: PhaseDivider):
        self.push_event(
            description=divider.value,
            event_name=EventName.PHASE_DIVIDER,
            public=False,
            data=PhaseDividerDataEntry(divider_type=divider.value),
        )

    def eliminate_player(self, pid: PlayerID):
        if pid not in self.all_player_ids:
            logger.warning(f"Tried to eliminate unknown player {pid}.")
            return
        player = self.get_player_by_id(pid)
        if player:
            player.eliminate(round_number=self.round_count, phase=self.phase)

    def consume_messages(self) -> List[Event]:
        messages = list(self._event_queue)
        self._event_queue.clear()
        return messages

    def get_elimination_info(self):
        return [player.report_elimination() for player in self.players]


def get_last_action_request(event_views: Sequence[PlayerEventView], event_name: EventName) -> Optional[PlayerEventView]:
    """Find the most recent action-request view by event name (search newest first)."""
    return next((entry for entry in reversed(list(event_views)) if entry.event_name == event_name), None)
