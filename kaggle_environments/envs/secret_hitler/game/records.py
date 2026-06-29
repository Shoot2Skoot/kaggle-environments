import json
from abc import ABC
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_serializer

from .base import BaseAction, BaseEvent, PlayerID
from .consts import (
    DetailedPhase,
    EventName,
    ObsKeys,
    PerceivedThreatLevel,
    Phase,
    PhaseDivider,
    PolicyColor,
    Power,
    RoleConst,
    Team,
    Vote,
)


def get_utc_now():
    return str(datetime.now(ZoneInfo("UTC")))


class DataAccessLevel(IntEnum):
    PUBLIC = 0
    PERSONAL = 1


class DataEntry(BaseModel, ABC):
    """Abstract base for all structured event payloads."""


class ActionDataMixin(BaseModel):
    """Mixin for action-derived payloads: carries the actor and their PERSONAL reasoning."""

    actor_id: PlayerID
    reasoning: Optional[str] = Field(
        default=None, description="Private reasoning.", json_schema_extra={"access": DataAccessLevel.PERSONAL}
    )
    perceived_threat_level: Optional[PerceivedThreatLevel] = Field(
        default=PerceivedThreatLevel.SAFE, json_schema_extra={"access": DataAccessLevel.PERSONAL}
    )
    action: Optional[BaseAction] = Field(default=None, json_schema_extra={"access": DataAccessLevel.PERSONAL})


class VisibleRawData(BaseModel):
    data_type: str
    json_str: str


class PlayerEventView(BaseModel):
    day: int
    phase: Phase
    detailed_phase: DetailedPhase
    event_name: EventName
    description: str
    data: Optional[dict | DataEntry] = None
    source: str
    created_at: str

    @model_serializer
    def serialize(self) -> dict:
        data = self.data.model_dump() if isinstance(self.data, DataEntry) else self.data
        return dict(
            day=self.day,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=self.event_name,
            description=self.description,
            data=data,
            source=self.source,
            created_at=self.created_at,
        )


class Event(BaseEvent):
    day: int  # round number (kept named `day` for compatibility with the history bucketing)
    phase: Phase
    detailed_phase: DetailedPhase
    event_name: EventName
    description: str
    public: bool = False
    visible_to: List[str] = Field(default_factory=list)
    data: Optional[dict | DataEntry] = None
    source: str
    created_at: str = Field(default_factory=get_utc_now)
    visible_in_ui: bool = True
    """Whether the event is shown to spectators in the UI. No effect on engine flow."""

    @field_serializer("data")
    def serialize_data(self, data):
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        if isinstance(data, BaseModel):
            return data.model_dump()
        return None

    def serialize(self):
        data_dict = self.model_dump()
        return VisibleRawData(data_type=self.data.__class__.__name__, json_str=json.dumps(data_dict)).model_dump()

    def view_by_access(self, user_level: DataAccessLevel) -> PlayerEventView:
        if isinstance(self.data, ActionDataMixin):
            fields_to_include = set()
            fields_to_exclude = set()
            for name, info in self.data.__class__.model_fields.items():
                if info.json_schema_extra:
                    if user_level >= info.json_schema_extra.get("access", DataAccessLevel.PUBLIC):
                        fields_to_include.add(name)
                    else:
                        fields_to_exclude.add(name)
                else:
                    fields_to_include.add(name)
            data = self.data.model_dump(include=fields_to_include, exclude=fields_to_exclude)
        else:
            data = self.data
        return PlayerEventView(
            day=self.day,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=self.event_name,
            description=self.description,
            data=data,
            source=self.source,
            created_at=self.created_at,
        )


# --- Setup ---
class GameStartDataEntry(DataEntry):
    player_ids: List[PlayerID]
    number_of_players: int
    role_counts: Dict[RoleConst, int]
    team_member_counts: Dict[Team, int]
    discussion_protocol_name: str
    discussion_protocol_rule: str


class GameStartRoleDataEntry(DataEntry):
    """Private setup info delivered to each player: their role, team, and known allies."""

    player_id: PlayerID
    team: Team
    role: RoleConst
    known_allies: Dict[PlayerID, RoleConst] = Field(default_factory=dict)
    rule_of_role: str


class PhaseDividerDataEntry(DataEntry):
    divider_type: PhaseDivider


class BoardSnapshotDataEntry(DataEntry):
    """Public board state, emitted each round for the visualizer (carried via env.info, not a
    player-queued event)."""

    liberal_policies: int
    fascist_policies: int
    election_tracker: int
    veto_unlocked: bool
    president_id: Optional[PlayerID] = None
    chancellor_id: Optional[PlayerID] = None
    draw_pile_count: int
    discard_pile_count: int
    alive_player_ids: List[PlayerID]


# --- Action requests ---
class RequestForActionDataEntry(DataEntry):
    action_json_schema: str


class RequestNominationDataEntry(RequestForActionDataEntry):
    valid_targets: List[PlayerID]


class RequestVoteDataEntry(RequestForActionDataEntry):
    president_id: PlayerID
    chancellor_nominee_id: PlayerID


class RequestDiscardDataEntry(RequestForActionDataEntry):
    policy_hand: List[PolicyColor]
    veto_available: bool = False


class RequestVetoConsentDataEntry(RequestForActionDataEntry):
    chancellor_id: PlayerID


class RequestPowerDataEntry(RequestForActionDataEntry):
    power: Power
    valid_targets: List[PlayerID] = Field(default_factory=list)


class RequestBidDataEntry(RequestForActionDataEntry):
    max_bid: int


class RequestChatDataEntry(RequestForActionDataEntry):
    pass


# --- Actions / results ---
class TargetedActionDataEntry(ActionDataMixin, DataEntry):
    target_id: PlayerID


class NominationDataEntry(TargetedActionDataEntry):
    """President's Chancellor nomination."""


class VoteDataEntry(ActionDataMixin, DataEntry):
    """A single player's Ja/Nein vote (public — votes are revealed in Secret Hitler)."""

    vote: Vote


class VoteResultDataEntry(DataEntry):
    ja_voters: List[PlayerID]
    nein_voters: List[PlayerID]
    passed: bool
    president_id: PlayerID
    chancellor_id: PlayerID


class ChatDataEntry(ActionDataMixin, DataEntry):
    message: str
    mentioned_player_ids: List[PlayerID] = Field(default_factory=list)


class BidDataEntry(ActionDataMixin, DataEntry):
    bid_amount: int


class BidResultDataEntry(DataEntry):
    winner_player_ids: List[PlayerID]
    bid_overview: Dict[PlayerID, int]


class DiscussionOrderDataEntry(DataEntry):
    chat_order_of_player_ids: List[PlayerID]


class DiscardActionDataEntry(ActionDataMixin, DataEntry):
    """A discard by President or Chancellor. ``discard`` is PERSONAL (the hidden choice)."""

    discard: Optional[PolicyColor] = Field(default=None, json_schema_extra={"access": DataAccessLevel.PERSONAL})
    veto_proposed: bool = False


class PolicyEnactedDataEntry(DataEntry):
    policy_color: PolicyColor
    liberal_policies: int
    fascist_policies: int
    president_id: Optional[PlayerID] = None
    chancellor_id: Optional[PlayerID] = None
    by_chaos: bool = False


class ChaosDataEntry(DataEntry):
    policy_color: PolicyColor


class ElectionTrackerDataEntry(DataEntry):
    election_tracker: int
    reason: str


class VetoResultDataEntry(DataEntry):
    consented: bool


class PowerUsedDataEntry(ActionDataMixin, DataEntry):
    power: Power
    target_id: Optional[PlayerID] = None


class InvestigationResultDataEntry(DataEntry):
    """PRIVATE: only the investigating President sees the membership. Kept out of the UI."""

    actor_id: PlayerID
    target_id: PlayerID
    membership: Team


class ExecutionResultDataEntry(DataEntry):
    executed_player_id: PlayerID


class GameEndResultsDataEntry(DataEntry):
    model_config = ConfigDict(use_enum_values=True)

    winner_team: Optional[Team]
    winner_ids: List[PlayerID]
    loser_ids: List[PlayerID]
    scores: Dict[str, int | float]
    reason: str
    last_round: int
    all_players_and_role: Dict[PlayerID, RoleConst]


# --- Observation ---
class SecretHitlerObservationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    player_id: PlayerID
    role: RoleConst
    team: Team
    is_alive: bool
    round_number: int
    detailed_phase: DetailedPhase
    game_state_phase: Phase
    all_player_ids: List[PlayerID]
    player_thumbnails: Dict[PlayerID, str] = {}
    alive_players: List[PlayerID]

    liberal_policies: int
    fascist_policies: int
    election_tracker: int
    veto_unlocked: bool
    draw_pile_count: int
    discard_pile_count: int

    president_id: Optional[PlayerID] = None
    chancellor_id: Optional[PlayerID] = None
    nominee_chancellor_id: Optional[PlayerID] = None
    last_president_id: Optional[PlayerID] = None
    last_chancellor_id: Optional[PlayerID] = None
    eligible_chancellor_ids: List[PlayerID] = Field(default_factory=list)

    # Private fields (populated only for this player).
    known_allies: Dict[PlayerID, RoleConst] = Field(default_factory=dict)
    policy_hand: Optional[List[PolicyColor]] = None
    pending_power: Optional[Power] = None
    power_valid_targets: List[PlayerID] = Field(default_factory=list)
    veto_available: bool = False
    investigation_results: Dict[PlayerID, Team] = Field(default_factory=dict)

    new_visible_announcements: List[str] = Field(default_factory=list)
    new_player_event_views: List[PlayerEventView] = Field(default_factory=list)

    def get_human_readable(self) -> str:
        return json.dumps(self.model_dump(), indent=2)


def set_raw_observation(kaggle_player_state, raw_obs: SecretHitlerObservationModel):
    kaggle_player_state.observation[ObsKeys.RAW_OBSERVATION] = raw_obs.model_dump()


def get_raw_observation(kaggle_observation) -> SecretHitlerObservationModel:
    return SecretHitlerObservationModel(**kaggle_observation[ObsKeys.RAW_OBSERVATION])
