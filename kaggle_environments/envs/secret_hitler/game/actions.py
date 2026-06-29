from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from pydantic import Field, create_model

from .base import BaseAction, PlayerID
from .consts import EventName, PerceivedThreatLevel, Phase, PolicyColor, Vote

ACTION_EVENT_MAP = {}


def register_event(event_name: EventName):
    """Class decorator registering an EventName for an Action class."""

    def decorator(cls):
        ACTION_EVENT_MAP[cls.__name__] = event_name
        setattr(cls, "event_name", event_name)
        return cls

    return decorator


class Action(BaseAction):
    """Root of the action tree. Mirrors werewolf's record-keeping fields."""

    round_number: int
    phase: Phase
    actor_id: PlayerID
    reasoning: Optional[str] = Field(
        default=None,
        max_length=1_000_000,
        description="Private self-monologue explaining the action. Invisible to other players.",
    )
    perceived_threat_level: PerceivedThreatLevel = Field(
        default=PerceivedThreatLevel.SAFE,
        description="Self-perceived threat level. Invisible to other players.",
    )
    error: Optional[str] = None
    raw_prompt: Optional[str] = None
    raw_completion: Optional[str] = None
    cost: Optional[float] = Field(default=None, description="The cost of generating this action.")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt token usage.")
    completion_tokens: Optional[int] = Field(default=None, description="Completion token usage.")

    def serialize(self):
        return {"action_type": self.__class__.__name__, "kwargs": self.model_dump()}

    @classmethod
    def schema_for_player(cls, fields: Tuple = None, new_cls_name=None):
        """Build a player-facing JSON schema exposing only the given fields."""
        fields = fields or []
        if not new_cls_name:
            new_cls_name = cls.__name__ + "Data"
        field_definitions = {
            field: (cls.model_fields[field].annotation, cls.model_fields[field])
            for field in fields
            if field in cls.model_fields
        }
        sub_cls = create_model(new_cls_name, **field_definitions)
        return sub_cls.model_json_schema()

    @property
    def action_field(self) -> Optional[str]:
        return None


class TargetedAction(Action):
    target_id: PlayerID = Field(description="The target player's id.")

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "target_id")
        return super(TargetedAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "target_id"


@register_event(EventName.NOMINATION)
class NominateChancellorAction(TargetedAction):
    """President nominates a Chancellor candidate (``target_id``)."""


@register_event(EventName.VOTE_ACTION)
class VoteAction(Action):
    """Ja/Nein vote on the proposed government."""

    vote: Vote

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "vote")
        return super(VoteAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "vote"


@register_event(EventName.DISCARD_ACTION)
class DiscardPolicyAction(Action):
    """Discard one policy by color. Used by the President (3->2) and the Chancellor (2->1).

    The optional ``veto`` field is the Chancellor's veto proposal (only meaningful once veto is
    unlocked); there is no standalone propose action. The engine enforces mutual exclusion
    (``veto`` XOR a valid in-hand ``discard`` color).
    """

    discard: Optional[PolicyColor] = Field(default=None, description="The policy color to discard.")
    veto: bool = Field(default=False, description="Chancellor only: propose to veto this agenda.")

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "discard")
        return super(DiscardPolicyAction, cls).schema_for_player(fields, new_cls_name)

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_chancellor(cls):
        return super(DiscardPolicyAction, cls).schema_for_player(
            ("perceived_threat_level", "reasoning", "discard", "veto"),
            new_cls_name="ChancellorDiscardData",
        )

    @property
    def action_field(self):
        return "discard"


@register_event(EventName.VETO_RESULT)
class VetoConsentAction(Action):
    """President consents to (or refuses) the Chancellor's veto proposal."""

    consent: bool = Field(description="Whether the President agrees to veto the agenda.")

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "consent")
        return super(VetoConsentAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "consent"


@register_event(EventName.DISCUSSION)
class ChatAction(Action):
    message: str = Field(default="", max_length=1_000_000)

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "message")
        return super(ChatAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "message"


@register_event(EventName.BID_ACTION)
class BidAction(Action):
    """Bid for a speaking slot in the discussion (0..max)."""

    amount: int = Field(ge=0)

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning", "amount")
        return super(BidAction, cls).schema_for_player(fields, new_cls_name)

    @property
    def action_field(self):
        return "amount"


@register_event(EventName.INVESTIGATE_RESULT)
class InvestigateAction(TargetedAction):
    """President investigates a player's party membership."""


@register_event(EventName.SPECIAL_ELECTION)
class SpecialElectionAction(TargetedAction):
    """President picks the next Presidential candidate."""


@register_event(EventName.EXECUTION)
class ExecutionAction(TargetedAction):
    """President executes a player."""


@register_event(EventName.POLICY_PEEK)
class PolicyPeekAckAction(Action):
    """President acknowledges peeking at the top three policies (no payload)."""

    @classmethod
    @lru_cache(maxsize=16)
    def schema_for_player(cls, fields=None, new_cls_name=None):
        fields = fields or ("perceived_threat_level", "reasoning")
        return super(PolicyPeekAckAction, cls).schema_for_player(fields, new_cls_name)


@register_event(EventName.NOOP_ACTION)
class NoOpAction(Action):
    pass


ACTIONS = [
    NominateChancellorAction,
    VoteAction,
    DiscardPolicyAction,
    VetoConsentAction,
    ChatAction,
    BidAction,
    InvestigateAction,
    SpecialElectionAction,
    ExecutionAction,
    PolicyPeekAckAction,
    NoOpAction,
]

ACTION_REGISTRY = {action.__name__: action for action in ACTIONS}


def create_action(serialized):
    return ACTION_REGISTRY[serialized["action_type"]](**serialized.get("kwargs", {}))
