from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Optional, Type

from pydantic import BaseModel, StringConstraints

from .consts import MODERATOR_ID, EventName

# The ID regex supports Unicode letters (\p{L}), numbers (\p{N}) and common symbols for IDs.
ROBUST_ID_REGEX = r"^[\p{L}\p{N} _.-]+$"

PlayerID = Annotated[str, StringConstraints(pattern=ROBUST_ID_REGEX, min_length=1, max_length=128)]


class BasePlayer(BaseModel, ABC):
    id: PlayerID
    """The unique id of the player. Also how the player is referred to in the game."""

    alive: bool = True

    @abstractmethod
    def set_role_state(self, key, value):
        """Set role related state, which is a dict."""

    @abstractmethod
    def get_role_state(self, key, default=None):
        """Get role related state."""


class BaseAction(BaseModel):
    pass


class BaseEvent(BaseModel):
    event_name: EventName


class BaseState(BaseModel):
    @abstractmethod
    def push_event(
        self,
        description: str,
        event_name: EventName,
        public: bool,
        visible_to: Optional[List[PlayerID]] = None,
        data: Any = None,
        source=MODERATOR_ID,
        visible_in_ui: bool = True,
    ):
        """Publish an event."""


class BaseModerator(ABC):
    @abstractmethod
    def advance(self, player_actions: Dict[PlayerID, BaseAction]):
        """Move one Kaggle environment step further. Used within Kaggle 'interpreter'."""

    @abstractmethod
    def request_action(
        self,
        action_cls: Type[BaseAction],
        player_id: PlayerID,
        prompt: str,
        data=None,
        event_name=EventName.MODERATOR_ANNOUNCEMENT,
    ):
        """Request an action from a player (enqueue + push a prompt event)."""

    @property
    @abstractmethod
    def state(self) -> BaseState:
        """The current game state."""


class BaseRole(BaseModel, ABC):
    """Secret Hitler roles carry team + descriptions only; all abilities are President/board
    driven by the Moderator, so there is no per-role event-handler machinery (unlike werewolf).
    """
