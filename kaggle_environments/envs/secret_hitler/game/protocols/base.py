from abc import ABC, abstractmethod
from typing import Dict, List, Sequence


class DiscussionProtocol(ABC):
    """Drives the debate that precedes the election vote.

    The engine alternates bidding ticks (all alive players bid for the floor) and chat ticks
    (the winning bidder speaks), until the protocol reports it is finished.
    """

    @property
    def display_name(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def rule(self) -> str:
        """Human-readable description of the protocol, shown to players."""

    @abstractmethod
    def begin(self, alive_ids: Sequence[str]) -> None:
        """Reset state for a fresh discussion among ``alive_ids``."""

    @abstractmethod
    def bidders(self, alive_ids: Sequence[str]) -> List[str]:
        """Players who should bid this tick (empty => discussion over)."""

    @abstractmethod
    def resolve_bids(self, bids: Dict[str, int]) -> List[str]:
        """Record this tick's bids and return the winning speaker(s)."""

    @abstractmethod
    def record_speech(self, speaker_ids: Sequence[str]) -> None:
        """Mark that the given speakers have spoken (advances the turn counter)."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Whether the discussion has ended."""

    @property
    @abstractmethod
    def max_bid(self) -> int:
        """The maximum legal bid amount."""
