from abc import ABC, abstractmethod
from typing import Optional, Sequence


class DiscussionProtocol(ABC):
    """Drives a round of structured debate (round-robin speaking).

    The engine calls ``begin`` once, then repeatedly asks ``next_speaker`` (one speaker per
    tick) and ``record_speech`` until ``is_finished`` is true.
    """

    @property
    def display_name(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def rule(self) -> str:
        """Human-readable description of the protocol, shown to players."""

    @abstractmethod
    def begin(self, alive_ids: Sequence[str], leader_id: Optional[str] = None) -> None:
        """Start a fresh discussion among ``alive_ids``, with ``leader_id`` (e.g. the President)
        speaking first each round."""

    @abstractmethod
    def next_speaker(self) -> Optional[str]:
        """The next player to speak, or ``None`` when the discussion is over."""

    @abstractmethod
    def record_speech(self, speaker_id: str) -> None:
        """Mark that ``speaker_id`` has spoken, advancing the schedule."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Whether the discussion has ended."""
