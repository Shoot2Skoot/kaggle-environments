from collections import deque
from typing import Optional, Sequence

from .base import DiscussionProtocol


class RoundRobinDiscussion(DiscussionProtocol):
    """Every alive player speaks once per round, in seat order starting from the leader
    (the President). Runs for a fixed number of ``rounds``."""

    def __init__(self, rounds: int = 2):
        self._rounds = max(1, int(rounds))
        self._queue: deque[str] = deque()

    @property
    def rule(self) -> str:
        return (
            f"Round-robin discussion: every living player speaks once per round, in seat order "
            f"starting from the President. {self._rounds} round(s) of speaking."
        )

    @property
    def rounds(self) -> int:
        return self._rounds

    def begin(self, alive_ids: Sequence[str], leader_id: Optional[str] = None) -> None:
        order = list(alive_ids)
        if leader_id in order:
            i = order.index(leader_id)
            order = order[i:] + order[:i]
        self._queue = deque(order * self._rounds)

    def next_speaker(self) -> Optional[str]:
        return self._queue[0] if self._queue else None

    def record_speech(self, speaker_id: str) -> None:
        if self._queue and self._queue[0] == speaker_id:
            self._queue.popleft()
        elif self._queue:
            # Defensive: a different player spoke (shouldn't happen); drop one to guarantee progress.
            self._queue.popleft()

    def is_finished(self) -> bool:
        return not self._queue
