from typing import Dict, List, Sequence

from .base import DiscussionProtocol
from .bid import highest_bidders


class TurnByTurnBiddingDiscussion(DiscussionProtocol):
    """Each turn every alive player bids for the floor; the highest bidder speaks. Repeats for
    up to ``max_turns`` speaking turns, or until every player bids 0 (all pass)."""

    def __init__(self, max_turns: int = 3, max_bid: int = 3):
        self._max_turns = max_turns
        self._max_bid = max_bid
        self._turns_taken = 0
        self._all_passed = False

    @property
    def rule(self) -> str:
        return (
            f"Turn-by-turn bidding discussion: each turn, every living player secretly bids 0-"
            f"{self._max_bid} for the floor; the highest bidder speaks (ties broken by seat "
            f"order). Up to {self._max_turns} speaking turns, ending early if everyone passes "
            f"(bids 0)."
        )

    @property
    def max_bid(self) -> int:
        return self._max_bid

    def begin(self, alive_ids: Sequence[str]) -> None:
        self._turns_taken = 0
        self._all_passed = False

    def bidders(self, alive_ids: Sequence[str]) -> List[str]:
        if self.is_finished():
            return []
        return list(alive_ids)

    def resolve_bids(self, bids: Dict[str, int]) -> List[str]:
        winners = highest_bidders(bids)
        if not winners:
            self._all_passed = True
            return []
        # Speak a single winner per turn (first tied bidder by insertion/seat order).
        return [winners[0]]

    def record_speech(self, speaker_ids: Sequence[str]) -> None:
        if speaker_ids:
            self._turns_taken += 1

    def is_finished(self) -> bool:
        return self._all_passed or self._turns_taken >= self._max_turns
