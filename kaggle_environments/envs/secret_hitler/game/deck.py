"""The policy deck: 6 Liberal + 11 Fascist tiles."""

from typing import List

from .consts import PolicyColor
from .rng import DECK_SHUFFLE_OFFSET, get_permutation

NUM_LIBERAL_POLICIES = 6
NUM_FASCIST_POLICIES = 11


class PolicyDeck:
    """Draw/discard pile with deterministic shuffling and automatic reshuffle.

    The reshuffle (discard pile -> draw pile when fewer than 3 tiles remain) lives *inside*
    ``draw`` so it covers both the President's 3-tile draw and the chaos single-tile draw —
    neither can ever hit an empty deck. Each reshuffle uses an evolving counter so reshuffles
    are reproducible yet uncorrelated with the initial deal.
    """

    def __init__(self, seed: int):
        self._seed = seed
        self._reshuffle_count = 0
        tiles = [PolicyColor.LIBERAL] * NUM_LIBERAL_POLICIES + [PolicyColor.FASCIST] * NUM_FASCIST_POLICIES
        self._draw: List[PolicyColor] = get_permutation(tiles, seed + DECK_SHUFFLE_OFFSET)
        self._discard: List[PolicyColor] = []

    @property
    def draw_count(self) -> int:
        return len(self._draw)

    @property
    def discard_count(self) -> int:
        return len(self._discard)

    def _reshuffle_if_needed(self, needed: int) -> None:
        if len(self._draw) < needed:
            self._reshuffle_count += 1
            combined = self._draw + self._discard
            self._draw = get_permutation(combined, self._seed + DECK_SHUFFLE_OFFSET + self._reshuffle_count)
            self._discard = []

    def draw(self, n: int) -> List[PolicyColor]:
        """Draw ``n`` tiles from the top of the deck, reshuffling first if needed."""
        self._reshuffle_if_needed(n)
        drawn, self._draw = self._draw[:n], self._draw[n:]
        return drawn

    def peek(self, n: int) -> List[PolicyColor]:
        """Look at the top ``n`` tiles without removing them (reshuffling first if needed)."""
        self._reshuffle_if_needed(n)
        return list(self._draw[:n])

    def discard(self, tiles: List[PolicyColor]) -> None:
        self._discard.extend(tiles)
