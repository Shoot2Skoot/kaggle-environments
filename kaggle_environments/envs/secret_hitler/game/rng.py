"""Deterministic, cross-platform pseudo-random helpers.

We avoid Python's ``random`` module so that shuffles are reproducible across platforms and
languages (e.g. a JS re-implementation). A Linear Congruential Generator (glibc parameters)
drives a Fisher-Yates shuffle.

Seed discipline (see plan invariant #4): callers use *distinct* offsets for role-shuffle,
id-shuffle, deck-shuffle and the first-president draw, and an *evolving* counter for each
mid-game deck reshuffle, so the streams are reproducible yet uncorrelated.
"""

from typing import List

# Distinct seed offsets so independent draws never share a stream.
ROLE_SHUFFLE_OFFSET = 0
ID_SHUFFLE_OFFSET = 123
DECK_SHUFFLE_OFFSET = 7919
FIRST_PRESIDENT_OFFSET = 104729

# LCG parameters (from glibc).
_M = 2**31
_A = 1103515245
_C = 12345


def get_permutation(items: List, seed: int) -> List:
    """Return a deterministic permutation of ``items`` for the given integer ``seed``."""
    shuffled_items = list(items)
    n = len(shuffled_items)
    current_seed = seed % _M
    for i in range(n - 1, 0, -1):
        current_seed = (_A * current_seed + _C) % _M
        j = current_seed % (i + 1)
        shuffled_items[i], shuffled_items[j] = shuffled_items[j], shuffled_items[i]
    return shuffled_items


def deterministic_index(n: int, seed: int) -> int:
    """Return a deterministic index in ``[0, n)`` for the given seed."""
    if n <= 0:
        raise ValueError("n must be positive")
    current_seed = (_A * (seed % _M) + _C) % _M
    return current_seed % n
