"""The Secret Hitler board: policy tracks, election tracker, and the presidential-power table."""

from typing import Optional

from .consts import PolicyColor, Power

LIBERAL_TRACK_SIZE = 5
FASCIST_TRACK_SIZE = 6
MAX_ELECTION_TRACKER = 3
VETO_UNLOCK_FASCIST_COUNT = 5
HITLER_CHANCELLOR_WIN_FASCIST_THRESHOLD = 3

# Presidential powers keyed by (player-count bucket, fascist-policy number that was just enacted).
# Buckets: "5-6", "7-8", "9-10".
_POWER_TABLE = {
    "5-6": {3: Power.POLICY_PEEK, 4: Power.EXECUTION, 5: Power.EXECUTION},
    "7-8": {2: Power.INVESTIGATE, 3: Power.SPECIAL_ELECTION, 4: Power.EXECUTION, 5: Power.EXECUTION},
    "9-10": {
        1: Power.INVESTIGATE,
        2: Power.INVESTIGATE,
        3: Power.SPECIAL_ELECTION,
        4: Power.EXECUTION,
        5: Power.EXECUTION,
    },
}


def _bucket(num_players: int) -> str:
    if num_players <= 6:
        return "5-6"
    if num_players <= 8:
        return "7-8"
    return "9-10"


def power_for(num_players: int, fascist_count: int) -> Optional[Power]:
    """Power granted when the ``fascist_count``-th fascist policy is enacted via a legislative
    session (never via chaos), for a game with ``num_players`` players."""
    return _POWER_TABLE[_bucket(num_players)].get(fascist_count)


class Board:
    def __init__(self):
        self.liberal_track = 0
        self.fascist_track = 0
        self.election_tracker = 0

    @property
    def veto_unlocked(self) -> bool:
        return self.fascist_track >= VETO_UNLOCK_FASCIST_COUNT

    def enact(self, color: PolicyColor) -> None:
        if color == PolicyColor.LIBERAL:
            self.liberal_track += 1
        else:
            self.fascist_track += 1

    def liberal_policy_win(self) -> bool:
        return self.liberal_track >= LIBERAL_TRACK_SIZE

    def fascist_policy_win(self) -> bool:
        return self.fascist_track >= FASCIST_TRACK_SIZE

    def advance_tracker(self) -> bool:
        """Advance the election tracker by one. Return True if it reaches chaos (3)."""
        self.election_tracker += 1
        return self.election_tracker >= MAX_ELECTION_TRACKER

    def reset_tracker(self) -> None:
        self.election_tracker = 0
