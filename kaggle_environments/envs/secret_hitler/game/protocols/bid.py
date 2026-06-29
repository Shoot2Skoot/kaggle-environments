from typing import Dict, List


def highest_bidders(bids: Dict[str, int]) -> List[str]:
    """Return the player id(s) with the maximum (positive) bid. Empty if every bid is 0.

    Ties are returned in the bids' insertion order; the engine resolves ties deterministically
    (it speaks the first tied bidder), so all tied players keep an equal chance across the game
    via rotation, without relying on a random draw.
    """
    if not bids:
        return []
    top = max(bids.values())
    if top <= 0:
        return []
    return [pid for pid, amount in bids.items() if amount == top]
