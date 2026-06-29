from typing import Any, Dict, Optional

from .base import DiscussionProtocol
from .chat import TurnByTurnBiddingDiscussion

PROTOCOL_REGISTRY = {
    "TurnByTurnBiddingDiscussion": TurnByTurnBiddingDiscussion,
}

DEFAULT_DISCUSSION_PROTOCOL_NAME = "TurnByTurnBiddingDiscussion"


def create_protocol(config: Optional[Dict[str, Any]], default_name: str = None) -> DiscussionProtocol:
    config = config or {}
    name = config.get("name") or default_name or DEFAULT_DISCUSSION_PROTOCOL_NAME
    params = config.get("params") or {}
    if name not in PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown discussion protocol '{name}'. Known: {list(PROTOCOL_REGISTRY)}")
    return PROTOCOL_REGISTRY[name](**params)
