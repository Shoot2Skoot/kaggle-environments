from collections import Counter, deque
from copy import deepcopy
from typing import Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .base import BasePlayer, BaseRole, PlayerID
from .consts import Phase, RoleConst, Team
from .records import PlayerEventView
from .rng import ID_SHUFFLE_OFFSET, ROLE_SHUFFLE_OFFSET, get_permutation


class RoleDescription:
    LIBERAL = (
        "You are a Liberal. You do not know anyone else's identity. Liberals win by enacting 5 "
        "Liberal policies or by assassinating Hitler."
    )
    FASCIST = (
        "You are a Fascist. You know who the other Fascists and Hitler are. Fascists win by "
        "enacting 6 Fascist policies or by getting Hitler elected Chancellor once 3 Fascist "
        "policies have been enacted."
    )
    HITLER = (
        "You are Hitler. You win with the Fascists. In 5-6 player games you know who the Fascist "
        "is; in 7+ player games you do not know your allies. Avoid being assassinated, and get "
        "elected Chancellor after 3 Fascist policies to win."
    )


class Role(BaseRole):
    model_config = ConfigDict(use_enum_values=True)

    name: RoleConst = Field(..., frozen=True)
    team: Team
    descriptions: str


class Liberal(Role):
    name: RoleConst = RoleConst.LIBERAL
    team: Team = Team.LIBERAL
    descriptions: str = RoleDescription.LIBERAL


class Fascist(Role):
    name: RoleConst = RoleConst.FASCIST
    team: Team = Team.FASCIST
    descriptions: str = RoleDescription.FASCIST


class Hitler(Role):
    name: RoleConst = RoleConst.HITLER
    team: Team = Team.FASCIST
    descriptions: str = RoleDescription.HITLER


ROLE_CLASS_MAP = {
    RoleConst.LIBERAL.value: Liberal,
    RoleConst.FASCIST.value: Fascist,
    RoleConst.HITLER.value: Hitler,
}


# Role counts by player count: regular Fascists exclude Hitler.
_ROLE_COUNTS = {
    5: {RoleConst.LIBERAL: 3, RoleConst.FASCIST: 1, RoleConst.HITLER: 1},
    6: {RoleConst.LIBERAL: 4, RoleConst.FASCIST: 1, RoleConst.HITLER: 1},
    7: {RoleConst.LIBERAL: 4, RoleConst.FASCIST: 2, RoleConst.HITLER: 1},
    8: {RoleConst.LIBERAL: 5, RoleConst.FASCIST: 2, RoleConst.HITLER: 1},
    9: {RoleConst.LIBERAL: 5, RoleConst.FASCIST: 3, RoleConst.HITLER: 1},
    10: {RoleConst.LIBERAL: 6, RoleConst.FASCIST: 3, RoleConst.HITLER: 1},
}


def assign_role_counts(num_players: int) -> Dict[RoleConst, int]:
    if num_players not in _ROLE_COUNTS:
        raise ValueError(f"Secret Hitler supports 5-10 players, got {num_players}.")
    return dict(_ROLE_COUNTS[num_players])


def hitler_knows_fascists(num_players: int) -> bool:
    """Hitler knows the (single) Fascist only in 5-6 player games."""
    return num_players <= 6


class LLM(BaseModel):
    model_name: str
    properties: Dict = {}


class Agent(BaseModel):
    id: PlayerID
    agent_id: str
    display_name: str = ""
    role: RoleConst
    role_params: Dict = Field(default_factory=dict)
    thumbnail: Optional[str] = ""
    agent_harness_name: str = "basic_llm"
    llms: List[LLM] = []


class Player(BasePlayer):
    model_config = ConfigDict(use_enum_values=True)

    id: PlayerID
    agent: Agent
    role: BaseRole
    alive: bool = True
    eliminated_during_round: int = -1
    eliminated_during_phase: Optional[Phase] = None

    _message_queue: Deque[PlayerEventView] = PrivateAttr(default_factory=deque)
    _role_state: Dict = PrivateAttr(default_factory=dict)

    def set_role_state(self, key, value):
        self._role_state[key] = value

    def get_role_state(self, key, default=None):
        return self._role_state.get(key, default)

    def update(self, entry: PlayerEventView):
        self._message_queue.append(entry)

    def consume_messages(self) -> List[PlayerEventView]:
        messages = list(self._message_queue)
        self._message_queue.clear()
        return messages

    def eliminate(self, round_number: int, phase: Phase):
        self.alive = False
        self.eliminated_during_round = round_number
        self.eliminated_during_phase = phase.value if isinstance(phase, Phase) else phase

    def report_elimination(self):
        return {
            "player_id": self.id,
            "eliminated_during_round": self.eliminated_during_round,
            "eliminated_during_phase": self.eliminated_during_phase,
        }


def _shuffle_roles(agents_config, seed):
    roles_config = [{"role": a["role"], "role_params": a.get("role_params", {})} for a in agents_config]
    permuted = get_permutation(roles_config, seed + ROLE_SHUFFLE_OFFSET)
    new_config = deepcopy(agents_config)
    for role, agent in zip(permuted, new_config):
        agent["role"] = role["role"]
        agent["role_params"] = role["role_params"]
    return new_config


def _shuffle_ids(agents_config, seed):
    ids = [a["id"] for a in agents_config]
    permuted = get_permutation(ids, seed + ID_SHUFFLE_OFFSET)
    new_config = deepcopy(agents_config)
    for player_id, agent in zip(permuted, new_config):
        agent["id"] = player_id
    return new_config


def create_players_from_agents_config(
    agents_config: List[Dict],
    randomize_roles: bool = False,
    randomize_ids: bool = False,
    seed: Optional[int] = None,
) -> List[Player]:
    if randomize_roles:
        assert seed is not None
        agents_config = _shuffle_roles(agents_config, seed)
    if randomize_ids:
        assert seed is not None
        agents_config = _shuffle_ids(agents_config, seed)

    agent_ids = [a["id"] for a in agents_config]
    if len(agent_ids) != len(set(agent_ids)):
        counts = Counter(agent_ids)
        duplicates = [item for item, count in counts.items() if count > 1 and item is not None]
        if duplicates:
            raise ValueError(f"Duplicate agent ids found: {', '.join(duplicates)}")

    agents = [Agent(**cfg) for cfg in agents_config]
    return [Player(id=agent.id, agent=agent, role=ROLE_CLASS_MAP[agent.role](**agent.role_params)) for agent in agents]
