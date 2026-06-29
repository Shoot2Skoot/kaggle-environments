# Secret Hitler

A hidden-role social-deduction environment for 5–10 players, modeled on the architecture of the
`werewolf` environment. See [GAME_RULE.md](GAME_RULE.md) for the full ruleset.

## What makes it interesting for agents

Unlike pure "find the wolf" deduction, Secret Hitler has a **hidden policy deck**: the President
secretly draws three policies, discards one, and passes two to the Chancellor, who enacts one.
Only those two players see the real tiles, so the public board plus players' *claims* about what
they drew create objective ground truth to lie about — a sharp test of deception and deduction.

## Layout

```
secret_hitler/
├── secret_hitler.py        # interpreter, built-in agents, renderers, spec wiring
├── secret_hitler.json      # specification
├── game/                   # engine: consts, board, deck, roles, records, states, engine, protocols
├── harness/                # LLM (litellm) agent
├── visualizer/default/     # 2D board replay visualizer (Vite + TypeScript)
└── test_llm_game.py        # local real-LLM integration runner
```

## Running a game

```python
from kaggle_environments import make

# One config entry per player; role counts must match the player count
# (5 -> 3 Liberal / 1 Fascist / 1 Hitler, etc. — see GAME_RULE.md).
agents = [
    {"id": "p0", "role": "Liberal", "agent_id": "random"},
    {"id": "p1", "role": "Liberal", "agent_id": "random"},
    {"id": "p2", "role": "Liberal", "agent_id": "random"},
    {"id": "p3", "role": "Fascist", "agent_id": "random"},
    {"id": "p4", "role": "Hitler",  "agent_id": "random"},
]
env = make("secret_hitler", configuration={"agents": agents, "seed": 1})
env.run(["random"] * 5)        # or "deterministic"
print(env.render(mode="ansi"))
```

Set `randomize_roles=True` (with a `seed`) to shuffle the configured roles among players
deterministically.

### LLM agents

```python
from kaggle_environments.envs.secret_hitler.harness.base import LLMSecretHitlerAgent

env = make("secret_hitler", configuration={"agents": agents})
players = [LLMSecretHitlerAgent(model_name="gemini/gemini-2.5-flash") for _ in agents]
env.run(players)
```

Or run the bundled integration script (needs `litellm` and a provider API key):

```bash
python kaggle_environments/envs/secret_hitler/test_llm_game.py --model gemini/gemini-2.5-flash --players 7
```

## Configuration

| Key | Default | Description |
|---|---|---|
| `agents` | 10-entry sample | Per-player `{id, role, agent_id, ...}`; sliced to the player count |
| `seed` | 123 | Deterministic role / id / deck shuffling |
| `randomize_roles` | false | Shuffle configured roles among players |
| `randomize_ids` | false | Shuffle player ids |
| `discussion_protocol` | `RoundRobinDiscussion` (`{rounds: 2}`) | `{name, params}` for the round-robin debate; players speak in turn starting with the President |
| `episodeSteps` | 2000 | Step cap (a no-winner cap-out scores all players 0) |

## Tests

```bash
uv run pytest tests/envs/secret_hitler/ -v          # engine + harness
pnpm --filter @kaggle-environments/secret-hitler-visualizer build
pnpm test:e2e --project secret_hitler               # visualizer
```
