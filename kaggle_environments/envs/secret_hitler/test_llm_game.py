"""Run a full Secret Hitler game with real LLM agents for local integration testing.

Usage:
    python kaggle_environments/envs/secret_hitler/test_llm_game.py --model gemini/gemini-2.5-flash --players 7

Requires litellm and the relevant provider API key in the environment (e.g. GEMINI_API_KEY).
"""

import argparse
import json
import os


def _load_dotenv() -> None:
    """Best-effort load of a local .env so provider API keys are picked up.

    Guarded: never fails if python-dotenv is absent or no .env exists, so this
    module stays importable in CI/Kaggle where neither is present.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_load_dotenv()

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.secret_hitler.game.consts import RoleConst  # noqa: E402
from kaggle_environments.envs.secret_hitler.game.roles import assign_role_counts  # noqa: E402
from kaggle_environments.envs.secret_hitler.harness.base import LLMSecretHitlerAgent  # noqa: E402


def build_config(num_players: int):
    counts = assign_role_counts(num_players)
    roles = (
        [RoleConst.LIBERAL.value] * counts[RoleConst.LIBERAL]
        + [RoleConst.FASCIST.value] * counts[RoleConst.FASCIST]
        + [RoleConst.HITLER.value] * counts[RoleConst.HITLER]
    )
    return [{"id": f"player_{i}", "role": roles[i], "agent_id": "llm"} for i in range(num_players)]


def resolve_models(num_players: int, model: str, models_arg: str | None) -> list[str]:
    """One model per player. --models (comma-separated) takes precedence over --model.

    A shorter --models list is cycled to cover all seats (handy for model-vs-model games).
    """
    if models_arg:
        chosen = [m.strip() for m in models_arg.split(",") if m.strip()]
        if chosen:
            return [chosen[i % len(chosen)] for i in range(num_players)]
    return [model] * num_players


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SECRET_HITLER_LLM_MODEL", "gemini/gemini-2.5-flash"))
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated per-player models (cycled to fill seats). Overrides --model.",
    )
    parser.add_argument("--players", type=int, default=7)
    parser.add_argument("--replay-path", default="secret_hitler_replay.json")
    parser.add_argument("--randomize-roles", action="store_true")
    args = parser.parse_args()

    models = resolve_models(args.players, args.model, args.models)
    player_models = {f"player_{i}": models[i] for i in range(args.players)}

    config = {"agents": build_config(args.players), "randomize_roles": args.randomize_roles, "seed": 1}
    env = make("secret_hitler", debug=True, configuration=config)
    agents = [LLMSecretHitlerAgent(model_name=models[i]) for i in range(args.players)]
    env.run(agents)

    print(env.render(mode="ansi"))
    end = env.info.get("GAME_END", {})
    print("\nWinner:", end.get("winner_team"), "-", end.get("reason"))

    # Record which model each player ran on so the visualizer can show it.
    replay = json.loads(env.render(mode="json"))
    replay.setdefault("info", {})["PlayerModels"] = player_models
    with open(args.replay_path, "w") as f:
        json.dump(replay, f)
    print(f"Replay written to {args.replay_path}")
    return env


if __name__ == "__main__":
    main()
