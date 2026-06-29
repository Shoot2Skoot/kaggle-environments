"""Run a full Secret Hitler game with real LLM agents for local integration testing.

Usage:
    python kaggle_environments/envs/secret_hitler/test_llm_game.py --model gemini/gemini-2.5-flash --players 7

Requires litellm and the relevant provider API key in the environment (e.g. GEMINI_API_KEY).
"""

import argparse
import os

from kaggle_environments import make
from kaggle_environments.envs.secret_hitler.game.consts import RoleConst
from kaggle_environments.envs.secret_hitler.game.roles import assign_role_counts
from kaggle_environments.envs.secret_hitler.harness.base import LLMSecretHitlerAgent


def build_config(num_players: int):
    counts = assign_role_counts(num_players)
    roles = (
        [RoleConst.LIBERAL.value] * counts[RoleConst.LIBERAL]
        + [RoleConst.FASCIST.value] * counts[RoleConst.FASCIST]
        + [RoleConst.HITLER.value] * counts[RoleConst.HITLER]
    )
    return [{"id": f"player_{i}", "role": roles[i], "agent_id": "llm"} for i in range(num_players)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("SECRET_HITLER_LLM_MODEL", "gemini/gemini-2.5-flash"))
    parser.add_argument("--players", type=int, default=7)
    parser.add_argument("--replay-path", default="secret_hitler_replay.json")
    parser.add_argument("--randomize-roles", action="store_true")
    args = parser.parse_args()

    config = {"agents": build_config(args.players), "randomize_roles": args.randomize_roles, "seed": 1}
    env = make("secret_hitler", debug=True, configuration=config)
    agents = [LLMSecretHitlerAgent(model_name=args.model) for _ in range(args.players)]
    env.run(agents)

    print(env.render(mode="ansi"))
    end = env.info.get("GAME_END", {})
    print("\nWinner:", end.get("winner_team"), "-", end.get("reason"))

    with open(args.replay_path, "w") as f:
        f.write(env.render(mode="json"))
    print(f"Replay written to {args.replay_path}")
    return env


if __name__ == "__main__":
    main()
