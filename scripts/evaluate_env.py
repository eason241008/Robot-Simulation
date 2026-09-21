"""Evaluate ScriptedPolicy exclusively through PickPlaceEnv's public API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.pick_place_env import PickPlaceEnv
from policies.scripted_policy import ScriptedPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main() -> None:
    args = parse_args()
    env = PickPlaceEnv(randomize=True)
    policy = ScriptedPolicy(env)
    successes = 0

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            observation = env.reset(seed=episode_seed)
            policy.reset(observation)
            done = False
            info: dict[str, object] = {}
            failure_reason: str | None = None

            try:
                while not done:
                    action = policy.predict(observation)
                    observation, _, done, info = env.step(action)
            except RuntimeError as error:
                failure_reason = str(error)

            success = bool(info.get("success", False))
            if success:
                successes += 1
            status = "SUCCESS" if success else "FAILURE"
            print(
                f"Episode {episode + 1:02d}/{args.episodes}: {status} | "
                f"steps={info.get('step_count', env.step_count)} | "
                f"distance={info.get('cube_target_xy_distance', np.nan):.4f} m"
            )
            if failure_reason:
                print("  Reason:", failure_reason)

        success_rate = 100.0 * successes / args.episodes
        print("\nPickPlaceEnv Evaluation")
        print(f"Episodes: {args.episodes}")
        print(f"Success: {successes}")
        print(f"Failure: {args.episodes - successes}")
        print(f"Success Rate: {success_rate:.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    main()
