"""Collect successful state-action demonstrations from ScriptedPolicy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.pick_place_env import PickPlaceEnv
from policies.scripted_policy import ScriptedPolicy


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "demonstrations.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect successful scripted pick-and-place demonstrations."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of successful demonstrations to save (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Seed assigned to the first collection attempt (default: 1000).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Stop after this many attempts (default: 3 x requested episodes).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_attempts is not None and args.max_attempts < args.episodes:
        parser.error("--max-attempts must be at least --episodes")
    return args


def main() -> None:
    args = parse_args()
    max_attempts = args.max_attempts or args.episodes * 3
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = PickPlaceEnv(randomize=True)
    policy = ScriptedPolicy(env)

    episode_observations: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    episode_phases: list[np.ndarray] = []
    episode_lengths: list[int] = []
    cube_initial_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    final_cube_positions: list[np.ndarray] = []
    final_xy_errors: list[float] = []
    episode_seeds: list[int] = []
    failure_records: list[dict[str, object]] = []

    attempts = 0
    started_at = time.perf_counter()

    try:
        while len(episode_lengths) < args.episodes and attempts < max_attempts:
            episode_seed = args.seed + attempts
            attempts += 1
            observation = env.reset(seed=episode_seed)
            policy.reset(observation)
            cube_initial = env.cube_position.copy()
            target = env.target_position.copy()

            observations: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            phases: list[str] = []
            done = False
            info: dict[str, object] = {}
            failure_reason: str | None = None

            try:
                while not done:
                    action = policy.predict(observation)

                    # This is the supervised-learning pair: state before the
                    # expert action and the action selected for that state.
                    observations.append(observation.copy())
                    actions.append(action.copy())
                    phases.append(policy.phase)

                    observation, _, done, info = env.step(action)
            except (RuntimeError, np.linalg.LinAlgError) as error:
                failure_reason = str(error)

            success = bool(info.get("success", False))
            if success:
                episode_observations.append(
                    np.asarray(observations, dtype=np.float32)
                )
                episode_actions.append(np.asarray(actions, dtype=np.float32))
                episode_phases.append(np.asarray(phases, dtype="U24"))
                episode_lengths.append(len(observations))
                cube_initial_positions.append(cube_initial.astype(np.float32))
                target_positions.append(target.astype(np.float32))
                final_cube_positions.append(
                    env.cube_position.astype(np.float32)
                )
                final_xy_errors.append(
                    float(info["cube_target_xy_distance"])
                )
                episode_seeds.append(episode_seed)

                collected = len(episode_lengths)
                if collected == 1 or collected % 10 == 0 or collected == args.episodes:
                    print(
                        f"Collected {collected:03d}/{args.episodes} successful "
                        f"episodes | attempts={attempts} | "
                        f"latest length={episode_lengths[-1]}"
                    )
            else:
                failure_records.append(
                    {
                        "attempt": attempts,
                        "seed": episode_seed,
                        "reason": failure_reason
                        or str(info.get("termination_reason", "unknown")),
                    }
                )
                print(
                    f"Attempt {attempts:03d} failed (seed={episode_seed}): "
                    f"{failure_records[-1]['reason']}"
                )
    finally:
        env.close()

    if len(episode_lengths) < args.episodes:
        raise RuntimeError(
            f"Collected only {len(episode_lengths)} successful episodes after "
            f"{attempts} attempts; dataset was not written."
        )

    lengths = np.asarray(episode_lengths, dtype=np.int32)
    starts = np.zeros(len(lengths), dtype=np.int64)
    if len(starts) > 1:
        starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    observations_array = np.concatenate(episode_observations, axis=0)
    actions_array = np.concatenate(episode_actions, axis=0)
    phases_array = np.concatenate(episode_phases, axis=0)
    episode_ids = np.repeat(
        np.arange(len(lengths), dtype=np.int32), lengths
    )

    metadata = {
        "format_version": 1,
        "observation_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "observation_fields": [
            "arm_qpos[7]",
            "arm_qvel[7]",
            "gripper_width[1]",
            "cube_position[3]",
            "target_position[3]",
        ],
        "action_fields": ["delta_x", "delta_y", "delta_z", "gripper"],
        "action_range": [-1.0, 1.0],
        "only_successful_episodes": True,
        "requested_episodes": args.episodes,
        "collection_attempts": attempts,
        "failed_attempts": len(failure_records),
        "base_seed": args.seed,
        "failure_records": failure_records,
    }

    np.savez_compressed(
        output_path,
        observations=observations_array,
        actions=actions_array,
        phases=phases_array,
        episode_ids=episode_ids,
        episode_starts=starts,
        episode_lengths=lengths,
        cube_initial_positions=np.asarray(
            cube_initial_positions, dtype=np.float32
        ),
        target_positions=np.asarray(target_positions, dtype=np.float32),
        final_cube_positions=np.asarray(
            final_cube_positions, dtype=np.float32
        ),
        final_xy_errors=np.asarray(final_xy_errors, dtype=np.float32),
        successes=np.ones(len(lengths), dtype=np.bool_),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )

    elapsed = time.perf_counter() - started_at
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print("\nDemonstration Collection")
    print(f"Collected episodes: {len(lengths)}")
    print(f"Successful episodes: {int(np.sum(np.ones(len(lengths), dtype=bool)))}")
    print(f"Failed attempts: {len(failure_records)}")
    print(f"Transitions: {len(observations_array):,}")
    print(f"Dataset size: {size_mb:.2f} MB")
    print(f"Elapsed time: {elapsed:.1f} s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
