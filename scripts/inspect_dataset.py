"""Validate and summarize a demonstrations.npz dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "demonstrations.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a demonstration dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        required_keys = {
            "observations",
            "actions",
            "episode_ids",
            "episode_starts",
            "episode_lengths",
            "cube_initial_positions",
            "target_positions",
            "successes",
            "episode_seeds",
            "metadata_json",
        }
        missing = sorted(required_keys - set(dataset.files))
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {missing}")

        observations = dataset["observations"]
        actions = dataset["actions"]
        episode_ids = dataset["episode_ids"]
        starts = dataset["episode_starts"]
        lengths = dataset["episode_lengths"]
        successes = dataset["successes"]
        phases = dataset["phases"]
        final_errors = dataset["final_xy_errors"]
        metadata = json.loads(str(dataset["metadata_json"].item()))

        episode_count = len(lengths)
        transition_count = len(observations)
        expected_starts = np.zeros(episode_count, dtype=np.int64)
        if episode_count > 1:
            expected_starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)

        checks = {
            "observations_are_2d": observations.ndim == 2,
            "actions_are_2d": actions.ndim == 2,
            "transition_counts_match": (
                len(actions) == transition_count == len(episode_ids)
            ),
            "episode_lengths_sum_to_transitions": (
                int(lengths.sum()) == transition_count
            ),
            "episode_starts_are_consistent": np.array_equal(
                starts, expected_starts
            ),
            "all_values_are_finite": bool(
                np.isfinite(observations).all() and np.isfinite(actions).all()
            ),
            "all_actions_in_range": bool(
                np.all(actions >= -1.00001) and np.all(actions <= 1.00001)
            ),
            "all_episodes_successful": bool(np.all(successes)),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise ValueError(f"Dataset validation failed: {failed_checks}")

        unique_phases, phase_counts = np.unique(phases, return_counts=True)

        print("Demonstration Dataset Inspection")
        print(f"Path: {dataset_path}")
        print(f"File size: {dataset_path.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"Observation dimension: {observations.shape[1]}")
        print(f"Action dimension: {actions.shape[1]}")
        print(f"Episodes: {episode_count}")
        print(f"Successful episodes: {int(successes.sum())}")
        print(f"Transitions: {transition_count:,}")
        print(
            "Episode length: "
            f"min={int(lengths.min())}, "
            f"mean={float(lengths.mean()):.1f}, "
            f"max={int(lengths.max())}"
        )
        print(
            "Final XY error: "
            f"min={float(final_errors.min()):.4f} m, "
            f"mean={float(final_errors.mean()):.4f} m, "
            f"max={float(final_errors.max()):.4f} m"
        )
        print(
            "Observation range: "
            f"[{float(observations.min()):.3f}, "
            f"{float(observations.max()):.3f}]"
        )
        print(
            "Action range: "
            f"[{float(actions.min()):.3f}, {float(actions.max()):.3f}]"
        )
        print("Phase counts:")
        for phase, count in zip(unique_phases, phase_counts, strict=True):
            print(f"  {phase}: {int(count):,}")
        print(
            "Collection attempts: "
            f"{metadata['collection_attempts']} "
            f"({metadata['failed_attempts']} failed)"
        )
        print("Validation: PASSED")


if __name__ == "__main__":
    main()
