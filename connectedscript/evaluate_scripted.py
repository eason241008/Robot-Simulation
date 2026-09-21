"""Evaluate the scripted pick-and-place controller on randomized episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripted_pick_place import (
    ModelHandles,
    SCENE_PATH,
    reset_episode,
    run_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "scripted_evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scripted pick-and-place on randomized scenes."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_duration <= 0:
        parser.error("--max-duration must be positive")
    return args


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    handles = ModelHandles(model)

    records: list[dict[str, object]] = []
    success_count = 0

    for episode in range(1, args.episodes + 1):
        sampled_cube, sampled_target = reset_episode(
            model, data, handles, rng=rng
        )
        failure_reason: str | None = None
        success = False

        try:
            controller = run_episode(
                model,
                data,
                handles,
                args.max_duration,
                print_interval=args.max_duration + 1.0,
                viewer=None,
                verbose=False,
            )
            success = bool(controller.done and controller.success)
            if not success:
                failure_reason = "Final placement condition was not satisfied."
        except (RuntimeError, np.linalg.LinAlgError) as error:
            failure_reason = str(error)

        cube_final = data.xpos[handles.cube_body].copy()
        target_final = data.site_xpos[handles.target_site].copy()
        xy_error = float(np.linalg.norm(cube_final[:2] - target_final[:2]))
        if success:
            success_count += 1

        record = {
            "episode": episode,
            "success": success,
            "cube_initial_position": sampled_cube.tolist(),
            "target_position": sampled_target.tolist(),
            "cube_final_position": cube_final.tolist(),
            "final_xy_error": xy_error,
            "simulation_time": float(data.time),
            "failure_reason": failure_reason,
        }
        records.append(record)

        status = "SUCCESS" if success else "FAILURE"
        message = (
            f"Episode {episode:02d}/{args.episodes}: {status} | "
            f"cube={np.round(sampled_cube[:2], 3)} | "
            f"target={np.round(sampled_target[:2], 3)} | "
            f"final error={xy_error:.4f} m"
        )
        if failure_reason:
            message += f" | {failure_reason}"
        print(message)

    success_rate = 100.0 * success_count / args.episodes
    result = {
        "seed": args.seed,
        "episodes": args.episodes,
        "success_count": success_count,
        "failure_count": args.episodes - success_count,
        "success_rate_percent": success_rate,
        "records": records,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nScripted Policy Evaluation")
    print(f"Episodes: {args.episodes}")
    print(f"Success: {success_count}")
    print(f"Failure: {args.episodes - success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
