"""Record randomized scripted pick-and-place episodes as animated GIF files."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from scripted_pick_place import (
    ModelHandles,
    SCENE_PATH,
    reset_episode,
    run_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDIA_DIR = PROJECT_ROOT / "media"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record each scripted pick-and-place episode as a GIF."
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible seed. Omit it for new positions each run.",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use the fixed scene instead of randomized cube/target positions.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MEDIA_DIR)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.max_duration <= 0:
        parser.error("--max-duration must be positive")
    return args


class GifRecorder:
    """Capture MuJoCo RGB frames at a fixed simulation-time frame rate."""

    def __init__(self, renderer: mujoco.Renderer, fps: int) -> None:
        self.renderer = renderer
        self.fps = fps
        self.frame_period = 1.0 / fps
        self.next_frame_time = 0.0
        self.frames: list[Image.Image] = []

    def capture(self, data: mujoco.MjData, *, force: bool = False) -> None:
        if not force and data.time + 1e-9 < self.next_frame_time:
            return

        self.renderer.update_scene(data, camera="overview")
        rgb = self.renderer.render()
        self.frames.append(Image.fromarray(rgb.copy()))

        if force:
            self.next_frame_time = max(
                self.next_frame_time, float(data.time) + self.frame_period
            )
        else:
            while self.next_frame_time <= data.time + 1e-9:
                self.next_frame_time += self.frame_period

    def save(self, path: Path) -> None:
        if not self.frames:
            raise RuntimeError("No frames were captured for the GIF.")

        duration_ms = max(1, round(1000 / self.fps))
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = None if args.fixed else np.random.default_rng(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    handles = ModelHandles(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    success_count = 0
    try:
        for episode in range(1, args.episodes + 1):
            cube_position, target_position = reset_episode(
                model, data, handles, rng=rng
            )
            recorder = GifRecorder(renderer, args.fps)
            recorder.capture(data, force=True)

            success = False
            failure_reason: str | None = None
            try:
                controller = run_episode(
                    model,
                    data,
                    handles,
                    args.max_duration,
                    print_interval=args.max_duration + 1.0,
                    viewer=None,
                    verbose=False,
                    frame_callback=recorder.capture,
                )
                success = bool(controller.done and controller.success)
                if not success:
                    failure_reason = "Final placement condition was not satisfied."
            except (RuntimeError, np.linalg.LinAlgError) as error:
                failure_reason = str(error)

            recorder.capture(data, force=True)
            status = "success" if success else "failure"
            output_path = output_dir / (
                f"scripted_{run_id}_episode_{episode:03d}_{status}.gif"
            )
            recorder.save(output_path)

            if success:
                success_count += 1
            print(
                f"Episode {episode:03d}: {status.upper()} | "
                f"cube={np.round(cube_position[:2], 3)} | "
                f"target={np.round(target_position[:2], 3)} | "
                f"frames={len(recorder.frames)}"
            )
            print(f"GIF: {output_path}")
            if failure_reason:
                print(f"Reason: {failure_reason}")
    finally:
        renderer.close()

    print(
        f"\nRecorded {args.episodes} episode(s): "
        f"{success_count} success, {args.episodes - success_count} failure."
    )
    print(f"Media directory: {output_dir}")


if __name__ == "__main__":
    main()
