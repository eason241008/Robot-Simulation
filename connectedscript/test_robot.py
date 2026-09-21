"""Basic control and state-reading smoke test for the Franka Panda scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "models" / "pick_place_scene.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
GRIPPER_ACTUATOR_NAME = "actuator8"

HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float
)
POSE_A_QPOS = np.array(
    [0.30, -0.20, 0.18, -1.75, 0.12, 1.78, -0.55], dtype=float
)
POSE_B_QPOS = np.array(
    [-0.25, 0.18, -0.16, -1.35, -0.12, 1.42, -1.00], dtype=float
)

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
CYCLE_DURATION = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Panda joints, exercise the gripper, and print robot state."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive MuJoCo viewer.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=CYCLE_DURATION,
        help=f"Simulation duration in seconds (default: {CYCLE_DURATION}).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds of simulation time between state reports (default: 0.5).",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.print_interval <= 0:
        parser.error("--print-interval must be positive")
    return args


def require_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ValueError(f"Required MuJoCo object not found: {name!r}")
    return object_id


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


   


def command_for_time(simulation_time: float) -> tuple[np.ndarray, float, str]:
    """Return arm target, gripper command, and phase label for one cycle."""
    if simulation_time < 1.0:
        return HOME_QPOS, GRIPPER_OPEN, "Home"
    if simulation_time < 3.0:
        arm = interpolate(HOME_QPOS, POSE_A_QPOS, 1.0, 3.0, simulation_time)
        return np.asarray(arm), GRIPPER_OPEN, "Move to pose A"
    if simulation_time < 5.0:
        arm = interpolate(POSE_A_QPOS, POSE_B_QPOS, 3.0, 5.0, simulation_time)
        return np.asarray(arm), GRIPPER_OPEN, "Move to pose B"
    if simulation_time < 6.0:
        gripper = interpolate(
            GRIPPER_OPEN, GRIPPER_CLOSED, 5.0, 6.0, simulation_time
        )
        return POSE_B_QPOS, float(gripper), "Close gripper"
    if simulation_time < 7.0:
        return POSE_B_QPOS, GRIPPER_CLOSED, "Hold gripper closed"
    if simulation_time < 8.0:
        gripper = interpolate(
            GRIPPER_CLOSED, GRIPPER_OPEN, 7.0, 8.0, simulation_time
        )
        return POSE_B_QPOS, float(gripper), "Open gripper"
    if simulation_time < 10.5:
        arm = interpolate(POSE_B_QPOS, HOME_QPOS, 8.0, 10.5, simulation_time)
        return np.asarray(arm), GRIPPER_OPEN, "Return home"
    return HOME_QPOS, GRIPPER_OPEN, "Home complete"


class RobotHandles:
    """Resolved model indices used by the controller and state reader."""

    def __init__(self, model: mujoco.MjModel) -> None:
        joint_ids = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ]
        )
        finger_joint_ids = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINT_NAMES
            ]
        )

        self.arm_qpos_addresses = model.jnt_qposadr[joint_ids]
        self.arm_qvel_addresses = model.jnt_dofadr[joint_ids]
        self.finger_qpos_addresses = model.jnt_qposadr[finger_joint_ids]
        self.arm_actuator_ids = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ]
        )
        self.gripper_actuator_id = require_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self.hand_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cube_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.scene_home_key_id = require_id(
            model, mujoco.mjtObj.mjOBJ_KEY, "scene_home"
        )
        self.overview_camera_id = require_id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )


def reset_scene(
    model: mujoco.MjModel, data: mujoco.MjData, handles: RobotHandles
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, handles.scene_home_key_id)
    mujoco.mj_forward(model, data)


def apply_command(
    data: mujoco.MjData,
    handles: RobotHandles,
    arm_target: np.ndarray,
    gripper_command: float,
) -> None:
    data.ctrl[handles.arm_actuator_ids] = arm_target
    data.ctrl[handles.gripper_actuator_id] = gripper_command


def print_state(data: mujoco.MjData, handles: RobotHandles, phase: str) -> None:
    joint_positions = data.qpos[handles.arm_qpos_addresses]
    joint_velocities = data.qvel[handles.arm_qvel_addresses]
    gripper_width = float(np.sum(data.qpos[handles.finger_qpos_addresses]))
    ee_position = data.xpos[handles.hand_body_id]
    cube_position = data.xpos[handles.cube_body_id]
    target_position = data.site_xpos[handles.target_site_id]

    print(f"\n[{data.time:5.2f} s] {phase}")
    print("  Joint position:", np.round(joint_positions, 3))
    print("  Joint velocity:", np.round(joint_velocities, 3))
    print("  EE position:   ", np.round(ee_position, 3))
    print("  Cube position: ", np.round(cube_position, 3))
    print("  Target position:", np.round(target_position, 3))
    print(f"  Gripper width:  {gripper_width:.4f} m")


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    duration: float,
    print_interval: float,
    viewer: object | None = None,
) -> None:
    next_print_time = 0.0
    previous_phase: str | None = None

    while data.time < duration:
        if viewer is not None and not viewer.is_running():
            print("Viewer closed; ending simulation.")
            break

        step_started = time.perf_counter()
        arm_target, gripper_command, phase = command_for_time(data.time)
        apply_command(data, handles, arm_target, gripper_command)
        mujoco.mj_step(model, data)

        if phase != previous_phase:
            print(f"\n--- Phase: {phase} ---")
            previous_phase = phase

        if data.time + 1e-9 >= next_print_time:
            print_state(data, handles, phase)
            next_print_time += print_interval

        if viewer is not None:
            viewer.sync()
            step_elapsed = time.perf_counter() - step_started
            time.sleep(max(0.0, model.opt.timestep - step_elapsed))


def validate_final_state(
    data: mujoco.MjData, handles: RobotHandles, completed_cycle: bool
) -> None:
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation produced NaN or infinite state values.")

    if completed_cycle:
        arm_qpos = data.qpos[handles.arm_qpos_addresses]
        home_error = float(np.max(np.abs(arm_qpos - HOME_QPOS)))
        print(f"\nMaximum final home error: {home_error:.6f} rad")
        if home_error > 0.05:
            raise RuntimeError("Robot did not return sufficiently close to Home.")

    print("Robot smoke test completed successfully.")


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    handles = RobotHandles(model)
    reset_scene(model, data, handles)

    print(f"Loaded scene: {SCENE_PATH}")
    print(f"Model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    if args.headless:
        run_simulation(
            model, data, handles, args.duration, args.print_interval, viewer=None
        )
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = handles.overview_camera_id
            run_simulation(
                model, data, handles, args.duration, args.print_interval, viewer
            )

    validate_final_state(data, handles, args.duration >= CYCLE_DURATION)


if __name__ == "__main__":
    main()
