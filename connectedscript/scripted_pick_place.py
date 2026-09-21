"""Scripted Franka Panda pick-and-place using waypoint IK and feedback.

The policy is deliberately simple: it builds Cartesian waypoints from the
current cube and target positions, solves each waypoint with damped-least-
squares Jacobian IK, and advances a state machine only after the robot reaches
the current goal.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "models" / "pick_place_scene.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")

HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float
)
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

PREGRASP_CLEARANCE = 0.14
LIFT_HEIGHT = 0.20
PLACE_HEIGHT = 0.035
POSITION_TOLERANCE = 0.012
MAX_JOINT_TARGET_SPEED = 0.75

FIXED_CUBE_POSITION = np.array([0.45, 0.10, 0.026], dtype=float)
FIXED_TARGET_POSITION = np.array([0.45, -0.20, 0.003], dtype=float)
CUBE_X_RANGE = (0.3, 0.5)
CUBE_Y_RANGE = (0.06, 0.20)
TARGET_X_RANGE = (0.38, 0.52)
TARGET_Y_RANGE = (-0.24, -0.06)
MIN_CUBE_TARGET_DISTANCE = 0.16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a scripted Panda pickup and putdown task."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the interactive viewer.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=45.0,
        help="Maximum simulation time in seconds (default: 45).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Simulation seconds between status lines (default: 0.5).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the cube and target positions before the episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --randomize (default: 0).",
    )
    args = parser.parse_args()
    if args.max_duration <= 0:
        parser.error("--max-duration must be positive")
    if args.print_interval <= 0:
        parser.error("--print-interval must be positive")
    return args


def require_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ValueError(f"Required MuJoCo object not found: {name!r}")
    return object_id


class ModelHandles:
    """Named MuJoCo objects resolved once at startup."""

    def __init__(self, model: mujoco.MjModel) -> None:
        arm_joint_ids = np.array(
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

        self.arm_qpos = model.jnt_qposadr[arm_joint_ids]
        self.arm_dofs = model.jnt_dofadr[arm_joint_ids]#这个dofs是什么东西 是关节自由度索引吗 请给出缩写的具体名称
        self.arm_ranges = model.jnt_range[arm_joint_ids].copy() #这又是什么东西
        self.finger_qpos = model.jnt_qposadr[finger_joint_ids]
        self.arm_actuators = np.array(
            [
                require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ]
        )
        self.gripper_actuator = require_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.grasp_site = require_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )#这是什么 
        self.cube_body = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")#这是什么 
        cube_joint = require_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        self.cube_qpos = int(model.jnt_qposadr[cube_joint])
        self.cube_qvel = int(model.jnt_dofadr[cube_joint])
        self.target_site = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.scene_home_key = require_id(
            model, mujoco.mjtObj.mjOBJ_KEY, "scene_home"
        )#这是什么
        self.overview_camera = require_id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )


def sample_scene_positions(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a conservative, reachable cube/target pair."""
    for _ in range(100):
        cube = np.array(
            [
                rng.uniform(*CUBE_X_RANGE),
                rng.uniform(*CUBE_Y_RANGE),
                FIXED_CUBE_POSITION[2],
            ]
        )
        target = np.array(
            [
                rng.uniform(*TARGET_X_RANGE),
                rng.uniform(*TARGET_Y_RANGE),
                FIXED_TARGET_POSITION[2],
            ]
        )
        if np.linalg.norm(cube[:2] - target[:2]) >= MIN_CUBE_TARGET_DISTANCE:
            return cube, target
    raise RuntimeError("Could not sample a valid cube/target pair.")


def reset_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reset the robot and optionally randomize one task configuration."""
    mujoco.mj_resetDataKeyframe(model, data, handles.scene_home_key)

    if rng is None:
        cube_position = FIXED_CUBE_POSITION.copy()
        target_position = FIXED_TARGET_POSITION.copy()
    else:
        cube_position, target_position = sample_scene_positions(rng)

    # Freejoint qpos layout: xyz position followed by a wxyz quaternion.
    data.qpos[handles.cube_qpos : handles.cube_qpos + 3] = cube_position
    data.qpos[handles.cube_qpos + 3 : handles.cube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[handles.cube_qvel : handles.cube_qvel + 6] = 0.0

    # Target is a site directly under worldbody, so site_pos is world-relative.
    model.site_pos[handles.target_site] = target_position
    mujoco.mj_forward(model, data)
    return cube_position, target_position


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Small-angle orientation error expressed in world coordinates."""
    return 0.5 * sum(
        np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
    ) #这是什么 这段代码是什么意思


def solve_ik(
    model: mujoco.MjModel,
    seed_data: mujoco.MjData,
    handles: ModelHandles,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    max_iterations: int = 600,
) -> tuple[np.ndarray, float, float]:
    """Solve a 6D grasp-site pose with damped least-squares Jacobian IK."""
    work = mujoco.MjData(model)
    work.qpos[:] = seed_data.qpos
    work.qvel[:] = 0.0

    jacobian_position = np.zeros((3, model.nv))#这段代码是干嘛的 为什么要把nv作为第二维度并且设置为0
    jacobian_rotation = np.zeros((3, model.nv))
    damping = 0.03#这又是什么 

    for _ in range(max_iterations):
        mujoco.mj_forward(model, work) #为什么这里要用forward来重新渲染
        current_position = work.site_xpos[handles.grasp_site]
        current_rotation = work.site_xmat[handles.grasp_site].reshape(3, 3)
        position_delta = target_position - current_position
        rotation_delta = orientation_error(current_rotation, target_rotation)

        position_norm = float(np.linalg.norm(position_delta))
        rotation_norm = float(np.linalg.norm(rotation_delta))
        if position_norm < 0.0015 and rotation_norm < 0.008:
            return work.qpos[handles.arm_qpos].copy(), position_norm, rotation_norm

        mujoco.mj_jacSite(
            model,
            work,
            jacobian_position,
            jacobian_rotation,
            handles.grasp_site,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, handles.arm_dofs],
                0.6 * jacobian_rotation[:, handles.arm_dofs],
            )
        )#这一步是干什么 
        error = np.concatenate((position_delta, 0.6 * rotation_delta))

        system = jacobian @ jacobian.T + (damping**2) * np.eye(6)#这段代码是什么意思
        joint_delta = jacobian.T @ np.linalg.solve(system, error)#这段是什么意思 为什么会有@
        max_delta = float(np.max(np.abs(joint_delta)))
        if max_delta > 0.18:
            joint_delta *= 0.18 / max_delta #为了规避singularity吗

        work.qpos[handles.arm_qpos] += 0.55 * joint_delta
        lower = handles.arm_ranges[:, 0] + 0.01
        upper = handles.arm_ranges[:, 1] - 0.01
        work.qpos[handles.arm_qpos] = np.clip(
            work.qpos[handles.arm_qpos], lower, upper
        )#这一串代码是什么意思 是在算IK的迭代吗 

    mujoco.mj_forward(model, work)
    position_norm = float(
        np.linalg.norm(target_position - work.site_xpos[handles.grasp_site])
    )
    rotation_norm = float(
        np.linalg.norm(
            orientation_error(
                work.site_xmat[handles.grasp_site].reshape(3, 3), target_rotation
            )
        )
    )
    raise RuntimeError(
        "IK failed to converge: "
        f"position error={position_norm:.4f} m, "
        f"orientation error={rotation_norm:.4f}"
    )


class ScriptedPickPlaceController:
    """Feedback state machine for one fixed-scene pick-and-place episode."""

    MOVE_PHASES = {
        "move_above_cube",
        "descend_to_cube",
        "lift_cube",
        "move_above_target",
        "descend_to_target",
        "retreat",
    }

    PHASE_TIMEOUTS = {
        "move_above_cube": 10.0,
        "descend_to_cube": 8.0,
        "close_gripper": 2.0,
        "lift_cube": 8.0,
        "move_above_target": 10.0,
        "descend_to_target": 8.0,
        "open_gripper": 2.0,
        "retreat": 8.0,
        "return_home": 10.0,
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        handles: ModelHandles,
        *,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.handles = handles
        self.phase = "move_above_cube"
        self.phase_started = float(data.time)
        self.done = False
        self.success = False
        self.verbose = verbose

        self.pick_position = data.xpos[handles.cube_body].copy()
        target_marker = data.site_xpos[handles.target_site].copy()
        self.place_position = np.array(
            [target_marker[0], target_marker[1], PLACE_HEIGHT], dtype=float
        )
        self.pregrasp_position = self.pick_position + np.array(
            [0.0, 0.0, PREGRASP_CLEARANCE]
        )
        self.lift_position = np.array(
            [self.pick_position[0], self.pick_position[1], LIFT_HEIGHT], dtype=float
        )
        self.preplace_position = np.array(
            [target_marker[0], target_marker[1], LIFT_HEIGHT], dtype=float
        )
        self.target_rotation = data.site_xmat[handles.grasp_site].reshape(3, 3).copy()

        self.joint_command = data.qpos[handles.arm_qpos].copy()
        self.joint_goal = self.joint_command.copy()
        self.gripper_command = GRIPPER_OPEN
        self.cartesian_goal = self.pregrasp_position.copy()
        self._configure_move(data, self.cartesian_goal)
        self._announce_phase(data)

    def _announce_phase(self, data: mujoco.MjData) -> None:
        if not self.verbose:
            return
        print(f"\n--- Phase: {self.phase} at {data.time:.2f} s ---")
        if self.phase in self.MOVE_PHASES:
            print("Cartesian goal:", np.round(self.cartesian_goal, 4))

    def _configure_move(
        self, data: mujoco.MjData, cartesian_goal: np.ndarray
    ) -> None:
        self.cartesian_goal = np.asarray(cartesian_goal, dtype=float).copy()
        self.joint_goal, position_error, rotation_error_value = solve_ik(
            self.model,
            data,
            self.handles,
            self.cartesian_goal,
            self.target_rotation,
        )
        if self.verbose:
            print(
                "IK solved: "
                f"position error={position_error:.6f} m, "
                f"orientation error={rotation_error_value:.6f}"
            )

    def _set_phase(
        self,
        phase: str,
        data: mujoco.MjData,
        cartesian_goal: np.ndarray | None = None,
    ) -> None:
        self.phase = phase
        self.phase_started = float(data.time)
        if cartesian_goal is not None:
            self._configure_move(data, cartesian_goal)
        self._announce_phase(data)

    def _move_joint_command_toward_goal(self) -> None:
        max_step = MAX_JOINT_TARGET_SPEED * self.model.opt.timestep
        difference = self.joint_goal - self.joint_command
        self.joint_command += np.clip(difference, -max_step, max_step)

    def _reached_cartesian_goal(self, data: mujoco.MjData) -> bool:
        error = np.linalg.norm(
            data.site_xpos[self.handles.grasp_site] - self.cartesian_goal
        )
        return bool(error < POSITION_TOLERANCE)

    def _check_timeout(self, data: mujoco.MjData) -> None:
        timeout = self.PHASE_TIMEOUTS.get(self.phase)
        if timeout is not None and data.time - self.phase_started > timeout:
            raise RuntimeError(f"Phase timed out: {self.phase}")

    def _cube_is_lifted(self, data: mujoco.MjData) -> bool:
        return bool(data.xpos[self.handles.cube_body, 2] > 0.08)

    def _evaluate_success(self, data: mujoco.MjData) -> bool:
        cube_position = data.xpos[self.handles.cube_body]
        target_position = data.site_xpos[self.handles.target_site]
        xy_error = float(np.linalg.norm(cube_position[:2] - target_position[:2]))
        height_ok = 0.015 < cube_position[2] < 0.055
        if self.verbose:
            print(f"Final cube-target XY error: {xy_error:.4f} m")
            print(f"Final cube height: {cube_position[2]:.4f} m")
        return xy_error < 0.045 and height_ok

    def update(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        self._check_timeout(data)
        elapsed = float(data.time - self.phase_started)

        if self.phase in self.MOVE_PHASES:
            self._move_joint_command_toward_goal()

        if self.phase == "move_above_cube" and self._reached_cartesian_goal(data):
            self._set_phase("descend_to_cube", data, self.pick_position)
        elif self.phase == "descend_to_cube" and self._reached_cartesian_goal(data):
            self.gripper_command = GRIPPER_CLOSED
            self._set_phase("close_gripper", data)
        elif self.phase == "close_gripper" and elapsed > 1.2:
            self._set_phase("lift_cube", data, self.lift_position)
        elif self.phase == "lift_cube" and self._reached_cartesian_goal(data):
            if not self._cube_is_lifted(data):
                raise RuntimeError("Grasp failed: the gripper moved up without the cube.")
            self._set_phase("move_above_target", data, self.preplace_position)
        elif self.phase == "move_above_target" and self._reached_cartesian_goal(data):
            self._set_phase("descend_to_target", data, self.place_position)
        elif self.phase == "descend_to_target" and self._reached_cartesian_goal(data):
            self.gripper_command = GRIPPER_OPEN
            self._set_phase("open_gripper", data)
        elif self.phase == "open_gripper" and elapsed > 1.2:
            self._set_phase("retreat", data, self.preplace_position)
        elif self.phase == "retreat" and self._reached_cartesian_goal(data):
            self.joint_goal = HOME_QPOS.copy()
            self._set_phase("return_home", data)
        elif self.phase == "return_home":
            self._move_joint_command_toward_goal()
            arm_error = np.max(
                np.abs(data.qpos[self.handles.arm_qpos] - HOME_QPOS)
            )
            if arm_error < 0.025 and elapsed > 0.5:
                self.success = self._evaluate_success(data)
                self.phase = "done"
                self.done = True
                self._announce_phase(data)

        return self.joint_command.copy(), self.gripper_command


def print_status(
    data: mujoco.MjData,
    handles: ModelHandles,
    controller: ScriptedPickPlaceController,
) -> None:
    ee = data.site_xpos[handles.grasp_site]
    cube = data.xpos[handles.cube_body]
    target = data.site_xpos[handles.target_site]
    gripper_width = float(np.sum(data.qpos[handles.finger_qpos]))
    if controller.phase in controller.MOVE_PHASES:
        ee_error = float(np.linalg.norm(ee - controller.cartesian_goal))
    else:
        ee_error = 0.0

    print(
        f"[{data.time:5.2f} s] {controller.phase:>18} | "
        f"EE={np.round(ee, 3)} | cube={np.round(cube, 3)} | "
        f"target={np.round(target, 3)} | grip={gripper_width:.3f} m | "
        f"EE error={ee_error:.3f} m"
    )


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    max_duration: float,
    print_interval: float,
    viewer: object | None,
    *,
    verbose: bool = True,
    frame_callback: Callable[[mujoco.MjData], None] | None = None,
) -> ScriptedPickPlaceController:
    controller = ScriptedPickPlaceController(
        model, data, handles, verbose=verbose
    )
    next_print_time = 0.0

    while not controller.done and data.time < max_duration:
        if viewer is not None and not viewer.is_running():
            print("Viewer closed; ending episode.")
            break

        step_started = time.perf_counter()
        joint_command, gripper_command = controller.update(data)
        data.ctrl[handles.arm_actuators] = joint_command
        data.ctrl[handles.gripper_actuator] = gripper_command
        mujoco.mj_step(model, data)

        if frame_callback is not None:
            frame_callback(data)

        if verbose and data.time + 1e-9 >= next_print_time:
            print_status(data, handles, controller)
            next_print_time += print_interval

        if viewer is not None:
            viewer.sync()
            elapsed = time.perf_counter() - step_started
            time.sleep(max(0.0, model.opt.timestep - elapsed))

    if not controller.done and data.time >= max_duration:
        raise RuntimeError(f"Episode exceeded {max_duration:.1f} simulation seconds.")
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation produced NaN or infinite state values.")
    return controller


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    handles = ModelHandles(model)
    rng = np.random.default_rng(args.seed) if args.randomize else None
    reset_episode(model, data, handles, rng=rng)

    print(f"Loaded scene: {SCENE_PATH}")
    print("Initial cube position:", np.round(data.xpos[handles.cube_body], 4))
    print("Target position:", np.round(data.site_xpos[handles.target_site], 4))

    if args.headless:
        controller = run_episode(
            model,
            data,
            handles,
            args.max_duration,
            args.print_interval,
            viewer=None,
        )
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = handles.overview_camera
            controller = run_episode(
                model,
                data,
                handles,
                args.max_duration,
                args.print_interval,
                viewer,
            )

    if not controller.done:
        raise RuntimeError("Episode ended before the state machine finished.")
    if not controller.success:
        raise RuntimeError("Pick-and-place completed, but the final placement failed.")
    print("\nPick-and-place completed successfully.")


if __name__ == "__main__":
    main()
