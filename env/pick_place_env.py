"""Gym-like state-based Franka Panda pick-and-place environment.

Observation (21 values):
    arm qpos (7), arm qvel (7), gripper width (1), cube xyz (3),
    target xyz (3).

Action (4 normalized values in [-1, 1]):
    delta x, delta y, delta z, gripper command.

The xyz values are multiplied by ``max_cartesian_delta``. A non-negative
gripper command opens the gripper and a negative command closes it.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "models" / "pick_place_scene.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
FIXED_CUBE_POSITION = np.array([0.45, 0.10, 0.026], dtype=float)
FIXED_TARGET_POSITION = np.array([0.45, -0.20, 0.003], dtype=float)


def _require_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise ValueError(f"Required MuJoCo object not found: {name!r}")
    return object_id


def _orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Return a small-angle orientation error in world coordinates."""
    return 0.5 * sum(
        np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
    )


class _ModelHandles:
    """Indices into MuJoCo arrays, resolved once during environment creation."""

    def __init__(self, model: mujoco.MjModel) -> None:
        arm_joint_ids = np.array(
            [
                _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ]
        )
        finger_joint_ids = np.array(
            [
                _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINT_NAMES
            ]
        )
        cube_joint_id = _require_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )

        self.arm_qpos = model.jnt_qposadr[arm_joint_ids]
        self.arm_dofs = model.jnt_dofadr[arm_joint_ids]
        self.arm_ranges = model.jnt_range[arm_joint_ids].copy()
        self.finger_qpos = model.jnt_qposadr[finger_joint_ids]
        self.arm_actuators = np.array(
            [
                _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ]
        )
        self.gripper_actuator = _require_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.grasp_site = _require_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self.cube_body = _require_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.cube_qpos = int(model.jnt_qposadr[cube_joint_id])
        self.cube_qvel = int(model.jnt_dofadr[cube_joint_id])
        self.target_site = _require_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.scene_home_key = _require_id(
            model, mujoco.mjtObj.mjOBJ_KEY, "scene_home"
        )


class PickPlaceEnv:
    """A compact Robot Learning environment around the MuJoCo scene."""

    observation_dim = 21
    action_dim = 4

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE_PATH,
        *,
        randomize: bool = True,
        frame_skip: int = 25,
        max_episode_steps: int = 400,
        max_cartesian_delta: float = 0.02,
        cube_x_range: tuple[float, float] = (0.38, 0.52),
        cube_y_range: tuple[float, float] = (0.06, 0.20),
        target_x_range: tuple[float, float] = (0.38, 0.52),
        target_y_range: tuple[float, float] = (-0.24, -0.06),
        minimum_task_distance: float = 0.16,
    ) -> None:
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if max_cartesian_delta <= 0:
            raise ValueError("max_cartesian_delta must be positive")

        self.scene_path = Path(scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.handles = _ModelHandles(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self.randomize = randomize
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.max_cartesian_delta = max_cartesian_delta
        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.minimum_task_distance = minimum_task_distance
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._target_rotation = np.eye(3)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        observation_low = np.concatenate(
            (
                self.handles.arm_ranges[:, 0],
                np.full(7, -np.inf),
                [0.0],
                np.full(6, -np.inf),
            )
        )
        observation_high = np.concatenate(
            (
                self.handles.arm_ranges[:, 1],
                np.full(7, np.inf),
                [0.08],
                np.full(6, np.inf),
            )
        )
        self.observation_space = gym.spaces.Box(
            low=observation_low.astype(np.float32),
            high=observation_high.astype(np.float32),
            dtype=np.float32,
        )

        self.reset()

    @property
    def ee_position(self) -> np.ndarray:
        """Current world position of the grasp site."""
        return self.data.site_xpos[self.handles.grasp_site].copy()

    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self.handles.cube_body].copy()

    @property
    def target_position(self) -> np.ndarray:
        return self.data.site_xpos[self.handles.target_site].copy()

    @property
    def step_count(self) -> int:
        return self._step_count

    def _sample_task(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(100):
            cube = np.array(
                [
                    self._rng.uniform(*self.cube_x_range),
                    self._rng.uniform(*self.cube_y_range),
                    FIXED_CUBE_POSITION[2],
                ],
                dtype=float,
            )
            target = np.array(
                [
                    self._rng.uniform(*self.target_x_range),
                    self._rng.uniform(*self.target_y_range),
                    FIXED_TARGET_POSITION[2],
                ],
                dtype=float,
            )
            if np.linalg.norm(cube[:2] - target[:2]) >= self.minimum_task_distance:
                return cube, target
        raise RuntimeError("Could not sample a valid cube/target pair.")

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the robot, randomize the task, and return one observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.handles.scene_home_key
        )
        if self.randomize:
            cube_position, target_position = self._sample_task()
        else:
            cube_position = FIXED_CUBE_POSITION.copy()
            target_position = FIXED_TARGET_POSITION.copy()

        cube_start = self.handles.cube_qpos
        self.data.qpos[cube_start : cube_start + 3] = cube_position
        self.data.qpos[cube_start + 3 : cube_start + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[
            self.handles.cube_qvel : self.handles.cube_qvel + 6
        ] = 0.0
        self.model.site_pos[self.handles.target_site] = target_position

        mujoco.mj_forward(self.model, self.data)
        self._target_rotation = self.data.site_xmat[
            self.handles.grasp_site
        ].reshape(3, 3).copy()
        self._step_count = 0
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Return [q, qvel, gripper width, cube xyz, target xyz]."""
        arm_qpos = self.data.qpos[self.handles.arm_qpos]
        arm_qvel = self.data.qvel[self.handles.arm_dofs]
        gripper_width = np.sum(self.data.qpos[self.handles.finger_qpos])
        observation = np.concatenate(
            (
                arm_qpos,
                arm_qvel,
                [gripper_width],
                self.cube_position,
                self.target_position,
            )
        ).astype(np.float32)
        return observation

    def _solve_ik(
        self,
        target_position: np.ndarray,
        *,
        max_iterations: int = 300,
    ) -> tuple[np.ndarray, float, float]:
        """Solve the grasp-site pose with damped least-squares Jacobian IK."""
        work = self._ik_data
        work.qpos[:] = self.data.qpos
        work.qvel[:] = 0.0
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        damping = 0.03

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, work)
            current_position = work.site_xpos[self.handles.grasp_site]
            current_rotation = work.site_xmat[
                self.handles.grasp_site
            ].reshape(3, 3)
            position_delta = target_position - current_position
            rotation_delta = _orientation_error(
                current_rotation, self._target_rotation
            )
            position_error = float(np.linalg.norm(position_delta))
            rotation_error = float(np.linalg.norm(rotation_delta))
            if position_error < 0.0015 and rotation_error < 0.008:
                return (
                    work.qpos[self.handles.arm_qpos].copy(),
                    position_error,
                    rotation_error,
                )

            mujoco.mj_jacSite(
                self.model,
                work,
                jacobian_position,
                jacobian_rotation,
                self.handles.grasp_site,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.handles.arm_dofs],
                    0.6 * jacobian_rotation[:, self.handles.arm_dofs],
                )
            )
            error = np.concatenate((position_delta, 0.6 * rotation_delta))
            system = jacobian @ jacobian.T + (damping**2) * np.eye(6)
            joint_delta = jacobian.T @ np.linalg.solve(system, error)
            largest_delta = float(np.max(np.abs(joint_delta)))
            if largest_delta > 0.18:
                joint_delta *= 0.18 / largest_delta

            work.qpos[self.handles.arm_qpos] += 0.55 * joint_delta
            lower = self.handles.arm_ranges[:, 0] + 0.01
            upper = self.handles.arm_ranges[:, 1] - 0.01
            work.qpos[self.handles.arm_qpos] = np.clip(
                work.qpos[self.handles.arm_qpos], lower, upper
            )

        mujoco.mj_forward(self.model, work)
        position_error = float(
            np.linalg.norm(
                target_position - work.site_xpos[self.handles.grasp_site]
            )
        )
        rotation_error = float(
            np.linalg.norm(
                _orientation_error(
                    work.site_xmat[self.handles.grasp_site].reshape(3, 3),
                    self._target_rotation,
                )
            )
        )
        raise RuntimeError(
            "IK failed to converge: "
            f"position error={position_error:.4f} m, "
            f"orientation error={rotation_error:.4f}"
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """Apply one normalized Cartesian action and advance the simulation."""
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape {(self.action_dim,)}, "
                f"received {action_array.shape}."
            )
        clipped_action = np.clip(action_array, -1.0, 1.0)
        target_position = self.ee_position + (
            clipped_action[:3] * self.max_cartesian_delta
        )
        target_position = np.clip(
            target_position,
            np.array([0.25, -0.40, 0.02]),
            np.array([0.70, 0.40, 0.60]),
        )

        ik_converged = True
        termination_reason: str | None = None
        try:
            joint_target, position_error, rotation_error = self._solve_ik(
                target_position
            )
        except (RuntimeError, np.linalg.LinAlgError) as error:
            ik_converged = False
            termination_reason = str(error)
            joint_target = self.data.qpos[self.handles.arm_qpos].copy()
            position_error = float("inf")
            rotation_error = float("inf")

        gripper_command = (
            GRIPPER_OPEN if clipped_action[3] >= 0.0 else GRIPPER_CLOSED
        )
        self.data.ctrl[self.handles.arm_actuators] = joint_target
        self.data.ctrl[self.handles.gripper_actuator] = gripper_command
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        observation = self.get_observation()
        success = self.is_success()
        timeout = self._step_count >= self.max_episode_steps
        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        done = success or timeout or not ik_converged or not finite
        if success:
            termination_reason = "success"
        elif timeout:
            termination_reason = "timeout"
        elif not finite:
            termination_reason = "non_finite_state"

        cube_target_xy_distance = float(
            np.linalg.norm(self.cube_position[:2] - self.target_position[:2])
        )
        reward = 1.0 if success else 0.0
        info: dict[str, object] = {
            "success": success,
            "timeout": timeout,
            "ik_converged": ik_converged,
            "ik_position_error": position_error,
            "ik_rotation_error": rotation_error,
            "cube_target_xy_distance": cube_target_xy_distance,
            "gripper_width": float(
                np.sum(self.data.qpos[self.handles.finger_qpos])
            ),
            "step_count": self._step_count,
            "termination_reason": termination_reason,
            "applied_action": clipped_action.copy(),
        }
        return observation, reward, done, info

    def is_success(self) -> bool:
        """Cube is near the target, resting on the table, and released."""
        cube = self.cube_position
        target = self.target_position
        xy_distance = np.linalg.norm(cube[:2] - target[:2])
        cube_on_table = 0.015 < cube[2] < 0.055
        gripper_width = float(
            np.sum(self.data.qpos[self.handles.finger_qpos])
        )
        gripper_released = gripper_width > 0.07
        return bool(xy_distance < 0.045 and cube_on_table and gripper_released)

    def close(self) -> None:
        """Release future rendering resources; currently no persistent viewer."""
        return None
