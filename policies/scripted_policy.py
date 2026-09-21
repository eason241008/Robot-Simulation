"""Waypoint expert that emits normalized PickPlaceEnv actions."""

from __future__ import annotations

import numpy as np

from env.pick_place_env import PickPlaceEnv


class ScriptedPolicy:
    """State machine whose output is [dx, dy, dz, gripper]."""

    def __init__(
        self,
        env: PickPlaceEnv,
        *,
        pregrasp_clearance: float = 0.14,
        lift_height: float = 0.20,
        place_height: float = 0.035,
        position_tolerance: float = 0.010,
        gripper_hold_seconds: float = 1.2,
        release_wait_seconds: float = 1.2,
        movement_scale: float = 0.70,
        maximum_grasp_separation: float = 0.065,
        max_phase_steps: int = 200,
    ) -> None:
        self.env = env
        self.pregrasp_clearance = pregrasp_clearance
        self.lift_height = lift_height
        self.place_height = place_height
        self.position_tolerance = position_tolerance
        if not 0.0 < movement_scale <= 1.0:
            raise ValueError("movement_scale must be in (0, 1].")
        if maximum_grasp_separation <= 0.0:
            raise ValueError("maximum_grasp_separation must be positive.")
        self.movement_scale = movement_scale
        self.maximum_grasp_separation = maximum_grasp_separation
        self.gripper_hold_steps = max(
            1,
            round(
                gripper_hold_seconds
                / (env.frame_skip * env.model.opt.timestep)
            ),
        )
        self.release_wait_steps = max(
            1,
            round(
                release_wait_seconds
                / (env.frame_skip * env.model.opt.timestep)
            ),
        )
        self.max_phase_steps = max_phase_steps
        self.phase = "idle"
        self.phase_steps = 0
        self.pick_position = np.zeros(3)
        self.target_position = np.zeros(3)
        self.pregrasp_position = np.zeros(3)
        self.lift_position = np.zeros(3)
        self.preplace_position = np.zeros(3)
        self.place_position = np.zeros(3)

    def reset(self, observation: np.ndarray) -> None:
        observation = np.asarray(observation)
        if observation.shape != (self.env.observation_dim,):
            raise ValueError(
                f"Expected observation shape {(self.env.observation_dim,)}, "
                f"received {observation.shape}."
            )
        self.pick_position = observation[15:18].astype(float).copy()
        self.target_position = observation[18:21].astype(float).copy()
        self.pregrasp_position = self.pick_position + np.array(
            [0.0, 0.0, self.pregrasp_clearance]
        )
        self.lift_position = np.array(
            [self.pick_position[0], self.pick_position[1], self.lift_height]
        )
        self.preplace_position = np.array(
            [self.target_position[0], self.target_position[1], self.lift_height]
        )
        self.place_position = np.array(
            [self.target_position[0], self.target_position[1], self.place_height]
        )
        self.phase = "move_above_cube"
        self.phase_steps = 0

    def _transition(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0

    def _position_action(
        self, cartesian_goal: np.ndarray, gripper_action: float
    ) -> np.ndarray:
        position_error = cartesian_goal - self.env.ee_position
        normalized_delta = position_error / self.env.max_cartesian_delta
        action_norm = float(np.linalg.norm(normalized_delta))
        if action_norm > self.movement_scale:
            normalized_delta *= self.movement_scale / action_norm
        return np.array(
            [
                normalized_delta[0],
                normalized_delta[1],
                normalized_delta[2],
                gripper_action,
            ],
            dtype=np.float32,
        )

    def _goal_reached(self, goal: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(self.env.ee_position - goal)
            < self.position_tolerance
        )

    def _assert_cube_is_held(self, observation: np.ndarray) -> None:
        """Fail immediately if the cube separates from the moving gripper."""
        cube_position = observation[15:18]
        separation = float(np.linalg.norm(cube_position - self.env.ee_position))
        if separation > self.maximum_grasp_separation:
            raise RuntimeError(
                "Object dropped during transport: "
                f"cube-gripper separation={separation:.4f} m."
            )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return one normalized action for the current environment state."""
        if self.phase == "idle":
            raise RuntimeError("Call policy.reset(observation) before predict().")
        if self.phase_steps >= self.max_phase_steps:
            raise RuntimeError(f"Scripted policy phase timed out: {self.phase}")

        observation = np.asarray(observation)
        cube_position = observation[15:18]

        # A loop allows a reached waypoint to transition and emit the next
        # phase's action immediately, without inserting a zero-action step.
        for _ in range(8):
            if self.phase == "move_above_cube":
                if self._goal_reached(self.pregrasp_position):
                    self._transition("descend_to_cube")
                    continue
                action = self._position_action(self.pregrasp_position, 1.0)
            elif self.phase == "descend_to_cube":
                if self._goal_reached(self.pick_position):
                    self._transition("close_gripper")
                    continue
                action = self._position_action(self.pick_position, 1.0)
            elif self.phase == "close_gripper":
                if self.phase_steps >= self.gripper_hold_steps:
                    self._transition("lift_cube")
                    continue
                action = self._position_action(self.pick_position, -1.0)
            elif self.phase == "lift_cube":
                if self._goal_reached(self.lift_position):
                    if cube_position[2] <= 0.08:
                        raise RuntimeError(
                            "Grasp failed: end effector lifted without the cube."
                        )
                    self._transition("move_above_target")
                    continue
                action = self._position_action(self.lift_position, -1.0)
            elif self.phase == "move_above_target":
                self._assert_cube_is_held(observation)
                if self._goal_reached(self.preplace_position):
                    self._transition("descend_to_target")
                    continue
                action = self._position_action(self.preplace_position, -1.0)
            elif self.phase == "descend_to_target":
                self._assert_cube_is_held(observation)
                if self._goal_reached(self.place_position):
                    self._transition("open_gripper")
                    continue
                action = self._position_action(self.place_position, -1.0)
            elif self.phase == "open_gripper":
                if self.phase_steps >= self.release_wait_steps:
                    cube_position = observation[15:18]
                    target_position = observation[18:21]
                    xy_distance = float(
                        np.linalg.norm(
                            cube_position[:2] - target_position[:2]
                        )
                    )
                    raise RuntimeError(
                        "Placement failed after release: "
                        f"cube-target distance={xy_distance:.4f} m, "
                        f"cube height={cube_position[2]:.4f} m."
                    )
                action = self._position_action(self.place_position, 1.0)
            else:
                raise RuntimeError(f"Unknown scripted policy phase: {self.phase}")

            self.phase_steps += 1
            return action

        raise RuntimeError("Too many scripted policy transitions in one step.")
