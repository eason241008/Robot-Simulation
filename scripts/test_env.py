"""Small executable smoke test for PickPlaceEnv."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.pick_place_env import PickPlaceEnv


def main() -> None:
    env = PickPlaceEnv(randomize=True)
    try:
        observation = env.reset(seed=42)
        first_cube = env.cube_position.copy()
        first_target = env.target_position.copy()

        repeated_observation = env.reset(seed=42)
        assert np.allclose(observation, repeated_observation)
        assert np.allclose(first_cube, env.cube_position)
        assert np.allclose(first_target, env.target_position)
        assert observation.shape == (21,)
        assert env.observation_space.contains(observation)

        different_observation = env.reset(seed=43)
        assert not np.allclose(observation[15:21], different_observation[15:21])

        action = np.array([0.0, 0.0, -0.25, 1.0], dtype=np.float32)
        next_observation, reward, done, info = env.step(action)
        assert next_observation.shape == (21,)
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        print("PickPlaceEnv smoke test passed.")
        print("Observation dimension:", env.observation_space.shape)
        print("Action dimension:", env.action_space.shape)
        print("Cube position:", np.round(env.cube_position, 4))
        print("Target position:", np.round(env.target_position, 4))
        print("Step info:", info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
