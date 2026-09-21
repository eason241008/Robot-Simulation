Task:
Move a cube from a randomized initial position to a target position.

Robot:
Franka Emika Panda

Simulator:
MuJoCo

Observation:
7 arm joint positions
7 arm joint velocities
gripper width
cube xyz position
target xyz position

Policy action:
[dx, dy, dz, gripper_command]

Low-level control:
Cartesian command -> IK -> 7 joint position targets

Project structure:

```text
RobotSimulation/
|-- assets/
|   `-- franka_emika_panda/   # Official MuJoCo Menagerie model
|-- models/                   # Project scene XML files
|-- scripts/                  # Executable Python scripts
|-- week1/                    # Initial 2-joint MuJoCo exercise
|-- README.md
`-- requirements.txt
```

Robot model source:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

Development environment:

```powershell
conda activate robotics
python -c "import mujoco; print(mujoco.__version__)"
```

Alternatively, create the project-local virtual environment used for the
latest validation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Panda model has been verified with MuJoCo 3.13.0 in the local `robotics`
Conda environment. Loading `assets/franka_emika_panda/panda.xml` reports 9
generalized positions, 9 generalized velocities, and 8 actuators.

Current scene:

```text
models/pick_place_scene.xml
```

The initial scene contains a fixed table, a free-moving cube at
`[0.45, 0.10, 0.026]`, and a non-colliding target marker at
`[0.45, -0.20, 0.003]`. Positions are intentionally fixed until randomized
reset is implemented in a later milestone.

When resetting the complete scene, use the `scene_home` keyframe. The
robot-only `home` keyframe inherited from `panda.xml` does not contain the
cube's freejoint state.

Load the scene with an absolute path so MuJoCo can resolve the included robot
XML and mesh directory reliably:

```python
from pathlib import Path

import mujoco

scene_path = Path("models/pick_place_scene.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(scene_path))
```

Robot smoke test:

```powershell
conda activate robotics
python scripts/test_robot.py
```

The interactive test runs one 12-second cycle: Home, two arm poses, gripper
close, gripper open, and return to Home. For an automated test without a
viewer, run:

```powershell
python scripts/test_robot.py --headless
```

Scripted pickup and putdown:

```powershell
conda activate robotics
python scripts/scripted_pick_place.py
```

Automated run without a viewer:

```powershell
python scripts/scripted_pick_place.py --headless
```

Run one randomized episode with a reproducible seed:

```powershell
python scripts/scripted_pick_place.py --randomize --seed 7
```

Evaluate 20 randomized episodes:

```powershell
python scripts/evaluate_scripted.py --episodes 20 --seed 42
```

The evaluator saves per-episode configurations, final positions, errors, and
failure reasons to `results/scripted_evaluation.json`.

Record every episode as a separate GIF in `media/`:

```powershell
.\.venv\Scripts\python.exe scripts\record_scripted_gif.py --episodes 5
```

Use `--seed 42` for reproducible positions or omit `--seed` for new random
positions on every run. GIF filenames include the run time, episode number,
and `success` or `failure` status.

Randomized validation result for seed 42:

```text
Episodes: 20
Success: 20
Failure: 0
Success Rate: 100.0%
Final XY errors: 0.0032-0.0054 m
```

The scripted controller reads the live cube and target positions, generates
safe Cartesian waypoints, solves them with Jacobian inverse kinematics, and
uses feedback to decide when to close, lift, place, release, and return Home.

Current fixed-scene validation result (five consecutive headless runs):

```text
Success: 5 / 5
Final cube-target XY error: 0.0047 m
Final cube height: 0.0250 m
```

## Robot Learning environment

`env/pick_place_env.py` provides the project-wide learning interface:

```python
from env.pick_place_env import PickPlaceEnv

env = PickPlaceEnv(randomize=True)
observation = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
```

The observation has 21 values: 7 arm positions, 7 arm velocities, gripper
width, cube XYZ, and target XYZ. The normalized action has four values:
`[delta_x, delta_y, delta_z, gripper]`.

Environment smoke test and scripted-policy evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\test_env.py
.\.venv\Scripts\python.exe scripts\evaluate_env.py --episodes 20 --seed 42
```

Validated results for the environment API:

```text
Previously failing seeds 1007-1216: 20/20 successful
Independent regression seeds 2000-2049: 50/50 successful
```

## Demonstration dataset

Collect 200 successful demonstrations from the scripted expert:

```powershell
.\.venv\Scripts\python.exe scripts\collect_demos.py --episodes 200 --seed 1000
```

Inspect and validate the saved dataset:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
```

The compressed `data/demonstrations.npz` file stores 21-dimensional
observations, 4-dimensional actions, episode boundaries and lengths, phase
labels, initial cube and target positions, final errors, success flags, seeds,
and collection metadata.

Current dataset (base seed 1000):

```text
Successful demonstrations: 200
Collection attempts: 200
Failed attempts excluded: 0
Transitions: 52,318
Episode length: 222-308 steps (mean 261.6)
Final XY error: 0.0036-0.0153 m (mean 0.0102 m)
Dataset size: 3.86 MB
Validation: PASSED
```
