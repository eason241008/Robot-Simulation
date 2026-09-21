# Franka Emika Panda model source

The `franka_emika_panda/` directory is vendored from the official Google
DeepMind MuJoCo Menagerie repository:

- Repository: https://github.com/google-deepmind/mujoco_menagerie
- Source directory: `franka_emika_panda/`
- Commit: `8161bba264d7fa7c99ca301e91e7fb44737676ad`
- License: Apache-2.0 (see `franka_emika_panda/LICENSE`)

The full directory is kept intact because `panda.xml` references mesh and
texture files under `franka_emika_panda/assets/`.

## Project-local modification

`panda.xml` contains one additional non-colliding site named `grasp_site`.
It is located between the fingertip pads and provides a Cartesian frame for
inverse kinematics. No robot dynamics, actuator, joint, geometry, or mesh
parameters were changed.
