# Third-party assets

## `unitree_a1/` — Unitree A1 robot description

The robot model, meshes and textures in `unitree_a1/` are **not my work**. They
are the Unitree A1 description from [MuJoCo Menagerie][menagerie], copyright
Unitree Robotics, released under the BSD-3-Clause license. The full license text
ships unmodified at [`unitree_a1/LICENSE`](unitree_a1/LICENSE), alongside the
upstream [`README.md`](unitree_a1/README.md) and
[`CHANGELOG.md`](unitree_a1/CHANGELOG.md).

[menagerie]: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_a1

### Local modifications

`unitree_a1/a1.xml` and the mesh assets are unmodified upstream files.

`unitree_a1/scene.xml` **has been modified** from upstream. The additions are
the task furniture this project needs, and nothing else:

- 24 non-colliding cylinder markers at 15-degree intervals on a 3 m circle,
  which draw the ring goals are sampled from.
- A mocap body named `goal_marker` — the green disc the training and evaluation
  scripts move to the current goal position.

Both additions set `contype="0" conaffinity="0"`, so they are visual only and
do not affect the physics the policies were trained against.
