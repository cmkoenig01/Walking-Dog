"""Walk-to-goal environment for the Unitree A1.

This module is the single definition of the task: the observation layout, the
action layout, the reward functions and the termination rule. `train.py`,
`watch.py` and `evaluate.py` all import from here, so training and evaluation
can never silently disagree about what a state vector means.

That matters more than it sounds: the observation layout is a contract with the
trained checkpoints in `checkpoints/`. Reordering a single field here would
quietly invalidate every saved policy, so the layout is frozen and
`tests/test_env.py` asserts it.

Task
----
The A1 spawns at the origin and must reach a goal sampled uniformly on a 3 m
circle. A goal counts as reached when the trunk is within `GOAL_RADIUS` of it.

Control runs at the MuJoCo timestep (2 ms, 500 Hz). Actions are joint-angle
*offsets* from a fixed home pose, so a zero action is a stable standing crouch
and the policy only has to learn the deviation from it.
"""

from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------

SCENE_PATH = Path(__file__).resolve().parent / "unitree_a1" / "scene.xml"

JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

ACTUATOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

FOOT_BODY_NAMES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")

# The trunk is the only free-floating body, so it is body index 1 (0 is world).
TRUNK_BODY_ID = 1

# --------------------------------------------------------------------------
# Task constants
# --------------------------------------------------------------------------

ACTION_SCALE  = 0.25   # rad; joint target = HOME_CTRL + ACTION_SCALE * action
CIRCLE_RADIUS = 3.0    # m; goals are sampled on this circle
GOAL_RADIUS   = 0.4    # m; trunk within this distance counts as "at goal"
STAND_STEPS   = 100    # steps held at the goal before an episode is a success
MAX_EP_LEN    = 7500   # steps (15 s at 2 ms) before an episode is truncated

# Standing crouch the policies act relative to (hip, thigh, calf) x 4 legs.
HOME_CTRL = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

# --------------------------------------------------------------------------
# Observation / action layout — FROZEN, the checkpoints depend on it
# --------------------------------------------------------------------------

NAV_STATE_DIM    = 15
NAV_ACTION_DIM   = 2    # [forward, turn]
WALK_STATE_DIM   = 34   # 32 body features + 2 nav command
WALK_ACTION_DIM  = 12   # one joint offset per actuator


class A1Scene:
    """A loaded MuJoCo model plus the index tables the task needs.

    Index lookups (`model.joint(name).qposadr` and friends) are not free, and
    the reward functions run every step for every environment, so they are
    resolved once here and passed around explicitly rather than recomputed or
    read from module-level globals.
    """

    def __init__(self, model):
        self.model = model
        self.joint_inds = np.array(
            [model.joint(j).qposadr[0] for j in JOINT_NAMES]
        )
        self.ctrl_inds = np.array(
            [model.actuator(a).id for a in ACTUATOR_NAMES]
        )
        self.foot_body_ids = [model.body(n).id for n in FOOT_BODY_NAMES]
        self.foot_body_id_to_idx = {
            bid: i for i, bid in enumerate(self.foot_body_ids)
        }
        goal_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )
        self.goal_mocap_id = model.body_mocapid[goal_body_id]

    @classmethod
    def load(cls, scene_path=SCENE_PATH):
        return cls(mujoco.MjModel.from_xml_path(str(scene_path)))

    def make_data(self):
        """A fresh MjData reset to the home keyframe."""
        data = mujoco.MjData(self.model)
        self.reset(data)
        return data

    def reset(self, data):
        mujoco.mj_resetDataKeyframe(self.model, data, 0)

    @property
    def timestep(self):
        return self.model.opt.timestep


def random_goal(rng):
    """Sample a goal uniformly on the circle. `rng` is a numpy Generator."""
    angle = rng.uniform(0, 2 * np.pi)
    return np.array([
        CIRCLE_RADIUS * np.cos(angle),
        CIRCLE_RADIUS * np.sin(angle),
    ])


def set_goal(scene, data, goal_pos):
    """Move the visual goal marker (a mocap body) to `goal_pos`."""
    data.mocap_pos[scene.goal_mocap_id, 0] = goal_pos[0]
    data.mocap_pos[scene.goal_mocap_id, 1] = goal_pos[1]


def get_foot_contacts(scene, data):
    """Boolean contact flag per foot, ordered FR, FL, RR, RL."""
    contacts = np.zeros(4, dtype=bool)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = scene.model.geom(c.geom1).bodyid[0]
        b2 = scene.model.geom(c.geom2).bodyid[0]
        if b1 in scene.foot_body_id_to_idx:
            contacts[scene.foot_body_id_to_idx[b1]] = True
        if b2 in scene.foot_body_id_to_idx:
            contacts[scene.foot_body_id_to_idx[b2]] = True
    return contacts


def body_yaw(data):
    """Trunk yaw in radians, extracted from the free-joint quaternion."""
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------

def get_nav_state(data, goal_pos):
    """Navigation observation, 15D.

    Deliberately excludes joint state: the nav policy decides *where to go*,
    and giving it 24 joint values would only invite it to overfit to a gait it
    does not control.
    """
    robot_xy     = data.xpos[TRUNK_BODY_ID][:2]
    to_goal      = goal_pos - robot_xy
    dist_to_goal = np.linalg.norm(to_goal)
    dx_norm      = np.clip(to_goal[0] / (CIRCLE_RADIUS * 2), -1.0, 1.0)
    dy_norm      = np.clip(to_goal[1] / (CIRCLE_RADIUS * 2), -1.0, 1.0)
    dist_norm    = np.clip(dist_to_goal / (CIRCLE_RADIUS * 2), 0.0, 1.0)

    robot_yaw  = body_yaw(data)
    goal_angle = np.arctan2(to_goal[1], to_goal[0] + 1e-6)
    angle_err  = goal_angle - robot_yaw

    return np.concatenate([
        data.qpos[3:7],                    # trunk orientation  — 4
        data.qvel[3:6],                    # trunk angular vel  — 3
        [data.xpos[TRUNK_BODY_ID][2]],     # trunk height       — 1
        [dx_norm],                         # dx to goal         — 1
        [dy_norm],                         # dy to goal         — 1
        [dist_norm],                       # dist to goal       — 1
        [np.cos(angle_err)],               # heading error cos  — 1
        [np.sin(angle_err)],               # heading error sin  — 1
        [np.cos(robot_yaw)],               # robot yaw cos      — 1
        [np.sin(robot_yaw)],               # robot yaw sin      — 1
    ])                                     # total: 15


def get_walk_state(data, nav_cmd):
    """Locomotion observation, 34D = 32 body features + the 2D nav command.

    The nav command is appended rather than the goal position: the walk policy
    is told *what to do*, not *where the goal is*, which is what keeps it a
    reusable gait controller instead of a second navigator.
    """
    return np.concatenate([
        data.qpos[3:7],                    # trunk orientation  — 4
        data.qvel[3:6],                    # trunk angular vel  — 3
        data.qpos[7:19],                   # joint positions    — 12
        data.qvel[6:18],                   # joint velocities   — 12
        [data.xpos[TRUNK_BODY_ID][2]],     # trunk height       — 1
        nav_cmd,                           # nav command        — 2
    ])                                     # total: 34


def is_fallen(data):
    """True once the trunk has tipped past ~60 deg or dropped below 15 cm."""
    qx, qy = data.qpos[4], data.qpos[5]
    upright_z = 1.0 - 2.0 * (qx ** 2 + qy ** 2)  # body z dot world z, yaw-free
    return upright_z < 0.5 or data.xpos[TRUNK_BODY_ID][2] < 0.15


def apply_action(scene, data, action):
    """Write a policy action to the actuators as an offset from the home pose."""
    data.ctrl[scene.ctrl_inds] = HOME_CTRL + action * ACTION_SCALE


# --------------------------------------------------------------------------
# Rewards
#
# Both functions are hand-tuned and the coefficients below are the ones the
# shipped checkpoints were trained against. See "Reward design" in the README
# for the reasoning, and "Known issues" for the terms I would change first.
# --------------------------------------------------------------------------

def compute_nav_reward(data, goal_pos, at_goal):
    """Reward for the navigation policy: face the goal, then close on it."""
    robot_xy     = data.xpos[TRUNK_BODY_ID][:2]
    to_goal      = goal_pos - robot_xy
    dist_to_goal = float(np.linalg.norm(to_goal))
    goal_dir     = to_goal / (dist_to_goal + 1e-6)
    body_speed   = float(np.linalg.norm(data.qvel[:2]))

    robot_yaw     = body_yaw(data)
    facing_dir    = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    heading_align = float(np.dot(facing_dir, goal_dir))
    goal_vel      = float(np.dot(data.qvel[:2], goal_dir))
    perp_dir      = np.array([-goal_dir[1], goal_dir[0]])
    lateral_vel   = float(np.dot(data.qvel[:2], perp_dir))
    yaw_signed    = float(data.qvel[5])
    yaw_rate      = abs(yaw_signed)
    goal_angle    = np.arctan2(to_goal[1], to_goal[0] + 1e-6)
    angle_err     = (goal_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi

    if at_goal:
        still_reward = 4.0 * max(0.0, 1.0 - body_speed / 0.3)
        return still_reward + 50.0

    TARGET_SPEED = 0.5
    ALIGN_THRESH = 0.7  # heading_align threshold (~45 degrees)

    facing_reward = 3.0 * max(0.0, heading_align) ** 2
    err_norm      = float(np.clip(angle_err / np.pi, -1.0, 1.0))
    desired_yaw   = err_norm * 1.0
    dist_factor   = min(1.0, dist_to_goal / 0.6)
    goal_bonus    = 50.0 * float(dist_to_goal < GOAL_RADIUS)

    if heading_align >= ALIGN_THRESH:
        # ALIGNED: reward forward speed, gentle correction only
        goal_vel_reward = 5.0 * min(max(0.0, goal_vel), TARGET_SPEED) * dist_factor
        turn_reward     = 2.0 * desired_yaw * float(np.tanh(yaw_signed))
        fast_fwd_pen    = 0.0
    else:
        # MISALIGNED: reward turning hard, allow only crawl speed forward
        goal_vel_reward = 1.0 * min(max(0.0, goal_vel), 0.15)
        turn_reward     = 7.0 * desired_yaw * float(np.tanh(yaw_signed))
        fast_fwd_pen    = 6.0 * max(0.0, goal_vel - 0.15)

    # Being close but misaligned is the worst state to be in: it is where the
    # robot circles the goal forever, so it is penalised hardest.
    misalign_mag           = min(1.0, max(0.0, ALIGN_THRESH - heading_align))
    proximity_misalign_pen = 12.0 * max(0.0, 1.0 - dist_to_goal / 1.5) * misalign_mag

    away_pen         = (8.0 + 8.0 * max(0.0, heading_align)) * max(0.0, -goal_vel)
    stationary_pen   = 2.0 * float(body_speed < 0.05) * max(0.0, 1.0 - abs(err_norm))
    lateral_pen      = 4.0 * lateral_vel ** 2
    speed_excess_pen = 3.0 * max(0.0, goal_vel - TARGET_SPEED)
    fast_spin_pen    = 2.0 * max(0.0, yaw_rate - 1.2)

    return (facing_reward + turn_reward + goal_vel_reward + goal_bonus
            - away_pen - stationary_pen - lateral_pen - speed_excess_pen
            - fast_spin_pen - fast_fwd_pen - proximity_misalign_pen)


def compute_walk_reward(scene, data, action, prev_action, at_goal, nav_cmd):
    """Reward for the locomotion policy: trot cleanly, and obey the command."""
    foot_contacts = get_foot_contacts(scene, data)
    fr, fl, rr, rl = foot_contacts
    n_contacts  = int(np.sum(foot_contacts))
    trot_active = bool((fr == rl) and (fl == rr) and (fr != fl))
    hop_active  = bool((fr == fl) and (rr == rl) and (fr != rr))

    body_speed   = float(np.linalg.norm(data.qvel[:2]))
    robot_yaw    = body_yaw(data)
    facing_dir   = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    perp_dir     = np.array([-facing_dir[1], facing_dir[0]])
    body_fwd_vel = float(np.dot(data.qvel[:2], facing_dir))
    body_lat_vel = float(np.dot(data.qvel[:2], perp_dir))
    tilt         = data.qpos[4] ** 2 + data.qpos[5] ** 2
    fwd_cmd      = float(nav_cmd[0])
    turn_cmd     = float(nav_cmd[1])

    TARGET_SPEED = 0.5

    if at_goal:
        still_reward = 4.0 * max(0.0, 1.0 - body_speed / 0.3)
        upright      = 0.5 * data.qpos[3]
        height_pen   = 2.0 * max(0.0, 0.28 - data.xpos[TRUNK_BODY_ID][2])
        torque_pen   = 0.001 * np.sum(action ** 2)
        return still_reward + upright - height_pen - torque_pen

    foot_heights     = [data.xpos[bid][2] for bid in scene.foot_body_ids]
    clearance_reward = 0.4 * sum(
        max(0.0, h - 0.04) for h, c in zip(foot_heights, foot_contacts) if not c
    )
    actual_yaw_rate = float(data.qvel[5])

    # Diagonal pair sync — FR+RL and FL+RR thigh/calf should match through a trot
    fr_tc = data.qpos[scene.joint_inds[[1, 2]]]
    rl_tc = data.qpos[scene.joint_inds[[10, 11]]]
    fl_tc = data.qpos[scene.joint_inds[[4, 5]]]
    rr_tc = data.qpos[scene.joint_inds[[7, 8]]]
    diag_diff = float(np.sum((fr_tc - rl_tc) ** 2) + np.sum((fl_tc - rr_tc) ** 2))

    trot_reward           = 3.0 * float(trot_active)
    even_gait_reward      = 1.0 * float(trot_active and n_contacts == 2)
    swing_momentum_reward = 0.15 * min(float(np.sum(np.abs(data.qvel[6:18]))), 20.0) * float(trot_active)
    diagonal_sync_pen     = 2.5 * diag_diff
    upright               = 0.3 * data.qpos[3]
    flatness_reward       = 0.5 * max(0.0, 1.0 - 20.0 * tilt)
    cmd_vel_reward        = 4.0 * min(max(0.0, body_fwd_vel), TARGET_SPEED) * max(0.0, fwd_cmd)
    # positive turn_cmd = turn right (positive yaw)
    turn_follow_reward    = 3.0 * float(np.clip(actual_yaw_rate * turn_cmd, -2.0, 2.0))

    fwd_bias          = 1.5 * min(max(0.0, body_fwd_vel), TARGET_SPEED) * max(0.0, fwd_cmd)
    backward_pen      = 15.0 * max(0.0, -body_fwd_vel)
    lateral_scale     = max(0.7, 1.0 - 0.3 * abs(turn_cmd))
    lateral_pen       = (15.0 * abs(body_lat_vel) + 30.0 * body_lat_vel ** 2) * lateral_scale
    hop_pen           = 2.0 * float(hop_active)
    grounded_pen      = 2.0 * float(n_contacts == 4)
    height_pen        = 3.0 * max(0.0, 0.28 - data.xpos[TRUNK_BODY_ID][2])
    tilt_pen          = 15.0 * tilt
    roll_rate_pen     = 3.0 * abs(data.qvel[3])
    torque_pen        = 0.001 * np.sum(action ** 2)
    smoothness        = 0.01 * np.sum((action - prev_action) ** 2)
    joint_vel_pen     = 0.001 * np.sum(data.qvel[6:18] ** 2)
    hip_pen           = 0.15 * np.sum(data.qpos[scene.joint_inds[[0, 3, 6, 9]]] ** 2)
    stationary_pen    = 3.0 * float(body_speed < 0.05) * max(0.0, fwd_cmd)
    unwanted_turn_pen = 1.5 * actual_yaw_rate ** 2 * max(0.0, 1.0 - abs(turn_cmd))

    return (trot_reward + clearance_reward + even_gait_reward + swing_momentum_reward
            + upright + flatness_reward + cmd_vel_reward + turn_follow_reward + fwd_bias
            - backward_pen - lateral_pen - hop_pen - grounded_pen - height_pen
            - tilt_pen - roll_rate_pen
            - torque_pen - smoothness - joint_vel_pen - hip_pen - stationary_pen
            - unwanted_turn_pen - diagonal_sync_pen)
