"""Tests for the environment contract and the PPO plumbing.

The observation layout is a contract with the checkpoints in `checkpoints/`:
if a field moves, every saved policy silently becomes garbage without raising.
These tests pin the layout so that change fails loudly instead.

    pip install pytest && pytest -q
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import a1_env as env
from ppo import PPO, RolloutBuffer


@pytest.fixture(scope="module")
def scene():
    return env.A1Scene.load()


@pytest.fixture
def data(scene):
    return scene.make_data()


# -- the frozen observation/action contract --------------------------------

def test_scene_matches_expected_model_layout(scene):
    """7 free-joint qpos + 12 joints, 6 free-joint qvel + 12 joints, 12 motors."""
    assert scene.model.nq == 19
    assert scene.model.nv == 18
    assert scene.model.nu == env.WALK_ACTION_DIM
    assert len(scene.joint_inds) == 12
    assert len(scene.ctrl_inds) == 12
    assert len(scene.foot_body_ids) == 4


def test_nav_state_dimension(scene, data):
    state = env.get_nav_state(data, np.array([1.0, 2.0]))
    assert state.shape == (env.NAV_STATE_DIM,)
    assert np.all(np.isfinite(state))


def test_walk_state_dimension(scene, data):
    state = env.get_walk_state(data, np.array([0.5, -0.5]))
    assert state.shape == (env.WALK_STATE_DIM,)
    assert np.all(np.isfinite(state))


def test_walk_state_ends_with_the_nav_command(scene, data):
    """The nav command is the last two entries — the hierarchy's only channel."""
    cmd = np.array([0.25, -0.75])
    assert env.get_walk_state(data, cmd)[-2:] == pytest.approx(cmd)


def test_walk_state_carries_all_twelve_joint_positions(scene, data):
    data.qpos[7:19] = np.linspace(-1, 1, 12)
    mujoco.mj_forward(scene.model, data)
    assert env.get_walk_state(data, np.zeros(2))[7:19] == pytest.approx(
        np.linspace(-1, 1, 12)
    )


# -- termination -----------------------------------------------------------

def test_home_keyframe_is_not_fallen(scene, data):
    mujoco.mj_forward(scene.model, data)
    assert not env.is_fallen(data)


def test_is_fallen_when_trunk_is_low(scene, data):
    data.qpos[2] = 0.10
    mujoco.mj_forward(scene.model, data)
    assert env.is_fallen(data)


def test_is_fallen_when_tipped_over(scene, data):
    data.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # 180 deg roll
    mujoco.mj_forward(scene.model, data)
    assert env.is_fallen(data)


# -- goals -----------------------------------------------------------------

def test_random_goal_lies_on_the_circle():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert np.linalg.norm(env.random_goal(rng)) == pytest.approx(env.CIRCLE_RADIUS)


def test_random_goal_is_reproducible_for_a_seed():
    a = env.random_goal(np.random.default_rng(42))
    b = env.random_goal(np.random.default_rng(42))
    assert a == pytest.approx(b)


def test_set_goal_moves_the_marker(scene, data):
    env.set_goal(scene, data, np.array([1.5, -2.5]))
    assert data.mocap_pos[scene.goal_mocap_id][:2] == pytest.approx([1.5, -2.5])


# -- rewards ---------------------------------------------------------------

def test_nav_reward_pays_the_arrival_bonus_at_the_goal(scene, data):
    mujoco.mj_forward(scene.model, data)
    goal = data.xpos[env.TRUNK_BODY_ID][:2].copy()
    assert env.compute_nav_reward(data, goal, at_goal=True) >= 50.0


def test_rewards_are_finite_over_random_states(scene, data):
    rng = np.random.default_rng(1)
    for _ in range(100):
        data.qpos[:] = rng.normal(0, 0.3, scene.model.nq)
        data.qpos[2] = abs(data.qpos[2]) + 0.2
        quat = rng.normal(0, 1, 4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qvel[:] = rng.normal(0, 0.5, scene.model.nv)
        mujoco.mj_forward(scene.model, data)

        goal = env.random_goal(rng)
        action = rng.normal(0, 0.5, env.WALK_ACTION_DIM)
        cmd = rng.normal(0, 0.5, env.NAV_ACTION_DIM)
        assert np.isfinite(env.compute_nav_reward(data, goal, False))
        assert np.isfinite(
            env.compute_walk_reward(scene, data, action, action, False, cmd)
        )


def test_foot_contacts_shape(scene, data):
    mujoco.mj_forward(scene.model, data)
    contacts = env.get_foot_contacts(scene, data)
    assert contacts.shape == (4,)
    assert contacts.dtype == bool


def test_standing_on_the_ground_puts_feet_in_contact(scene, data):
    for _ in range(200):  # let it settle onto the floor
        env.apply_action(scene, data, np.zeros(env.WALK_ACTION_DIM))
        mujoco.mj_step(scene.model, data)
    assert env.get_foot_contacts(scene, data).sum() >= 3
    assert not env.is_fallen(data)


# -- actions ---------------------------------------------------------------

def test_zero_action_commands_the_home_pose(scene, data):
    env.apply_action(scene, data, np.zeros(env.WALK_ACTION_DIM))
    assert data.ctrl[scene.ctrl_inds] == pytest.approx(env.HOME_CTRL)


def test_action_scales_as_an_offset_from_home(scene, data):
    action = np.ones(env.WALK_ACTION_DIM)
    env.apply_action(scene, data, action)
    assert data.ctrl[scene.ctrl_inds] == pytest.approx(
        env.HOME_CTRL + env.ACTION_SCALE
    )


# -- PPO -------------------------------------------------------------------

def test_agent_action_shapes():
    agent = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    state = np.zeros(env.NAV_STATE_DIM)
    action, log_prob, value = agent.get_action(state)
    assert action.shape == (env.NAV_ACTION_DIM,)
    assert isinstance(log_prob, float) and isinstance(value, float)
    assert agent.act(state).shape == (env.NAV_ACTION_DIM,)


def test_deterministic_action_is_repeatable():
    agent = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    state = np.ones(env.NAV_STATE_DIM) * 0.3
    a, _, _ = agent.get_action(state, deterministic=True)
    b, _, _ = agent.get_action(state, deterministic=True)
    assert a == pytest.approx(b)
    assert a == pytest.approx(agent.act(state))


def test_batch_actions_match_the_batch_size():
    agent = PPO(env.WALK_STATE_DIM, env.WALK_ACTION_DIM, 256, 512)
    states = np.zeros((7, env.WALK_STATE_DIM))
    actions, log_probs, values = agent.get_actions_batch(states)
    assert actions.shape == (7, env.WALK_ACTION_DIM)
    assert log_probs.shape == (7,)
    assert values.shape == (7,)


def test_gae_reduces_to_discounted_reward_without_bootstrapping():
    """With lambda=1, gamma=1 and zero baselines, GAE is the reward-to-go."""
    agent = PPO(4, 2, 16, 16)
    agent.cfg.gamma = 1.0
    agent.cfg.lam = 1.0
    buf = RolloutBuffer()
    for r in (1.0, 2.0, 3.0):
        buf.add(np.zeros(4), np.zeros(2), r, 0.0, 0.0, False)
    assert agent._gae(buf, next_value=0.0) == pytest.approx([6.0, 5.0, 3.0])


def test_gae_stops_at_an_episode_boundary():
    agent = PPO(4, 2, 16, 16)
    agent.cfg.gamma = 1.0
    agent.cfg.lam = 1.0
    buf = RolloutBuffer()
    buf.add(np.zeros(4), np.zeros(2), 1.0, 0.0, 0.0, True)   # terminal
    buf.add(np.zeros(4), np.zeros(2), 5.0, 0.0, 0.0, False)
    assert agent._gae(buf, next_value=0.0) == pytest.approx([1.0, 5.0])


def test_update_runs_and_changes_the_policy():
    torch.manual_seed(0)
    agent = PPO(4, 2, 16, 16)
    agent.cfg.batch_size = 8
    agent.cfg.epochs = 1
    buffers = []
    rng = np.random.default_rng(0)
    for _ in range(2):
        buf = RolloutBuffer()
        for _ in range(16):
            buf.add(rng.normal(size=4), rng.normal(size=2),
                    float(rng.normal()), -1.0, 0.0, False)
        buffers.append(buf)

    before = agent.actor.net[0].weight.detach().clone()
    agent.update(buffers, next_values=[0.0, 0.0])
    assert not torch.allclose(before, agent.actor.net[0].weight)
    assert len(buffers[0]) == 0, "buffers should be cleared after an update"


def test_checkpoint_round_trip(tmp_path):
    agent = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    state = np.ones(env.NAV_STATE_DIM) * 0.2
    expected = agent.act(state)

    path = tmp_path / "ckpt.pt"
    agent.save(path, best_reward=1.25)

    restored = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    assert restored.load(path) == pytest.approx(1.25)
    assert restored.act(state) == pytest.approx(expected)

    # Must stay loadable under torch's safe loader.
    assert isinstance(torch.load(path, weights_only=True)["best_reward"], float)


# -- the shipped checkpoints ----------------------------------------------

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.mark.skipif(not (CHECKPOINTS / "nav_best.pt").exists(),
                    reason="shipped checkpoints not present")
def test_shipped_checkpoints_load_into_the_current_architecture():
    nav = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    walk = PPO(env.WALK_STATE_DIM, env.WALK_ACTION_DIM, 256, 512)
    assert np.isfinite(nav.load(CHECKPOINTS / "nav_best.pt", load_optimizer=False))
    assert np.isfinite(walk.load(CHECKPOINTS / "walk_best.pt", load_optimizer=False))
    assert nav.act(np.zeros(env.NAV_STATE_DIM)).shape == (env.NAV_ACTION_DIM,)
    assert walk.act(np.zeros(env.WALK_STATE_DIM)).shape == (env.WALK_ACTION_DIM,)
