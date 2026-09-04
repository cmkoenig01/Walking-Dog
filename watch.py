"""Watch the trained policies in the interactive MuJoCo viewer.

    python watch.py            # Linux / Windows
    mjpython watch.py          # macOS — the passive viewer needs mjpython

macOS requires `mjpython` (shipped with the mujoco wheel) because the viewer
has to own the main thread there. Running `python watch.py` on macOS raises a
RuntimeError from mujoco itself; this script checks for that up front and says
so, rather than letting it fail deep in the stack.

For numbers rather than pictures, use `evaluate.py` — it is headless and much
faster than real time.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import a1_env as env
from evaluate import load_policies

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Watch the trained A1 policies in the MuJoCo viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="reset the episode after this much simulated time")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--nav-ckpt", type=Path, default=None)
    p.add_argument("--walk-ckpt", type=Path, default=None)
    p.add_argument("--scene", type=Path, default=env.SCENE_PATH)
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (2.0 = twice real time)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if platform.system() == "Darwin" and Path(sys.executable).name != "mjpython":
        sys.exit(
            "On macOS the MuJoCo viewer must run under mjpython:\n"
            "    mjpython watch.py\n"
            "(Or use `python evaluate.py` for a headless benchmark.)"
        )

    import mujoco.viewer

    rng = np.random.default_rng(args.seed)
    scene = env.A1Scene.load(args.scene)
    data = scene.make_data()
    nav_ppo, walk_ppo = load_policies(args)

    dt = scene.timestep
    max_steps = min(int(args.max_seconds / dt), env.MAX_EP_LEN)

    goal = env.random_goal(rng)
    env.set_goal(scene, data, goal)
    mujoco.mj_forward(scene.model, data)

    episode = 1
    held = 0
    stand_count = 0
    step = 0

    def new_episode(reason):
        nonlocal goal, episode, stand_count, step
        print(f"  episode {episode}: {reason}", flush=True)
        episode += 1
        stand_count = 0
        step = 0
        goal = env.random_goal(rng)
        scene.reset(data)
        env.set_goal(scene, data, goal)
        mujoco.mj_forward(scene.model, data)

    print(f"watching — {env.STAND_STEPS} steps inside {env.GOAL_RADIUS} m counts as a hold. "
          f"Close the window to quit.", flush=True)

    with mujoco.viewer.launch_passive(scene.model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()

            nav_cmd = nav_ppo.act(env.get_nav_state(data, goal))
            action = walk_ppo.act(env.get_walk_state(data, nav_cmd))
            env.apply_action(scene, data, action)
            mujoco.mj_step(scene.model, data)
            viewer.sync()
            step += 1

            dist = float(np.linalg.norm(goal - data.xpos[env.TRUNK_BODY_ID][:2]))
            if dist < env.GOAL_RADIUS:
                stand_count += 1
            elif not (stand_count > 0 and dist < env.GOAL_RADIUS * 1.5):
                stand_count = 0

            if step % 250 == 0:  # twice a second of simulated time
                print(f"    ep {episode:>3} | t={step * dt:5.1f}s | dist={dist:5.2f}m "
                      f"| hold={stand_count:>3}/{env.STAND_STEPS}", flush=True)

            if stand_count >= env.STAND_STEPS:
                held += 1
                new_episode(f"reached and held the goal ({held} so far)")
            elif env.is_fallen(data):
                new_episode("fell")
            elif step >= max_steps:
                new_episode("timed out")

            # Pace to wall clock so the motion is watchable.
            remaining = dt / max(args.speed, 1e-6) - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

    print(f"\nheld the goal in {held} of {episode - 1} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
