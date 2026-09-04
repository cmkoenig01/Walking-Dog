"""Headless benchmark: run the trained policies over many episodes and report.

This is what produces the numbers in the README. No viewer, no real-time
pacing, deterministic actions (the distribution mean rather than a sample), and
a fixed seed, so the result is reproducible:

    python evaluate.py --episodes 100 --seed 0

Two success criteria are reported, because they answer different questions:

  reached  — the trunk came within GOAL_RADIUS of the goal at any point.
  held     — the trunk then stayed there for STAND_STEPS (0.2 s), which is the
             criterion `train.py` actually rewards.

`held` is the honest headline number; `reached` shows how much of the gap is
"navigated there but drifted off" versus "never arrived".
"""

import argparse
import statistics
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

import a1_env as env
from ppo import PPO

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark the trained A1 policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=20.0,
                   help="simulated-time cap per episode")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--nav-ckpt", type=Path, default=None)
    p.add_argument("--walk-ckpt", type=Path, default=None)
    p.add_argument("--scene", type=Path, default=env.SCENE_PATH)
    p.add_argument("--quiet", action="store_true", help="suppress per-episode progress")
    return p.parse_args(argv)


def load_policies(args):
    nav_path = args.nav_ckpt or args.checkpoint_dir / "nav_best.pt"
    walk_path = args.walk_ckpt or args.checkpoint_dir / "walk_best.pt"
    for path in (nav_path, walk_path):
        if not path.exists():
            raise SystemExit(
                f"checkpoint not found: {path}\n"
                f"Train one with `python train.py`, or pass --nav-ckpt/--walk-ckpt."
            )
    nav = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM, 128, 256)
    walk = PPO(env.WALK_STATE_DIM, env.WALK_ACTION_DIM, 256, 512)
    nav.load(nav_path, load_optimizer=False)
    walk.load(walk_path, load_optimizer=False)
    return nav.eval_mode(), walk.eval_mode()


def run_episode(scene, data, nav_ppo, walk_ppo, goal, max_steps):
    """Roll out one episode. Returns a dict describing how it ended."""
    scene.reset(data)
    env.set_goal(scene, data, goal)
    mujoco.mj_forward(scene.model, data)

    stand_count = 0
    reached_step = None
    path_len = 0.0
    path_len_at_arrival = None
    prev_xy = data.xpos[env.TRUNK_BODY_ID][:2].copy()

    for step in range(max_steps):
        nav_cmd = nav_ppo.act(env.get_nav_state(data, goal))
        action = walk_ppo.act(env.get_walk_state(data, nav_cmd))
        env.apply_action(scene, data, action)
        mujoco.mj_step(scene.model, data)

        xy = data.xpos[env.TRUNK_BODY_ID][:2]
        path_len += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy.copy()

        dist = float(np.linalg.norm(goal - xy))
        if dist < env.GOAL_RADIUS:
            if reached_step is None:
                reached_step = step
                path_len_at_arrival = path_len
            stand_count += 1
        elif not (stand_count > 0 and dist < env.GOAL_RADIUS * 1.5):
            stand_count = 0

        if stand_count >= env.STAND_STEPS:
            return dict(outcome="held", steps=step, reached_step=reached_step,
                        final_dist=dist, path_len=path_len_at_arrival)
        if env.is_fallen(data):
            return dict(outcome="fell", steps=step, reached_step=reached_step,
                        final_dist=dist, path_len=path_len_at_arrival)

    return dict(outcome="timeout", steps=max_steps, reached_step=reached_step,
                final_dist=float(np.linalg.norm(goal - data.xpos[env.TRUNK_BODY_ID][:2])),
                path_len=path_len_at_arrival)


def summarise(results, dt, episodes):
    counts = Counter(r["outcome"] for r in results)
    held = counts["held"]
    reached = sum(1 for r in results if r["reached_step"] is not None)

    def pct(k):
        return 100.0 * k / episodes

    lines = [
        f"  held goal ({env.STAND_STEPS} steps)   {held:>4}  {pct(held):5.1f}%",
        f"  reached goal (any contact)  {reached:>4}  {pct(reached):5.1f}%",
        f"  fell                        {counts['fell']:>4}  {pct(counts['fell']):5.1f}%",
        f"  timed out                   {counts['timeout']:>4}  {pct(counts['timeout']):5.1f}%",
    ]

    times = [r["reached_step"] * dt for r in results if r["reached_step"] is not None]
    if times:
        lines.append(
            f"\n  time to reach goal (n={len(times)}): "
            f"median {statistics.median(times):.1f}s, mean {statistics.mean(times):.1f}s, "
            f"min {min(times):.1f}s, max {max(times):.1f}s"
        )
        # Shortest possible path: straight from the origin to the goal circle,
        # stopping as soon as the trunk crosses into GOAL_RADIUS.
        shortest = env.CIRCLE_RADIUS - env.GOAL_RADIUS
        eff = [shortest / r["path_len"] for r in results
               if r["path_len"]]
        if eff:
            lines.append(
                f"  path efficiency (shortest {shortest:.1f} m / travelled): "
                f"median {statistics.median(eff):.2f}"
            )
    failures = [r["final_dist"] for r in results if r["outcome"] != "held"]
    if failures:
        lines.append(
            f"  distance to goal when it failed (n={len(failures)}): "
            f"median {statistics.median(failures):.2f} m"
        )
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scene = env.A1Scene.load(args.scene)
    data = scene.make_data()
    nav_ppo, walk_ppo = load_policies(args)

    dt = scene.timestep
    max_steps = min(int(args.max_seconds / dt), env.MAX_EP_LEN)

    print(f"evaluating {args.episodes} episodes "
          f"(seed {args.seed}, cap {args.max_seconds:g}s = {max_steps} steps, deterministic)",
          flush=True)

    results = []
    t0 = time.time()
    for i in range(args.episodes):
        results.append(
            run_episode(scene, data, nav_ppo, walk_ppo, env.random_goal(rng), max_steps)
        )
        if not args.quiet and (i + 1) % 10 == 0:
            done = Counter(r["outcome"] for r in results)
            print(f"  {i + 1}/{args.episodes} — held {done['held']}, "
                  f"fell {done['fell']}, timeout {done['timeout']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== {args.episodes} episodes, seed {args.seed} ===")
    print(summarise(results, dt, args.episodes))
    print(f"\n  wall clock: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
