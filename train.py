"""Train the navigation and locomotion policies jointly.

Both policies learn at the same time from the same stream of experience. Each
step, the nav policy reads the goal geometry and emits a 2D command; the walk
policy reads the body state plus that command and emits 12 joint offsets. Only
the walk policy touches the actuators, so the nav policy is graded on where the
robot ends up, and the walk policy on how well it followed the command.

Usage
-----
    python train.py --steps 5_000_000              # fresh run
    python train.py --init best --steps 1_000_000  # continue from the shipped policies

Ctrl-C saves and exits cleanly.
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

import a1_env as env
from ppo import PPO, RolloutBuffer

ROOT = Path(__file__).resolve().parent

DEFAULT_STEPS_PER_UPDATE = 2048
DEFAULT_N_ENVS = 16


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the A1 walk-to-goal policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=5_000_000,
                   help="total environment steps to collect")
    p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS,
                   help="parallel MuJoCo environments")
    p.add_argument("--steps-per-update", type=int, default=DEFAULT_STEPS_PER_UPDATE,
                   help="transitions per environment between PPO updates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", choices=("scratch", "best", "latest"), default="scratch",
                   help="scratch: random init. best: the shipped checkpoints. "
                        "latest: the rolling checkpoints from --run-dir")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "latest",
                   help="where this run's checkpoints and training log are written")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints",
                   help="where `--init best` reads the shipped policies from. "
                        "Training never writes here: a run that beats its own "
                        "best writes runs/<run>/nav_best.pt, and promoting it "
                        "is a deliberate copy into checkpoints/.")
    p.add_argument("--scene", type=Path, default=env.SCENE_PATH)
    return p.parse_args(argv)


def build_agents(args):
    nav = PPO(env.NAV_STATE_DIM, env.NAV_ACTION_DIM,
              hidden_dim=128, critic_hidden_dim=256)
    walk = PPO(env.WALK_STATE_DIM, env.WALK_ACTION_DIM,
               hidden_dim=256, critic_hidden_dim=512)

    nav_best = walk_best = float("-inf")
    if args.init != "scratch":
        if args.init == "best":
            nav_path = args.checkpoint_dir / "nav_best.pt"
            walk_path = args.checkpoint_dir / "walk_best.pt"
        else:
            nav_path = args.run_dir / "nav_latest.pt"
            walk_path = args.run_dir / "walk_latest.pt"
        for path in (nav_path, walk_path):
            if not path.exists():
                sys.exit(f"--init {args.init} but {path} does not exist")
        nav_best = nav.load(nav_path)
        walk_best = walk.load(walk_path)
        print(f"resumed from {nav_path.name} / {walk_path.name} "
              f"(best so far: nav {nav_best:.3f}, walk {walk_best:.3f})")
    return nav, walk, nav_best, walk_best


def main(argv=None):
    args = parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.run_dir.mkdir(parents=True, exist_ok=True)

    # One model shared by every environment; only the state (MjData) differs.
    scene = env.A1Scene.load(args.scene)
    datas = [scene.make_data() for _ in range(args.n_envs)]

    nav_ppo, walk_ppo, nav_best, walk_best = build_agents(args)

    n = args.n_envs
    nav_buffers  = [RolloutBuffer() for _ in range(n)]
    walk_buffers = [RolloutBuffer() for _ in range(n)]
    ep_steps     = [0] * n
    prev_actions = [np.zeros(env.WALK_ACTION_DIM) for _ in range(n)]
    goals        = [env.random_goal(rng) for _ in range(n)]
    stand_counts = [0] * n
    for i in range(n):
        env.set_goal(scene, datas[i], goals[i])

    log_path = args.run_dir / "training_log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(
            ["step", "nav_reward", "walk_reward", "episodes", "falls", "goals", "wall_s"]
        )

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        if stopping:                      # second Ctrl-C: give up immediately
            sys.exit(130)
        stopping = True
        print("\nstopping after this update — Ctrl-C again to abort now", flush=True)

    signal.signal(signal.SIGINT, request_stop)

    step_count = 0
    nav_accum = walk_accum = 0.0
    episodes = falls = goals_reached = 0
    t0 = time.time()

    print(f"training {args.steps:,} steps across {n} envs (seed {args.seed})",
          flush=True)

    while step_count < args.steps and not stopping:
        # 1. nav policy proposes a [forward, turn] command from goal geometry
        nav_states = np.array([env.get_nav_state(datas[i], goals[i]) for i in range(n)])
        nav_cmds, nav_lp, nav_vals = nav_ppo.get_actions_batch(nav_states)

        # 2. walk policy turns body state + that command into joint offsets
        walk_states = np.array([env.get_walk_state(datas[i], nav_cmds[i]) for i in range(n)])
        walk_acts, walk_lp, walk_vals = walk_ppo.get_actions_batch(walk_states)

        # 3. step physics
        for i in range(n):
            env.apply_action(scene, datas[i], walk_acts[i])
            mujoco.mj_step(scene.model, datas[i])

        # 4. reward, store, reset finished episodes
        for i in range(n):
            d = datas[i]
            ep_steps[i] += 1

            dist = np.linalg.norm(goals[i] - d.xpos[env.TRUNK_BODY_ID][:2])
            if dist < env.GOAL_RADIUS:
                stand_counts[i] += 1
            elif not (stand_counts[i] > 0 and dist < env.GOAL_RADIUS * 1.5):
                stand_counts[i] = 0
            at_goal = stand_counts[i] > 0

            nav_r = env.compute_nav_reward(d, goals[i], at_goal)
            walk_r = env.compute_walk_reward(
                scene, d, walk_acts[i], prev_actions[i], at_goal, nav_cmds[i]
            )
            prev_actions[i] = walk_acts[i].copy()
            nav_accum += nav_r
            walk_accum += walk_r

            fell = env.is_fallen(d)
            stood = stand_counts[i] >= env.STAND_STEPS
            terminated = fell or stood or ep_steps[i] >= env.MAX_EP_LEN

            nav_buffers[i].add(nav_states[i], nav_cmds[i], nav_r, nav_lp[i], nav_vals[i], terminated)
            walk_buffers[i].add(walk_states[i], walk_acts[i], walk_r, walk_lp[i], walk_vals[i], terminated)

            if terminated:
                episodes += 1
                falls += int(fell)
                goals_reached += int(stood)
                scene.reset(d)
                prev_actions[i] = np.zeros(env.WALK_ACTION_DIM)
                ep_steps[i] = 0
                stand_counts[i] = 0
                goals[i] = env.random_goal(rng)
                env.set_goal(scene, d, goals[i])

        step_count += n

        # 5. update both policies once the buffers are full
        if len(nav_buffers[0]) >= args.steps_per_update:
            next_nav_states = np.array([env.get_nav_state(datas[i], goals[i]) for i in range(n)])
            next_nav_cmds, _, _ = nav_ppo.get_actions_batch(next_nav_states)
            next_walk_states = np.array(
                [env.get_walk_state(datas[i], next_nav_cmds[i]) for i in range(n)]
            )

            nav_ppo.update(nav_buffers, nav_ppo.value_of(next_nav_states))
            walk_ppo.update(walk_buffers, walk_ppo.value_of(next_walk_states))

            denom = args.steps_per_update * n
            nav_avg, walk_avg = nav_accum / denom, walk_accum / denom
            nav_accum = walk_accum = 0.0

            nav_ppo.save(args.run_dir / "nav_latest.pt")
            walk_ppo.save(args.run_dir / "walk_latest.pt")

            tags = ["", ""]
            if nav_avg > nav_best:
                nav_best = nav_avg
                tags[0] = " *"
                nav_ppo.save(args.run_dir / "nav_best.pt", best_reward=nav_avg)
            if walk_avg > walk_best:
                walk_best = walk_avg
                tags[1] = " *"
                walk_ppo.save(args.run_dir / "walk_best.pt", best_reward=walk_avg)

            elapsed = time.time() - t0
            log_writer.writerow([
                step_count, f"{nav_avg:.4f}", f"{walk_avg:.4f}",
                episodes, falls, goals_reached, f"{elapsed:.1f}",
            ])
            log_file.flush()

            print(
                f"step {step_count:>9,} | nav {nav_avg:7.3f}{tags[0]:<2} | "
                f"walk {walk_avg:7.3f}{tags[1]:<2} | eps {episodes:>5} "
                f"(goals {goals_reached}, falls {falls}) | "
                f"{step_count / max(elapsed, 1e-9):,.0f} steps/s",
                flush=True,
            )

    log_file.close()
    print(f"\ndone: {step_count:,} steps in {time.time() - t0:.0f}s")
    print(f"  checkpoints: {args.run_dir}  (promote with: cp {args.run_dir}/nav_best.pt checkpoints/)")
    print(f"  training log:        {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
