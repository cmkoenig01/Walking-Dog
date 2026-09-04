"""Render an episode to an animated GIF, for the README.

Offscreen rendering, so it needs no display and no viewer:

    python record.py --out media/walk.gif --seed 3

Needs Pillow, which is not required to train or evaluate:

    pip install pillow

By default it retries seeds until it captures an episode that actually reaches
the goal, so the recording shows the behaviour rather than a lucky or unlucky
draw. Pass --any-outcome to record the first episode regardless.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import a1_env as env
from evaluate import load_policies

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Record an episode to a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=ROOT / "media" / "walk.gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tries", type=int, default=25,
                   help="seeds to try before giving up on a successful episode")
    p.add_argument("--any-outcome", action="store_true",
                   help="record the first episode whatever happens")
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--colors", type=int, default=96,
                   help="GIF palette size; lower is smaller on disk")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--nav-ckpt", type=Path, default=None)
    p.add_argument("--walk-ckpt", type=Path, default=None)
    p.add_argument("--scene", type=Path, default=env.SCENE_PATH)
    return p.parse_args(argv)


def record_episode(scene, data, renderer, camera, nav_ppo, walk_ppo,
                   goal, max_steps, frame_every):
    """Roll out one episode, capturing frames. Returns (frames, outcome)."""
    scene.reset(data)
    env.set_goal(scene, data, goal)
    mujoco.mj_forward(scene.model, data)

    frames = []
    stand_count = 0
    outcome = "timeout"

    for step in range(max_steps):
        nav_cmd = nav_ppo.act(env.get_nav_state(data, goal))
        action = walk_ppo.act(env.get_walk_state(data, nav_cmd))
        env.apply_action(scene, data, action)
        mujoco.mj_step(scene.model, data)

        if step % frame_every == 0:
            # Frame the robot AND the goal: look at the midpoint between them
            # and pull back as they separate, so a viewer can always see both
            # where the robot is and where it is trying to get to.
            trunk = data.xpos[env.TRUNK_BODY_ID]
            midpoint = np.array([
                (trunk[0] + goal[0]) / 2, (trunk[1] + goal[1]) / 2, 0.25
            ])
            separation = float(np.linalg.norm(goal - trunk[:2]))
            camera.lookat[:] = midpoint
            camera.distance = float(np.clip(separation * 1.15 + 1.4, 2.0, 5.5))
            renderer.update_scene(data, camera)
            frames.append(renderer.render().copy())

        dist = float(np.linalg.norm(goal - data.xpos[env.TRUNK_BODY_ID][:2]))
        if dist < env.GOAL_RADIUS:
            stand_count += 1
        elif not (stand_count > 0 and dist < env.GOAL_RADIUS * 1.5):
            stand_count = 0

        if stand_count >= env.STAND_STEPS:
            outcome = "held"
            break
        if env.is_fallen(data):
            outcome = "fell"
            break

    return frames, outcome


def main(argv=None):
    args = parse_args(argv)
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("record.py needs Pillow:  pip install pillow")

    scene = env.A1Scene.load(args.scene)
    data = scene.make_data()
    nav_ppo, walk_ppo = load_policies(args)

    renderer = mujoco.Renderer(scene.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 3.2
    camera.elevation = -20
    camera.azimuth = 130

    dt = scene.timestep
    max_steps = min(int(args.max_seconds / dt), env.MAX_EP_LEN)
    frame_every = max(1, round(1.0 / (args.fps * dt)))

    frames = outcome = None
    for attempt in range(args.tries):
        seed = args.seed + attempt
        goal = env.random_goal(np.random.default_rng(seed))
        frames, outcome = record_episode(
            scene, data, renderer, camera, nav_ppo, walk_ppo,
            goal, max_steps, frame_every,
        )
        print(f"seed {seed}: {outcome} ({len(frames)} frames)", flush=True)
        if args.any_outcome or outcome == "held":
            break
    else:
        print(f"no successful episode in {args.tries} tries — writing the last one")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Quantise every frame against one shared adaptive palette. Per-frame
    # palettes would shimmer, and full colour would be ~3x the file size.
    images = [Image.fromarray(f) for f in frames]
    palette = images[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
              for im in images]

    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} — {len(images)} frames, {size_mb:.1f} MB, outcome: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
