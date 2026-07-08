"""Fast success-rate check for the scripted expert (no rendering).

Runs the expert in the env with image rendering stubbed out, so hundreds of
episodes take seconds instead of minutes. Use this to tune the expert before
committing to a full (rendered) data collection.

Usage:
    python scripts/eval_expert.py --episodes 300
"""

import argparse

import numpy as np

from wm_sim.env import HomeTabletopEnv
from wm_sim.expert import ScriptedExpert


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = HomeTabletopEnv(max_steps=args.max_steps)
    # Stub rendering — we only need states/success here.
    zero_img = np.zeros((env.image_size, env.image_size, 3), dtype=np.uint8)
    env._get_image = lambda: zero_img

    expert = ScriptedExpert(env.target_pos)

    succ, steps_on_success, final_dists = 0, [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for step in range(args.max_steps):
            action = expert(obs["state"])
            obs, _, success, truncated, info = env.step(action)
            if success or truncated:
                break
        if success:
            succ += 1
            steps_on_success.append(step + 1)
        final_dists.append(info["dist_to_target"])
    env.close()

    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {succ}/{args.episodes} ({100*succ/args.episodes:.1f}%)")
    if steps_on_success:
        print(f"Avg steps (success): {np.mean(steps_on_success):.1f}")
    print(f"Avg final distance: {np.mean(final_dists):.2f}")


if __name__ == "__main__":
    main()
