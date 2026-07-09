"""Closed-loop evaluation of a STATE-SPACE world model + CEM planner.

The observation state `[agent_xy, agent_vxy, block_xy]` is used directly as the
planning state (no encoder, no decoder). CEM rolls the learned state dynamics
and scores candidate action sequences by the predicted block-to-target distance.

Usage:
    python scripts/evaluate_state.py --checkpoint checkpoints/state_dynamics.pt \
        --episodes 20
"""

import argparse

import numpy as np
import torch

from wm_core.dynamics import build_dynamics
from wm_core.planner.cem_planner import CEMPlanner
from wm_sim.env import HomeTabletopEnv


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop eval, state-space world model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--replan_every", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_elites", type=int, default=20)
    p.add_argument("--num_iterations", type=int, default=4)
    p.add_argument("--running", type=int, default=0, help="1=accumulated running cost, 0=terminal only")
    p.add_argument("--approach_weight", type=float, default=0.0,
                   help="score shaping: weight on agent-behind-block term (0 disables)")
    p.add_argument("--record_dir", type=str, default=None,
                   help="if set, save a GIF of each successful episode here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    dynamics = build_dynamics(config).to(device)
    dynamics.load_state_dict(ckpt["dynamics_state"])
    dynamics.eval()

    env = HomeTabletopEnv(max_steps=args.max_steps)
    target = torch.from_numpy(np.asarray(env.target_pos)).float().to(device)

    def score_fn(pred_state: torch.Tensor) -> torch.Tensor:
        agent = pred_state[..., 0:2]
        block = pred_state[..., 4:6]
        block_to_target = torch.norm(block - target, dim=-1)
        if args.approach_weight <= 0:
            return -block_to_target
        # Encourage the agent to get *behind* the block relative to the target
        # (the push point), so CEM has signal even before contact.
        direction = target - block
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)
        push_point = block - direction * 0.4
        agent_to_push = torch.norm(agent - push_point, dim=-1)
        return -block_to_target - args.approach_weight * agent_to_push

    planner = CEMPlanner(dynamics, action_dim=2, horizon=args.horizon,
                         num_samples=args.num_samples, num_elites=args.num_elites,
                         num_iterations=args.num_iterations)

    record = args.record_dir is not None
    if record:
        from pathlib import Path
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        action_seq = None
        frames = [obs["image"]] if record else None
        for step in range(args.max_steps):
            if step % args.replan_every == 0:
                state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action_seq = planner.plan(state, score_fn, running=bool(args.running))
            action = action_seq[step % len(action_seq)].cpu().numpy()
            obs, _, success, truncated, info = env.step(action)
            if record:
                frames.append(obs["image"])
            if success or truncated:
                break
        if record and success:
            from PIL import Image
            imgs = [Image.fromarray(fr).resize((336, 336), Image.NEAREST) for fr in frames]
            gif_path = f"{args.record_dir}/success_ep{ep+1:02d}.gif"
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
            print(f"    saved demo -> {gif_path}")
        results.append({"success": bool(success), "steps": step + 1, "dist": info["dist_to_target"]})
        print(f"  Episode {ep+1}/{args.episodes}: {'SUCCESS' if success else 'FAIL'} "
              f"(steps={step+1}, dist={info['dist_to_target']:.2f})")
    env.close()

    succ = [r for r in results if r["success"]]
    print(f"\n=== State-space CEM results ===")
    print(f"Success rate: {len(succ)}/{len(results)} ({100*len(succ)/len(results):.1f}%)")
    if succ:
        print(f"Avg steps (success): {np.mean([r['steps'] for r in succ]):.1f}")
    print(f"Avg final distance: {np.mean([r['dist'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
