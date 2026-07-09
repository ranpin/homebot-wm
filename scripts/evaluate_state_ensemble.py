"""Closed-loop eval with an ENSEMBLE state-space world model (PETS-style).

Single-model CEM exploited the imperfect dynamics: rolling one model out and
optimizing hard found action sequences that looked good to the model but failed
in reality (more planning -> lower success). An ensemble fixes this by
penalizing disagreement: each candidate is rolled through all N members and
scored `mean(score) - lambda * std(score)`. Sequences where the members diverge
(i.e. the model is uncertain / being exploited) are down-weighted, so CEM sticks
to actions the whole ensemble trusts.

Usage:
    python scripts/evaluate_state_ensemble.py --ckpt_prefix checkpoints/state_dynamics_ \
        --num_models 5 --lambda_std 1.0 --episodes 40
"""

import argparse

import numpy as np
import torch

from wm_core.dynamics import build_dynamics
from wm_sim.env import HomeTabletopEnv


@torch.no_grad()
def plan_ensemble(models, start, score_fn, args, device):
    """CEM with ensemble-disagreement penalty. start: (1, dim)."""
    H, A = args.horizon, 2
    mean = torch.zeros(H, A, device=device)
    std = torch.ones(H, A, device=device) * 0.5
    N = len(models)
    for _ in range(args.num_iterations):
        noise = torch.randn(args.num_samples, H, A, device=device)
        actions = (mean[None] + std[None] * noise).clamp(-1.0, 1.0)
        # Roll every candidate through every ensemble member in parallel.
        states = [start.expand(args.num_samples, -1).contiguous() for _ in range(N)]
        for t in range(H):
            for i in range(N):
                states[i] = models[i].predict_next(states[i], actions[:, t])
        member_scores = torch.stack([score_fn(states[i]) for i in range(N)], dim=0)  # (N, S)
        scores = member_scores.mean(0) - args.lambda_std * member_scores.std(0)
        elite_idx = scores.topk(args.num_elites).indices
        elite = actions[elite_idx]
        mean = elite.mean(0)
        std = elite.std(0).clamp(min=0.01)
    return mean


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop eval, ensemble state-space world model")
    p.add_argument("--ckpt_prefix", type=str, default="checkpoints/state_dynamics_")
    p.add_argument("--num_models", type=int, default=5)
    p.add_argument("--lambda_std", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--replan_every", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--num_elites", type=int, default=20)
    p.add_argument("--num_iterations", type=int, default=4)
    p.add_argument("--record_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = []
    for i in range(args.num_models):
        ckpt = torch.load(f"{args.ckpt_prefix}{i}.pt", map_location=device)
        m = build_dynamics(ckpt["config"]).to(device)
        m.load_state_dict(ckpt["dynamics_state"])
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} ensemble members; lambda_std={args.lambda_std}")

    env = HomeTabletopEnv(max_steps=args.max_steps)
    target = torch.from_numpy(np.asarray(env.target_pos)).float().to(device)

    def score_fn(state):
        return -torch.norm(state[..., 4:6] - target, dim=-1)

    if args.record_dir:
        from pathlib import Path
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        action_seq = None
        frames = [obs["image"]] if args.record_dir else None
        for step in range(args.max_steps):
            if step % args.replan_every == 0:
                state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
                action_seq = plan_ensemble(models, state, score_fn, args, device)
            action = action_seq[step % len(action_seq)].cpu().numpy()
            obs, _, success, truncated, info = env.step(action)
            if args.record_dir:
                frames.append(obs["image"])
            if success or truncated:
                break
        if args.record_dir and success:
            from PIL import Image
            imgs = [Image.fromarray(f).resize((336, 336), Image.NEAREST) for f in frames]
            imgs[0].save(f"{args.record_dir}/ens_success_ep{ep+1:02d}.gif",
                         save_all=True, append_images=imgs[1:], duration=40, loop=0)
        results.append({"success": bool(success), "steps": step + 1, "dist": info["dist_to_target"]})
        print(f"  Episode {ep+1}/{args.episodes}: {'SUCCESS' if success else 'FAIL'} "
              f"(steps={step+1}, dist={info['dist_to_target']:.2f})")
    env.close()

    succ = [r for r in results if r["success"]]
    print(f"\n=== Ensemble state-space CEM (lambda_std={args.lambda_std}) ===")
    print(f"Success rate: {len(succ)}/{len(results)} ({100*len(succ)/len(results):.1f}%)")
    if succ:
        print(f"Avg steps (success): {np.mean([r['steps'] for r in succ]):.1f}")
    print(f"Avg final distance: {np.mean([r['dist'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
