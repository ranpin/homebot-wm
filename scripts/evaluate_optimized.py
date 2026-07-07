"""Closed-loop evaluation with block position decoder.

Uses decoded block position to compute actual distance to target,
enabling effective CEM planning.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wm_core.dynamics import build_dynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_core.planner.cem_planner import CEMPlanner
from wm_sim.env import HomeTabletopEnv


class BlockPositionDecoder(nn.Module):
    """Decode block position from latent representation."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained encoder and dynamics from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    encoder = ResNetEncoder(
        adapter_dim=config["adapter_dim"],
        output_dim=config["latent_dim"],
    ).to(device)

    dynamics = build_dynamics(config).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])
    dynamics.load_state_dict(ckpt["dynamics_state"])

    encoder.eval()
    dynamics.eval()

    return encoder, dynamics, config


def load_block_decoder(path: str, latent_dim: int, device: torch.device) -> BlockPositionDecoder:
    """Load trained block decoder."""
    decoder = BlockPositionDecoder(latent_dim=latent_dim).to(device)
    decoder.load_state_dict(torch.load(path, map_location=device))
    decoder.eval()
    return decoder


def evaluate_episode(
    env: HomeTabletopEnv,
    encoder: ResNetEncoder,
    dynamics: DiffusionDynamics,
    block_decoder: BlockPositionDecoder,
    planner: CEMPlanner,
    device: torch.device,
    target_pos: np.ndarray,
    max_steps: int = 200,
    diffusion_steps: int = 10,  # Reduced for faster planning
    render: bool = False,
) -> dict:
    """Run one evaluation episode with closed-loop planning."""
    obs, info = env.reset()

    frames = []
    plan_idx = 0

    for step in range(max_steps):
        # Match training normalization (TrajectoryDataset divides images by 255).
        image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            latent = encoder(image)

            # Score: negative distance to target (higher = better)
            def score_predicted_position(pred_latent: torch.Tensor) -> torch.Tensor:
                pred_block_pos = block_decoder(pred_latent)
                target_t = torch.from_numpy(target_pos).float().to(device)
                dist = torch.norm(pred_block_pos - target_t, dim=-1)
                return -dist

            # Plan every N steps to reduce overhead
            if step % 3 == 0 or step == 0:
                if step % 20 == 0:
                    print(f"    Step {step}: planning (plan_idx={plan_idx})...", flush=True)

                # Use reduced diffusion steps for faster planning
                action_seq = planner.plan(latent, score_predicted_position, num_steps=diffusion_steps)
                plan_idx += 1

            # Execute action from sequence
            action_idx = min(step % len(action_seq), len(action_seq) - 1)
            action = action_seq[action_idx].cpu().numpy()

        obs, reward, success, truncated, info = env.step(action)

        if render:
            frame = env.render()
            frames.append(frame)

        if success or truncated:
            return {
                "success": success,
                "steps": step + 1,
                "final_dist": info["dist_to_target"],
                "frames": frames if render else None,
            }

    return {
        "success": False,
        "steps": max_steps,
        "final_dist": info["dist_to_target"],
        "frames": frames if render else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained world model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--block_decoder", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--diffusion_steps", type=int, default=10, help="Reduced steps for faster planning")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}")
    encoder, dynamics, config = load_model(args.checkpoint, device)

    print(f"Loading block decoder from {args.block_decoder}")
    block_decoder = load_block_decoder(args.block_decoder, config["latent_dim"], device)

    env = HomeTabletopEnv()
    target_pos = env.target_pos

    # Increase CEM parameters for better planning
    planner = CEMPlanner(
        dynamics,
        action_dim=2,
        horizon=5,
        num_samples=50,
        num_elites=10,
        num_iterations=3
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nRunning {args.episodes} evaluation episodes...")
    print(f"Target position: {target_pos}")
    print(f"Diffusion steps during planning: {args.diffusion_steps}")
    results = []

    for ep in range(args.episodes):
        result = evaluate_episode(
            env, encoder, dynamics, block_decoder, planner, device,
            target_pos, max_steps=args.max_steps,
            diffusion_steps=args.diffusion_steps, render=args.render,
        )
        results.append(result)

        status = "SUCCESS" if result["success"] else "FAIL"
        print(f"  Episode {ep+1}/{args.episodes}: {status} "
              f"(steps={result['steps']}, dist={result['final_dist']:.2f})")

    env.close()

    # Summary
    successes = [r for r in results if r["success"]]
    print(f"\n=== Evaluation Results ===")
    print(f"Success rate: {len(successes)}/{len(results)} ({100*len(successes)/len(results):.1f}%)")

    if successes:
        avg_steps = np.mean([r["steps"] for r in successes])
        print(f"Avg steps (success): {avg_steps:.1f}")

    avg_dist = np.mean([r["final_dist"] for r in results])
    print(f"Avg final distance: {avg_dist:.2f}")


if __name__ == "__main__":
    main()
