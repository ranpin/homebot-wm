"""Closed-loop evaluation of trained world model + CEM planner.

Loads trained encoder + dynamics + reward predictor, runs CEM planning in MuJoCo,
measures success rate and task completion metrics.
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


class RewardPredictor(nn.Module):
    """Predict distance to target from latent state."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


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


def load_reward_predictor(path: str, latent_dim: int, device: torch.device) -> RewardPredictor:
    """Load trained reward predictor."""
    reward_pred = RewardPredictor(latent_dim=latent_dim).to(device)
    reward_pred.load_state_dict(torch.load(path, map_location=device))
    reward_pred.eval()
    return reward_pred


def evaluate_episode(
    env: HomeTabletopEnv,
    encoder: ResNetEncoder,
    dynamics: nn.Module,
    reward_predictor: RewardPredictor,
    planner: CEMPlanner,
    device: torch.device,
    max_steps: int = 200,
    render: bool = False,
) -> dict:
    """Run one evaluation episode with closed-loop planning."""
    obs, info = env.reset()

    frames = []
    for step in range(max_steps):
        # Match training normalization (TrajectoryDataset divides images by 255).
        image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            latent = encoder(image)

            def score_predicted_dist(pred_latent: torch.Tensor) -> torch.Tensor:
                """Score: negative predicted distance (higher = better = closer)."""
                return -reward_predictor(pred_latent)

            if step % 20 == 0:
                print(f"    Step {step}: planning...", flush=True)

            action_seq = planner.plan(latent, score_predicted_dist)
            action = action_seq[0].cpu().numpy()

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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--reward_predictor", type=str, required=True, help="Path to reward predictor")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Save rendered frames")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}")
    encoder, dynamics, config = load_model(args.checkpoint, device)

    print(f"Loading reward predictor from {args.reward_predictor}")
    reward_predictor = load_reward_predictor(
        args.reward_predictor, config["latent_dim"], device
    )

    env = HomeTabletopEnv()
    planner = CEMPlanner(dynamics, action_dim=2, horizon=3, num_samples=20, num_elites=5, num_iterations=2)

    np.random.seed(args.seed)

    print(f"\nRunning {args.episodes} evaluation episodes...")
    results = []

    for ep in range(args.episodes):
        result = evaluate_episode(
            env, encoder, dynamics, reward_predictor, planner, device,
            max_steps=args.max_steps, render=args.render,
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
