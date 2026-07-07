"""Component-wise diagnostic for the world-model pipeline (CPU-friendly).

Answers "where does the 0% success come from?" by measuring each learned
component against dataset ground truth, without running the (slow) closed-loop
MuJoCo eval:

  1. Encoder+decoder block-position error, with correct [0,1] normalization.
  2. The same with WRONG [0,255] input, to quantify the train/eval mismatch.
  3. Dynamics 1-step rollout error vs an identity ("no change") baseline —
     does the world model actually beat predicting no motion?
  4. End-to-end planning-signal error: block position decoded from the model's
     predicted next latent vs the true next block position. This is exactly the
     signal CEM optimizes.

Usage:
    python scripts/diagnose_components.py \
        --checkpoint checkpoints/world_model_ep100.pt \
        --block_decoder checkpoints/block_decoder.pt \
        --data data/trajectories.h5 --n 512
"""

import argparse

import torch
import torch.nn as nn

from wm_core.dynamics.diffusion_dynamics import DiffusionDynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


class BlockPositionDecoder(nn.Module):
    """Same architecture as scripts/train_block_decoder.py."""

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


def euclidean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose world-model components")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--block_decoder", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n", type=int, default=512, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rollout_steps", type=int, default=None,
                        help="Diffusion steps for predict_next (default: model full steps)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    encoder = ResNetEncoder(adapter_dim=config["adapter_dim"], output_dim=config["latent_dim"]).to(device)
    dynamics = DiffusionDynamics(
        latent_dim=config["latent_dim"], action_dim=2,
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        num_diffusion_steps=config["diffusion_steps"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    dynamics.load_state_dict(ckpt["dynamics_state"])
    encoder.eval()
    dynamics.eval()

    decoder = BlockPositionDecoder(latent_dim=config["latent_dim"]).to(device)
    decoder.load_state_dict(torch.load(args.block_decoder, map_location=device))
    decoder.eval()

    dataset = TrajectoryDataset(args.data)
    n = min(args.n, len(dataset))
    idxs = torch.randperm(len(dataset))[:n].tolist()
    print(f"Dataset: {len(dataset)} transitions; evaluating {n}\n")

    dec_err_ok, dec_err_bad = [], []
    roll_mse_model, roll_mse_identity = [], []
    plan_err_model, plan_err_static = [], []

    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            batch_idx = idxs[start:start + args.batch_size]
            images = torch.stack([dataset[i]["image"] for i in batch_idx]).to(device)       # [0,1]
            next_images = torch.stack([dataset[i]["next_image"] for i in batch_idx]).to(device)
            actions = torch.stack([dataset[i]["action"] for i in batch_idx]).to(device)
            states = torch.stack([dataset[i]["state"] for i in batch_idx]).to(device)
            next_states = torch.stack([dataset[i]["next_state"] for i in batch_idx]).to(device)
            block_pos = states[:, 4:6]
            next_block_pos = next_states[:, 4:6]

            # (1)/(2) Decoder accuracy: correct [0,1] vs wrong [0,255] input.
            latent = encoder(images)
            dec_err_ok.append(euclidean(decoder(latent), block_pos))
            latent_bad = encoder(images * 255.0)
            dec_err_bad.append(euclidean(decoder(latent_bad), block_pos))

            # (3) Dynamics 1-step rollout vs identity baseline.
            next_latent = encoder(next_images)
            pred_latent = dynamics.predict_next(latent, actions, num_steps=args.rollout_steps)
            roll_mse_model.append(((pred_latent - next_latent) ** 2).mean(dim=-1))
            roll_mse_identity.append(((latent - next_latent) ** 2).mean(dim=-1))

            # (4) End-to-end planning signal: decoded predicted-next block pos vs truth.
            plan_err_model.append(euclidean(decoder(pred_latent), next_block_pos))
            plan_err_static.append(euclidean(decoder(latent), next_block_pos))

    def stat(chunks):
        v = torch.cat(chunks)
        return v.mean().item(), v.std().item()

    print("=== (1) Block-decoder position error, CORRECT [0,1] input ===")
    m, s = stat(dec_err_ok); print(f"  mean euclidean err: {m:.3f} +/- {s:.3f}  (success threshold = 1.0)")
    print("=== (2) Block-decoder position error, WRONG [0,255] input (eval bug) ===")
    m, s = stat(dec_err_bad); print(f"  mean euclidean err: {m:.3f} +/- {s:.3f}")
    print("=== (3) Dynamics 1-step rollout latent MSE ===")
    mm, _ = stat(roll_mse_model); im, _ = stat(roll_mse_identity)
    print(f"  model predict_next : {mm:.4f}")
    print(f"  identity (no move) : {im:.4f}")
    verdict = "BEATS identity" if mm < im else "WORSE than identity (model not learning dynamics)"
    print(f"  -> model {verdict}")
    print("=== (4) End-to-end planning signal (decoded predicted-next block pos vs truth) ===")
    m, s = stat(plan_err_model); print(f"  model rollout+decode err: {m:.3f} +/- {s:.3f}")
    m, s = stat(plan_err_static); print(f"  static (decode current) : {m:.3f}")


if __name__ == "__main__":
    main()
