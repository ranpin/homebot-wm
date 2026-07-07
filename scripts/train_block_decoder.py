"""Train block position decoder: latent -> (block_x, block_y).

Allows CEM planner to optimize actual distance to target.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


class BlockPositionDecoder(nn.Module):
    """Decode block position from latent representation."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # block_x, block_y
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class BlockPositionDataset(Dataset):
    """Wrap TrajectoryDataset to provide (image, block_position) pairs."""

    def __init__(self, traj_path: str):
        self.traj_ds = TrajectoryDataset(traj_path)

    def __len__(self):
        return len(self.traj_ds)

    def __getitem__(self, idx):
        sample = self.traj_ds[idx]
        # Extract block position from state: [agent_x, agent_y, agent_vx, agent_vy, block_x, block_y]
        state = sample["state"]
        block_pos = state[4:6]  # Already float32 from dataset
        return sample["image"], block_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="checkpoints/block_decoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen encoder
    encoder_ckpt = torch.load(args.encoder_ckpt, map_location=device)
    config = encoder_ckpt["config"]
    encoder = ResNetEncoder(
        adapter_dim=config["adapter_dim"],
        output_dim=config["latent_dim"],
    ).to(device)
    encoder.load_state_dict(encoder_ckpt["encoder_state"])
    encoder.eval()

    # Load dataset
    dataset = BlockPositionDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Train decoder
    decoder = BlockPositionDecoder(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        for images, block_positions in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            block_positions = block_positions.to(device)

            with torch.no_grad():
                latents = encoder(images)

            pred_block_pos = decoder(latents)
            loss = criterion(pred_block_pos, block_positions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), output_path)
    print(f"Saved block decoder to {output_path}")


if __name__ == "__main__":
    main()
