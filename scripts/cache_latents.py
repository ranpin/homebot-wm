"""Precompute and cache frozen-encoder latents for the whole dataset.

Because the encoder is fully frozen, its latents are deterministic — so encoding
every frame once and reusing the result makes dynamics training ~30x faster
(the per-epoch ResNet pass over 360k images was the bottleneck, not the MLP).

Saves a dict of tensors (in dataset order) to --out:
    latent, next_latent : (N, latent_dim)
    action              : (N, action_dim)
    block_pos, next_block_pos : (N, 2)   # ground-truth from state, for eval
    config              : encoder config dict (adapter_dim, latent_dim, ...)

Usage:
    python scripts/cache_latents.py \
        --encoder_ckpt checkpoints/world_model_ep100.pt \
        --data data/trajectories.h5 --out data/latents_ep100.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen-encoder latents")
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/latents_ep100.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    config = ckpt["config"]
    encoder = ResNetEncoder(
        adapter_dim=config["adapter_dim"], output_dim=config["latent_dim"]
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    dataset = TrajectoryDataset(args.data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Encoding {len(dataset)} transitions from {args.data}")

    latents, next_latents, actions, block_pos, next_block_pos = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="caching"):
            image = batch["image"].to(device)
            next_image = batch["next_image"].to(device)
            latents.append(encoder(image).cpu())
            next_latents.append(encoder(next_image).cpu())
            actions.append(batch["action"])
            block_pos.append(batch["state"][:, 4:6])
            next_block_pos.append(batch["next_state"][:, 4:6])

    out = {
        "latent": torch.cat(latents),
        "next_latent": torch.cat(next_latents),
        "action": torch.cat(actions),
        "block_pos": torch.cat(block_pos),
        "next_block_pos": torch.cat(next_block_pos),
        "config": config,
        "encoder_ckpt": args.encoder_ckpt,
    }
    torch.save(out, args.out)
    print(f"Saved {out['latent'].shape[0]} latents (dim={out['latent'].shape[1]}) to {args.out}")


if __name__ == "__main__":
    main()
