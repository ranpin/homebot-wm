"""Train a deterministic residual-MLP latent dynamics model.

Uses a FROZEN encoder loaded from an existing world_model checkpoint, so the
latent space (and therefore the already-trained block decoder) stays valid.
Only the MLP dynamics is trained, on the residual (next_latent - latent).

Acceptance metric (printed each epoch): validation rollout MSE must drop below
the identity "no-move" baseline — that is the bar the diffusion model failed.

Usage (run on the 3070 once the GPU is free):
    python scripts/train_dynamics_mlp.py \
        --data data/trajectories.h5 \
        --encoder_ckpt checkpoints/world_model_ep100.pt \
        --epochs 30 --hidden_dim 256 --num_layers 3
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from wm_core.dynamics.mlp_dynamics import MLPDynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


@torch.no_grad()
def validate(loader, encoder, dynamics, device):
    """Return (model_rollout_mse, identity_baseline_mse) over the val set."""
    dynamics.eval()
    model_se, identity_se, count = 0.0, 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        next_image = batch["next_image"].to(device)
        action = batch["action"].to(device)
        latent = encoder(image)
        next_latent = encoder(next_image)
        pred = dynamics.predict_next(latent, action)
        bs = image.shape[0]
        model_se += ((pred - next_latent) ** 2).mean(dim=-1).sum().item()
        identity_se += ((latent - next_latent) ** 2).mean(dim=-1).sum().item()
        count += bs
    dynamics.train()
    return model_se / max(count, 1), identity_se / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual-MLP dynamics")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="world_model checkpoint to load the frozen encoder from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="dynamics_mlp.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Frozen encoder from existing checkpoint ---
    enc_ckpt = torch.load(args.encoder_ckpt, map_location=device)
    enc_config = enc_ckpt["config"]
    encoder = ResNetEncoder(
        adapter_dim=enc_config["adapter_dim"],
        output_dim=enc_config["latent_dim"],
    ).to(device)
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded frozen encoder from {args.encoder_ckpt} (latent_dim={enc_config['latent_dim']})")

    # --- Data ---
    dataset = TrajectoryDataset(args.data)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")

    # --- Model ---
    dynamics = MLPDynamics(
        latent_dim=enc_config["latent_dim"],
        action_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    dynamics.train()

    optimizer = AdamW(dynamics.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint config compatible with build_dynamics / eval load_model.
    out_config = {
        "adapter_dim": enc_config["adapter_dim"],
        "latent_dim": enc_config["latent_dim"],
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "action_dim": 2,
        "diffusion_steps": enc_config.get("diffusion_steps", 50),
        "dynamics_type": "mlp",
    }

    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        dynamics.load_state_dict(ckpt["dynamics_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1

    best_mse = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        dynamics.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            image = batch["image"].to(device)
            next_image = batch["next_image"].to(device)
            action = batch["action"].to(device)
            with torch.no_grad():
                latent = encoder(image)
                next_latent = encoder(next_image)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                loss = dynamics.loss(latent, action, next_latent) / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{running_loss / (step + 1):.5f}")

        model_mse, identity_mse = validate(val_loader, encoder, dynamics, device)
        verdict = "OK (beats identity)" if model_mse < identity_mse else "FAIL (worse than identity)"
        print(
            f"Epoch {epoch}: train_loss={running_loss / len(train_loader):.5f} "
            f"val_rollout_mse={model_mse:.5f} identity_mse={identity_mse:.5f} -> {verdict}"
        )

        if model_mse < best_mse:
            best_mse = model_mse
            ckpt = {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "dynamics_state": dynamics.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": out_config,
                "val_rollout_mse": model_mse,
                "identity_mse": identity_mse,
            }
            torch.save(ckpt, save_dir / args.save_name)
            print(f"  Saved best checkpoint to {save_dir / args.save_name} (val_mse={model_mse:.5f})")


if __name__ == "__main__":
    main()
