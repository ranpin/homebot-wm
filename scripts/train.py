"""Train world model from offline trajectory dataset.

Usage:
    python scripts/train.py --data data/trajectories.h5 --epochs 100

Designed for RTX 3070 (8GB VRAM):
  - AMP FP16 mixed precision
  - Gradient accumulation (effective batch = 64)
  - Frozen encoder, only adapter + dynamics are trained
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from wm_core.dynamics.diffusion_dynamics import DiffusionDynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


def get_trainable_params(encoder: nn.Module, dynamics: nn.Module) -> list[nn.Parameter]:
    params = []
    for p in encoder.adapter.parameters():
        if p.requires_grad:
            params.append(p)
    params.extend(p for p in dynamics.parameters() if p.requires_grad)
    return params


def train_step(
    batch: dict[str, torch.Tensor],
    encoder: nn.Module,
    dynamics: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    image = batch["image"].to(device)
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    next_image = batch["next_image"].to(device)

    latent = encoder(image)
    next_latent = encoder(next_image).detach()

    loss = dynamics.loss(latent, action, next_latent)
    return loss


def evaluate(
    loader: DataLoader,
    encoder: nn.Module,
    dynamics: nn.Module,
    device: torch.device,
) -> float:
    encoder.eval()
    dynamics.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            loss = train_step(batch, encoder, dynamics, device)
            total_loss += loss.item() * batch["image"].shape[0]
            count += batch["image"].shape[0]

    encoder.train()
    dynamics.train()
    return total_loss / max(count, 1)


def log_vram(prefix: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{prefix}] VRAM: allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train world model")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 trajectory file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    print(f"Loading dataset: {args.data}")
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
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    encoder = ResNetEncoder(adapter_dim=args.adapter_dim, output_dim=args.latent_dim).to(device)
    dynamics = DiffusionDynamics(
        latent_dim=args.latent_dim,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    encoder.train()
    dynamics.train()

    params = get_trainable_params(encoder, dynamics)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    log_vram("after model init")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        dynamics.load_state_dict(ckpt["dynamics_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        dynamics.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            with autocast("cuda", enabled=torch.cuda.is_available()):
                loss = train_step(batch, encoder, dynamics, device)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

            if (step + 1) % (args.log_interval * args.grad_accum) == 0:
                log_vram(f"step {step+1}")

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = evaluate(val_loader, encoder, dynamics, device)

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
        )

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "dynamics_state": dynamics.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args),
            }
            torch.save(ckpt, save_dir / f"world_model_ep{epoch:03d}.pt")
            print(f"  Saved checkpoint to {save_dir / f'world_model_ep{epoch:03d}.pt'}")


if __name__ == "__main__":
    main()
