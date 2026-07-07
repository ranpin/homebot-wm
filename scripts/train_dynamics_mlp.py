"""Train a deterministic residual-MLP latent dynamics model.

Uses a FROZEN encoder loaded from an existing world_model checkpoint, so the
latent space (and therefore the already-trained block decoder) stays valid.
Only the MLP dynamics is trained, on the residual (next_latent - latent).

Two data paths:
  --latent_cache PATH : train on precomputed latents (from scripts/cache_latents.py)
                        — no per-epoch ResNet pass, ~30x faster. PREFERRED.
  (otherwise)         : encode images on the fly each epoch (slow, data-bound).

Acceptance metric (printed each epoch): validation rollout MSE must drop below
the identity "no-move" baseline — the bar the diffusion model failed.

Usage (fast path):
    python scripts/cache_latents.py --encoder_ckpt checkpoints/world_model_ep100.pt \
        --data data/trajectories.h5 --out data/latents_ep100.pt
    python scripts/train_dynamics_mlp.py --encoder_ckpt checkpoints/world_model_ep100.pt \
        --latent_cache data/latents_ep100.pt --epochs 30 --hidden_dim 256
"""

import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from wm_core.dynamics.mlp_dynamics import MLPDynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset


@torch.no_grad()
def validate_images(loader, encoder, dynamics, device):
    dynamics.eval()
    model_se, identity_se, count = 0.0, 0.0, 0
    for batch in loader:
        latent = encoder(batch["image"].to(device))
        next_latent = encoder(batch["next_image"].to(device))
        action = batch["action"].to(device)
        pred = dynamics.predict_next(latent, action)
        model_se += ((pred - next_latent) ** 2).mean(dim=-1).sum().item()
        identity_se += ((latent - next_latent) ** 2).mean(dim=-1).sum().item()
        count += latent.shape[0]
    dynamics.train()
    return model_se / max(count, 1), identity_se / max(count, 1)


@torch.no_grad()
def validate_cache(loader, dynamics, device):
    dynamics.eval()
    model_se, identity_se, count = 0.0, 0.0, 0
    for latent, action, next_latent in loader:
        latent, action, next_latent = latent.to(device), action.to(device), next_latent.to(device)
        pred = dynamics.predict_next(latent, action)
        model_se += ((pred - next_latent) ** 2).mean(dim=-1).sum().item()
        identity_se += ((latent - next_latent) ** 2).mean(dim=-1).sum().item()
        count += latent.shape[0]
    dynamics.train()
    return model_se / max(count, 1), identity_se / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual-MLP dynamics")
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="world_model checkpoint for the frozen encoder (config + weights)")
    parser.add_argument("--data", type=str, default=None, help="HDF5 dataset (image path)")
    parser.add_argument("--latent_cache", type=str, default=None,
                        help="precomputed latents from cache_latents.py (fast path)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="dynamics_mlp.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.latent_cache and not args.data:
        parser.error("provide --latent_cache (fast) or --data (image path)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Encoder config + weights (needed in the output checkpoint for eval/diagnose).
    enc_ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
    enc_config = enc_ckpt["config"]
    latent_dim = enc_config["latent_dim"]

    use_cache = args.latent_cache is not None
    encoder = None
    if use_cache:
        cache = torch.load(args.latent_cache, map_location="cpu")
        full = TensorDataset(cache["latent"], cache["action"], cache["next_latent"])
        print(f"Loaded latent cache: {cache['latent'].shape[0]} transitions (dim={cache['latent'].shape[1]})")
    else:
        encoder = ResNetEncoder(adapter_dim=enc_config["adapter_dim"], output_dim=latent_dim).to(device)
        encoder.load_state_dict(enc_ckpt["encoder_state"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        full = TrajectoryDataset(args.data)
        print(f"Image path (slow): {len(full)} transitions")

    val_size = int(len(full) * args.val_split)
    train_set, val_set = random_split(full, [len(full) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    dynamics = MLPDynamics(latent_dim=latent_dim, action_dim=2,
                           hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    dynamics.train()
    optimizer = AdamW(dynamics.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_config = {
        "adapter_dim": enc_config["adapter_dim"], "latent_dim": latent_dim,
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "action_dim": 2, "diffusion_steps": enc_config.get("diffusion_steps", 50),
        "dynamics_type": "mlp",
    }

    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        dynamics.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            if use_cache:
                latent, action, next_latent = (t.to(device) for t in batch)
            else:
                with torch.no_grad():
                    latent = encoder(batch["image"].to(device))
                    next_latent = encoder(batch["next_image"].to(device))
                action = batch["action"].to(device)
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

        if use_cache:
            model_mse, identity_mse = validate_cache(val_loader, dynamics, device)
        else:
            model_mse, identity_mse = validate_images(val_loader, encoder, dynamics, device)
        verdict = "OK (beats identity)" if model_mse < identity_mse else "FAIL (worse than identity)"
        print(f"Epoch {epoch}: train_loss={running_loss / len(train_loader):.5f} "
              f"val_rollout_mse={model_mse:.5f} identity_mse={identity_mse:.5f} -> {verdict}")

        if model_mse < best_mse:
            best_mse = model_mse
            torch.save({
                "epoch": epoch,
                "encoder_state": enc_ckpt["encoder_state"],
                "dynamics_state": dynamics.state_dict(),
                "config": out_config,
                "val_rollout_mse": model_mse, "identity_mse": identity_mse,
            }, save_dir / args.save_name)
            print(f"  Saved best to {save_dir / args.save_name} (val_mse={model_mse:.5f})")


if __name__ == "__main__":
    main()
