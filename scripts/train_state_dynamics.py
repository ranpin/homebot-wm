"""Train a residual-MLP dynamics model directly on the low-dim TRUE state.

Motivation: 1-step *latent* dynamics carried almost no action signal (most
transitions the block is static while the agent navigates), so CEM could not
plan. The environment's 6-D state `[agent_xy, agent_vxy, block_xy]` is fully
observed in the dataset and physically learnable — a state-space world model
gives CEM a usable model and is the fastest path to a working demo. (The
vision-latent world model remains the harder research track.)

Reuses MLPDynamics (dimension-agnostic residual regressor) with latent_dim=6.
No encoder / no images — reads states straight from the HDF5.

Usage:
    python scripts/train_state_dynamics.py --data data/trajectories.h5 \
        --epochs 60 --hidden_dim 256
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

from wm_core.dynamics.mlp_dynamics import MLPDynamics


def load_transitions(path: str):
    """Return (state, action, next_state) tensors over all episodes."""
    states, actions, next_states = [], [], []
    with h5py.File(path, "r") as f:
        for k in sorted(f.keys()):
            if not k.startswith("episode"):
                continue
            ep = f[k]
            s = np.asarray(ep["observations/states"])
            a = np.asarray(ep["actions"])
            n = len(a)
            states.append(s[:n])
            actions.append(a[:n])
            next_states.append(s[1:n + 1])
    S = torch.from_numpy(np.concatenate(states)).float()
    A = torch.from_numpy(np.concatenate(actions)).float()
    N = torch.from_numpy(np.concatenate(next_states)).float()
    return S, A, N


@torch.no_grad()
def validate(loader, dynamics, device):
    dynamics.eval()
    model_se = identity_se = count = 0
    blk_model = blk_ident = 0.0
    for s, a, n in loader:
        s, a, n = s.to(device), a.to(device), n.to(device)
        pred = dynamics.predict_next(s, a)
        model_se += ((pred - n) ** 2).mean(dim=-1).sum().item()
        identity_se += ((s - n) ** 2).mean(dim=-1).sum().item()
        # block-position (dims 4:6) euclidean error — what planning actually needs
        blk_model += torch.norm(pred[:, 4:6] - n[:, 4:6], dim=-1).sum().item()
        blk_ident += torch.norm(s[:, 4:6] - n[:, 4:6], dim=-1).sum().item()
        count += s.shape[0]
    dynamics.train()
    return (model_se / count, identity_se / count, blk_model / count, blk_ident / count)


def main() -> None:
    p = argparse.ArgumentParser(description="Train state-space residual dynamics")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_name", type=str, default="state_dynamics.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    S, A, N = load_transitions(args.data)
    print(f"Loaded {S.shape[0]} transitions, state_dim={S.shape[1]}, action_dim={A.shape[1]}")
    full = TensorDataset(S, A, N)
    val_size = int(len(full) * args.val_split)
    train_set, val_set = random_split(full, [len(full) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False)

    dynamics = MLPDynamics(latent_dim=S.shape[1], action_dim=A.shape[1],
                           hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = AdamW(dynamics.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_config = {
        "latent_dim": S.shape[1], "action_dim": A.shape[1],
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "dynamics_type": "mlp", "state_space": True,
    }

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        dynamics.train()
        running = 0.0
        for step, (s, a, n) in enumerate(train_loader):
            s, a, n = s.to(device), a.to(device), n.to(device)
            loss = dynamics.loss(s, a, n)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        m_mse, i_mse, blk_m, blk_i = validate(val_loader, dynamics, device)
        verdict = "OK" if m_mse < i_mse else "FAIL"
        print(f"Epoch {epoch}: loss={running/len(train_loader):.5f} "
              f"val_mse={m_mse:.5f} identity={i_mse:.5f} | block_err={blk_m:.4f} "
              f"(identity {blk_i:.4f}) -> {verdict}")
        if m_mse < best:
            best = m_mse
            torch.save({"epoch": epoch, "dynamics_state": dynamics.state_dict(),
                        "config": out_config, "val_mse": m_mse, "block_err": blk_m},
                       save_dir / args.save_name)
    print(f"Saved best to {save_dir / args.save_name} (val_mse={best:.5f})")


if __name__ == "__main__":
    main()
