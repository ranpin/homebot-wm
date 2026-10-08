"""Multi-step action sensitivity: does the action's effect grow with rollout horizon?

Hypothesis: the action is a force -> it changes VELOCITY, which only becomes a
POSITION change over several steps. A single image carries position, not velocity,
so at 1 step the action is nearly invisible. If action_effect(k) grows with k,
CEM needs a longer horizon (or the latent needs temporal context), rather than
the 1-step signal being fundamentally broken.

The dynamics here is deterministic (residual MLP), so the spread of the predicted
future across DIFFERENT action sequences IS the action effect -- no noise floor.

Two views are reported:
  DECODED  -- block-position units via the block decoder. Intuitive, but NOT
              comparable across models: each model has its own decoder with its
              own gain/accuracy, which scales the number.
  LATENT   -- decoder-independent. Normalised two ways so different latent
              spaces (e.g. 64-d pooled vs 144-d spatial) can be compared:
                /true_1step  = action effect as a fraction of that latent's own
                               real frame-to-frame change
                action_frac  = fraction of the rollout's movement that is
                               attributable to varying the action
"""
import argparse
import h5py
import numpy as np
import torch

from wm_core.dynamics import build_dynamics
from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_sim.dataset import TrajectoryDataset
from scripts.diagnose_components import BlockPositionDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--block_decoder", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n_states", type=int, default=64)
    p.add_argument("--n_seqs", type=int, default=16)
    p.add_argument("--max_horizon", type=int, default=20)
    p.add_argument("--n_delta", type=int, default=256,
                   help="transitions used to measure the true 1-step latent delta")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    encoder = ResNetEncoder.from_config(config).to(device)
    dynamics = build_dynamics(config).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    dynamics.load_state_dict(ckpt["dynamics_state"])
    encoder.eval(); dynamics.eval()
    latent_dim = encoder.latent_dim
    action_dim = config.get("action_dim", 2)
    print(f"Encoder spatial={encoder.spatial} latent_dim={latent_dim} "
          f"dynamics={config.get('dynamics_type')}")

    decoder = BlockPositionDecoder(latent_dim=latent_dim).to(device)
    decoder.load_state_dict(torch.load(args.block_decoder, map_location=device))
    decoder.eval()

    dataset = TrajectoryDataset(args.data, obs_horizon=config.get("obs_horizon", 1))
    idxs = torch.randperm(len(dataset))[:args.n_states].tolist()
    with torch.no_grad():
        latents = torch.stack([encoder(dataset[i]["image"][None].to(device))[0] for i in idxs])

    # Decoder-independent scale: this latent space's own true 1-step change.
    didx = torch.randperm(len(dataset))[:args.n_delta].tolist()
    with torch.no_grad():
        d = torch.stack([encoder(dataset[i]["image"][None].to(device))[0] for i in didx])
        dn = torch.stack([encoder(dataset[i]["next_image"][None].to(device))[0] for i in didx])
    true_1step = torch.norm(dn - d, dim=-1).mean().item()
    print(f"true 1-step latent delta (this encoder): {true_1step:.5f}")

    N, K, A = len(idxs), args.max_horizon, args.n_seqs
    gen = torch.Generator(device=device).manual_seed(args.seed)
    acts = (torch.rand(A, N, K, action_dim, generator=gen, device=device) * 2 - 1)

    states = latents.unsqueeze(0).expand(A, -1, -1).contiguous()          # (A,N,D)
    block_traj = torch.zeros(A, N, K + 1, 2, device=device)
    block_traj[:, :, 0] = decoder(latents).unsqueeze(0)
    lat_traj = torch.zeros(A, N, K + 1, latent_dim, device=device)
    lat_traj[:, :, 0] = latents.unsqueeze(0)
    with torch.no_grad():
        for k in range(K):
            states = dynamics.predict_next(states.reshape(A * N, -1),
                                           acts[:, :, k].reshape(A * N, -1)).reshape(A, N, -1)
            block_traj[:, :, k + 1] = decoder(states.reshape(A * N, -1)).reshape(A, N, 2)
            lat_traj[:, :, k + 1] = states

    # --- DECODED view (block-position units; decoder-gain dependent) ---
    dec_ae = torch.norm(block_traj.std(dim=0), dim=-1).mean(dim=0)        # (K+1,)

    # --- LATENT view (decoder-independent) ---
    lat_ae = lat_traj.std(dim=0).norm(dim=-1).mean(dim=0)                 # (K+1,) spread across seqs
    roll = (lat_traj[:, :, 1:] - lat_traj[:, :, 0:1]).norm(dim=-1)        # (A,N,K)
    roll_mag = roll.mean(dim=(0, 1))                                      # (K,)

    # ground-truth k-step block displacement (physical scale, model-independent)
    want = [k for k in [1, 2, 5, 10, 15, 20] if k <= K]
    acc = {k: [] for k in want}
    with h5py.File(args.data, "r") as f:
        rng = np.random.default_rng(args.seed)
        keys = [k for k in f.keys() if k.startswith("episode")]
        for _ in range(400):
            st = np.asarray(f[rng.choice(keys)]["observations/states"])[:, 4:6]
            t = int(rng.integers(0, max(1, len(st) - K)))
            for k in want:
                if t + k < len(st):
                    acc[k].append(np.linalg.norm(st[t + k] - st[t]))
    true_disp = {k: float(np.mean(v)) for k, v in acc.items() if v}

    print(f"\n{'k':>3} | {'DECODED ae':>10} {'/true_disp':>10} | "
          f"{'LATENT ae':>10} {'/true_1step':>11} {'rollout_mag':>11} {'action_frac':>11}")
    for k in range(1, K + 1):
        td = true_disp.get(k)
        td_s = f"{dec_ae[k].item()/td:.3f}" if td else "-"
        print(f"{k:>3} | {dec_ae[k].item():>10.4f} {td_s:>10} | "
              f"{lat_ae[k].item():>10.4f} {lat_ae[k].item()/true_1step:>11.3f} "
              f"{roll_mag[k-1].item():>11.4f} {lat_ae[k].item()/max(roll_mag[k-1].item(),1e-9):>11.3f}")

    g_dec = dec_ae[K].item() / max(dec_ae[1].item(), 1e-9)
    g_lat = lat_ae[K].item() / max(lat_ae[1].item(), 1e-9)
    print(f"\ngrowth k=1 -> k={K}:  decoded {g_dec:.1f}x   latent {g_lat:.1f}x")
    print(f"latent action_fraction: k=1 {lat_ae[1].item()/max(roll_mag[0].item(),1e-9):.3f}"
          f"  ->  k={K} {lat_ae[K].item()/max(roll_mag[K-1].item(),1e-9):.3f}")


if __name__ == "__main__":
    main()
