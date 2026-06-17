"""Diagnose why world model training loss is stuck at 1.0.

Checks:
1. Data distribution (state/action ranges, image stats)
2. Latent space distribution (encoder output stats)
3. Gradient flow (are gradients non-zero?)
4. Loss function correctness
5. Quick overfit test (can model memorize a small batch?)
"""

import torch
import torch.nn as nn
import numpy as np

from wm_core.encoder.resnet_encoder import ResNetEncoder
from wm_core.dynamics.diffusion_dynamics import DiffusionDynamics
from wm_sim.dataset import TrajectoryDataset


def check_data_distribution(dataset: TrajectoryDataset) -> dict:
    """Check raw data statistics."""
    indices = np.random.choice(len(dataset), size=100, replace=False)
    states, actions, images = [], [], []

    for idx in indices:
        sample = dataset[idx]
        states.append(sample["state"].numpy())
        actions.append(sample["action"].numpy())
        images.append(sample["image"].numpy())

    states = np.stack(states)
    actions = np.stack(actions)
    images = np.stack(images)

    print("=== Data Distribution ===")
    print(f"State shape: {states.shape}, range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"  Per-dim mean: {states.mean(axis=0).round(3)}")
    print(f"  Per-dim std:  {states.std(axis=0).round(3)}")
    print(f"Action shape: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Per-dim mean: {actions.mean(axis=0).round(3)}")
    print(f"  Per-dim std:  {actions.std(axis=0).round(3)}")
    print(f"Image shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mean: {images.mean():.3f}, Std: {images.std():.3f}")
    print()
    return {"states": states, "actions": actions, "images": images}


def check_latent_space(encoder: nn.Module, dataset: TrajectoryDataset, device: torch.device):
    """Check encoder output distribution."""
    indices = np.random.choice(len(dataset), size=32, replace=False)
    images = torch.stack([dataset[i]["image"] for i in indices]).to(device)

    with torch.no_grad():
        latents = encoder(images)

    print("=== Latent Space ===")
    print(f"Latent shape: {latents.shape}")
    print(f"  Mean: {latents.mean():.6f}")
    print(f"  Std:  {latents.std():.6f}")
    print(f"  Min:  {latents.min():.6f}")
    print(f"  Max:  {latents.max():.6f}")
    print(f"  Norm (L2): {latents.norm(dim=1).mean():.6f}")
    print()
    return latents


def check_gradient_flow(encoder: nn.Module, dynamics: nn.Module, dataset: TrajectoryDataset, device: torch.device):
    """Check if gradients are flowing through the model."""
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    next_image = sample["next_image"].unsqueeze(0).to(device)
    action = sample["action"].unsqueeze(0).to(device)

    latent = encoder(image)
    next_latent = encoder(next_image).detach()

    loss = dynamics.loss(latent, action, next_latent)
    loss.backward()

    print("=== Gradient Flow ===")
    print(f"Loss value: {loss.item():.6f}")

    adapter_grads = []
    for name, param in encoder.adapter.named_parameters():
        if param.grad is not None:
            adapter_grads.append((name, param.grad.norm().item()))
            print(f"  Encoder adapter grad [{name}]: norm={param.grad.norm():.6e}")
        else:
            print(f"  Encoder adapter grad [{name}]: None!")

    dynamics_grads = []
    for name, param in list(dynamics.named_parameters())[:5]:
        if param.grad is not None:
            dynamics_grads.append((name, param.grad.norm().item()))
            print(f"  Dynamics grad [{name}]: norm={param.grad.norm():.6e}")
        else:
            print(f"  Dynamics grad [{name}]: None!")

    if adapter_grads:
        avg_adapter = np.mean([g for _, g in adapter_grads])
        print(f"  Avg adapter grad norm: {avg_adapter:.6e}")
    if dynamics_grads:
        avg_dyn = np.mean([g for _, g in dynamics_grads])
        print(f"  Avg dynamics grad norm: {avg_dyn:.6e}")
    print()


def check_loss_function(dynamics: nn.Module, device: torch.device):
    """Verify loss function is working correctly with known inputs."""
    print("=== Loss Function Sanity Check ===")

    batch = 8
    latent = torch.randn(batch, dynamics.latent_dim, device=device)
    action = torch.randn(batch, 2, device=device)
    next_latent_known = torch.randn(batch, dynamics.latent_dim, device=device)

    loss = dynamics.loss(latent, action, next_latent_known)
    print(f"Random input loss: {loss.item():.4f} (expected ~1.0 for random noise)")

    next_latent_zero = torch.zeros(batch, dynamics.latent_dim, device=device)
    loss_zero = dynamics.loss(latent, action, next_latent_zero)
    print(f"Zero next_latent loss: {loss_zero.item():.4f}")

    next_latent_same = latent.clone()
    loss_same = dynamics.loss(latent, action, next_latent_same)
    print(f"Same latent loss: {loss_same.item():.4f}")
    print()


def overfit_test(encoder: nn.Module, dynamics: nn.Module, dataset: TrajectoryDataset, device: torch.device):
    """Can the model memorize a small batch? If not, there's a bug."""
    print("=== Overfit Test (10 samples, 1000 steps) ===")

    indices = list(range(10))
    images = torch.stack([dataset[i]["image"] for i in indices]).to(device)
    next_images = torch.stack([dataset[i]["next_image"] for i in indices]).to(device)
    actions = torch.stack([dataset[i]["action"] for i in indices]).to(device)

    params = list(encoder.adapter.parameters()) + list(dynamics.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    initial_loss = None
    for step in range(1000):
        optimizer.zero_grad()
        latent = encoder(images)
        next_latent = encoder(next_images).detach()
        loss = dynamics.loss(latent, actions, next_latent)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

        if step % 200 == 0 or step == 999:
            print(f"  Step {step}: loss={loss.item():.6f}")

    final_loss = loss.item()
    improvement = initial_loss - final_loss
    print(f"\n  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {improvement:.6f}")

    if improvement > 0.2:
        print("  PASS: Model can learn (loss decreased significantly)")
    else:
        print("  FAIL: Model cannot learn - likely a bug in model or loss function")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset = TrajectoryDataset("data/trajectories.h5")
    print(f"Dataset size: {len(dataset)} samples\n")

    check_data_distribution(dataset)

    encoder = ResNetEncoder(adapter_dim=64, output_dim=64).to(device)
    dynamics = DiffusionDynamics(
        latent_dim=64, action_dim=2, hidden_dim=128,
        num_layers=3, num_diffusion_steps=50,
    ).to(device)
    encoder.train()
    dynamics.train()

    check_latent_space(encoder, dataset, device)
    check_loss_function(dynamics, device)
    check_gradient_flow(encoder, dynamics, dataset, device)
    overfit_test(encoder, dynamics, dataset, device)


if __name__ == "__main__":
    main()
