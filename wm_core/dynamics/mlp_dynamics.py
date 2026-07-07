"""Deterministic residual-MLP latent dynamics.

Diagnostics showed the frame-to-frame latent change in this environment is tiny
(identity "no-move" MSE ~0.005), while the diffusion dynamics model — sampling
each next latent from pure noise — injected more variance than signal and scored
*worse* than identity. This model instead regresses the residual delta latent
and adds it back, and is zero-initialized so it *starts* exactly at the identity
baseline and can only improve from there.

Exposes the same ``predict_next(latent, action, num_steps=None)`` interface as
DiffusionDynamics so it is a drop-in for the CEM planner and eval scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDynamics(nn.Module):
    """Predict next latent as ``latent + MLP(latent, action)`` (residual)."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers = [nn.Linear(latent_dim + action_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, latent_dim)

        # Zero-init the head so the untrained model is the identity map
        # (predicts delta=0). Training can only reduce the rollout error below
        # the identity baseline.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def delta(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.backbone(torch.cat([latent, action], dim=-1))
        return self.head(h)

    def loss(self, latent: torch.Tensor, action: torch.Tensor, next_latent: torch.Tensor) -> torch.Tensor:
        pred_delta = self.delta(latent, action)
        target_delta = next_latent - latent
        return F.mse_loss(pred_delta, target_delta)

    @torch.no_grad()
    def predict_next(
        self, latent: torch.Tensor, action: torch.Tensor, num_steps: int | None = None
    ) -> torch.Tensor:
        """Return the predicted next latent. ``num_steps`` is accepted for
        interface parity with DiffusionDynamics and ignored."""
        return latent + self.delta(latent, action)
