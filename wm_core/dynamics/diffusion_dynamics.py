"""Small diffusion-based dynamics model for next-state prediction.

Predicts next latent state given (current_latent, action) using DDPM.
Designed to fit within 8GB VRAM: hidden_dim=128, 3 layers, 50 steps.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freq[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionDynamics(nn.Module):
    """DDPM dynamics model: predict next latent from (current_latent, action).

    Args:
        latent_dim: Dimension of the latent space.
        action_dim: Dimension of the action space.
        hidden_dim: Hidden dimension of the denoising MLP.
        num_layers: Number of hidden layers.
        num_diffusion_steps: Number of DDPM timesteps.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_diffusion_steps: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_diffusion_steps = num_diffusion_steps

        cond_dim = latent_dim + action_dim + latent_dim  # current_latent + action + noisy_next_latent
        time_dim = hidden_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(hidden_dim, latent_dim)

        beta = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    def _noise_forward(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ab = self.sqrt_alpha_bar[t][:, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None]
        return sqrt_ab * x0 + sqrt_1mab * noise

    def loss(self, latent: torch.Tensor, action: torch.Tensor, next_latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=latent.device)
        noise = torch.randn_like(next_latent)

        noisy_next = self._noise_forward(next_latent, t, noise)

        cond = torch.cat([latent, action, noisy_next], dim=-1)
        h = self.cond_proj(cond) + self.time_embed(t)
        h = self.backbone(h)
        pred_noise = self.head(h)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict_next(self, latent: torch.Tensor, action: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """Sample next latent from the DDPM reverse process.

        With ``num_steps`` unset (or >= the model's step count) this runs the
        full stochastic DDPM reverse chain from t=T-1 down to t=0. With a smaller
        ``num_steps`` it runs deterministic DDIM sampling over an evenly spaced
        subsequence of timesteps that still terminates at t=0 — fewer, larger
        steps rather than a truncated (and therefore still-noisy) chain.
        """
        device = latent.device
        batch_size = latent.shape[0]
        T = self.num_diffusion_steps
        cond_base = torch.cat([latent, action], dim=-1)
        x = torch.randn(batch_size, self.latent_dim, device=device)

        def eps_at(x_t: torch.Tensor, i: int) -> torch.Tensor:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            cond = torch.cat([cond_base, x_t], dim=-1)
            h = self.cond_proj(cond) + self.time_embed(t)
            h = self.backbone(h)
            return self.head(h)

        if num_steps is None or num_steps >= T:
            # Full stochastic DDPM reverse process.
            for i in reversed(range(T)):
                pred_noise = eps_at(x, i)
                alpha_t = self.alpha[i]
                alpha_bar_t = self.alpha_bar[i]
                beta_t = self.beta[i]
                x = (x - beta_t / (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_t.sqrt()
                if i > 0:
                    x = x + beta_t.sqrt() * torch.randn_like(x)
            return x

        # Respaced deterministic DDIM sampling (eta=0) over a subsequence ending at t=0.
        seq = torch.linspace(T - 1, 0, max(1, num_steps)).round().long().tolist()
        for idx, i in enumerate(seq):
            pred_noise = eps_at(x, i)
            alpha_bar_t = self.alpha_bar[i]
            x0 = (x - (1.0 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            i_prev = seq[idx + 1] if idx + 1 < len(seq) else -1
            if i_prev < 0:
                x = x0
            else:
                alpha_bar_prev = self.alpha_bar[i_prev]
                x = alpha_bar_prev.sqrt() * x0 + (1.0 - alpha_bar_prev).sqrt() * pred_noise
        return x
