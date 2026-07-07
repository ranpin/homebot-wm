"""Cross-Entropy Method (CEM) action planner.

Uses the dynamics model to evaluate candidate action sequences
and iteratively refines the distribution toward high-scoring elites.
"""

import torch
import torch.nn as nn


class CEMPlanner:
    """CEM planner conditioned on a dynamics model.

    Args:
        dynamics: Diffusion dynamics model for next-state prediction.
        num_samples: Number of action sequences to sample per iteration.
        num_elites: Number of top-scoring sequences to keep.
        horizon: Number of timesteps in each action sequence.
        action_dim: Dimension of the action space.
        num_iterations: Number of CEM refinement iterations.
        action_low: Lower bound of action space.
        action_high: Upper bound of action space.
    """

    def __init__(
        self,
        dynamics: nn.Module,
        num_samples: int = 32,
        num_elites: int = 4,
        horizon: int = 10,
        action_dim: int = 2,
        num_iterations: int = 3,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        self.dynamics = dynamics
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_iterations = num_iterations
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(
        self,
        latent: torch.Tensor,
        score_fn,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Plan an action sequence starting from latent state.

        Args:
            latent: Current latent state (1, latent_dim).
            score_fn: Callable(next_latent) -> scalar score. Higher is better.
            num_steps: Optional reduced number of diffusion steps for the
                dynamics rollout (fewer = faster planning). Defaults to the
                model's full step count.

        Returns:
            Best action sequence (horizon, action_dim).
        """
        device = latent.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * 0.5

        for _ in range(self.num_iterations):
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
            actions = mean[None] + std[None] * noise
            actions = actions.clamp(self.action_low, self.action_high)

            scores = torch.zeros(self.num_samples, device=device)
            for s in range(self.num_samples):
                state = latent
                for t in range(self.horizon):
                    state = self.dynamics.predict_next(
                        state, actions[s, t].unsqueeze(0), num_steps=num_steps
                    )
                scores[s] = score_fn(state)

            elite_idx = scores.topk(self.num_elites).indices
            elite_actions = actions[elite_idx]

            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

        return mean
