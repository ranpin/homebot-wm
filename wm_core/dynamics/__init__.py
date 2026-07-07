"""Dynamics prediction module — predicts next state given current state and action."""

import torch.nn as nn


def build_dynamics(config: dict) -> nn.Module:
    """Construct a dynamics model from a checkpoint ``config`` dict.

    Dispatches on ``config['dynamics_type']`` (default ``"diffusion"`` for
    backward compatibility with existing world_model checkpoints).
    """
    dynamics_type = config.get("dynamics_type", "diffusion")
    action_dim = config.get("action_dim", 2)

    if dynamics_type == "mlp":
        from wm_core.dynamics.mlp_dynamics import MLPDynamics

        return MLPDynamics(
            latent_dim=config["latent_dim"],
            action_dim=action_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

    from wm_core.dynamics.diffusion_dynamics import DiffusionDynamics

    return DiffusionDynamics(
        latent_dim=config["latent_dim"],
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_diffusion_steps=config.get("diffusion_steps", 50),
    )
