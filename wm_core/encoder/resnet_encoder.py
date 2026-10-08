"""Frozen ResNet18 encoder with lightweight adapter layers.

Only the adapter parameters are trained; the ResNet backbone is frozen
to stay within the 8GB VRAM training budget.

Two latent modes:
  spatial=False (legacy): global-average-pooled feature -> (B, latent_dim).
      Destroys spatial layout; kept only to load pre-existing checkpoints.
  spatial=True (default for new runs): keeps the conv feature map and flattens
      it -> (B, out_channels * grid * grid). Global average pooling averages
      away WHERE objects are, which is exactly the signal a world model that
      predicts object motion needs. Flattening preserves all of it.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """Frozen ResNet18 + trainable adapter.

    Args:
        adapter_dim: Hidden dimension of the adapter bottleneck.
        output_dim: For spatial=False, the final latent dim. For spatial=True,
            the number of output channels per spatial cell; the flat latent dim
            is ``output_dim * grid * grid`` (see ``latent_dim``).
        pretrained: Use ImageNet-pretrained weights.
        spatial: Keep the conv feature map instead of global-average-pooling.
        image_size: Input image size, used to infer the feature-map grid.
    """

    def __init__(
        self,
        adapter_dim: int = 64,
        output_dim: int = 64,
        pretrained: bool = True,
        spatial: bool = True,
        image_size: int = 84,
        obs_horizon: int = 1,
    ):
        super().__init__()
        self.spatial = spatial
        self.obs_horizon = obs_horizon
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        if spatial:
            # Drop avgpool AND fc -> keep the layer4 conv feature map.
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        else:
            # Legacy: drop only fc, keep AdaptiveAvgPool2d.
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        if spatial:
            # Infer the spatial grid of the backbone feature map.
            with torch.no_grad():
                probe = self.backbone(torch.zeros(1, 3, image_size, image_size))
            self._grid = int(probe.shape[-1])
            self.adapter = nn.Sequential(
                nn.Conv2d(512, adapter_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(adapter_dim, output_dim, kernel_size=1),
            )
            self._frame_dim = output_dim * self._grid * self._grid
        else:
            self._grid = 1
            self.adapter = nn.Sequential(
                nn.Linear(512, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, output_dim),
            )
            self._frame_dim = output_dim
        # k stacked frames are encoded independently and concatenated, so the
        # latent carries motion (velocity) -- a single frame cannot.
        self.latent_dim = self._frame_dim * obs_horizon

    @classmethod
    def from_config(cls, config: dict) -> "ResNetEncoder":
        """Build an encoder from a checkpoint ``config`` dict.

        Old checkpoints lack the ``spatial`` key and load via the pooled path.
        """
        spatial = bool(config.get("spatial", False))
        obs_horizon = int(config.get("obs_horizon", 1))
        if spatial:
            output_dim = config.get("spatial_out_channels", 16)
        else:
            # Legacy pooled: the stored latent_dim is the TOTAL, so divide the
            # horizon back out to recover the per-frame width.
            output_dim = config["latent_dim"] // obs_horizon
        return cls(
            adapter_dim=config["adapter_dim"],
            output_dim=output_dim,
            pretrained=config.get("pretrained", True),
            spatial=spatial,
            image_size=config.get("image_size", 84),
            obs_horizon=obs_horizon,
        )

    def train(self, mode: bool = True) -> "ResNetEncoder":
        # Keep the frozen backbone in eval so BatchNorm running stats stay fixed.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        if self.obs_horizon > 1:
            # (B, k*3, H, W): encode each frame with the frozen 3-channel
            # backbone, then concatenate -- keeps the ImageNet conv1 intact.
            image = image.reshape(b * self.obs_horizon, 3, *image.shape[-2:])
        features = self._extract_features(image)
        if self.spatial:
            lat = self.adapter(features).flatten(1)   # (B*k, C*grid*grid)
        else:
            lat = self.adapter(features.flatten(1))   # (B*k, C)
        if self.obs_horizon > 1:
            lat = lat.reshape(b, -1)                  # (B, k*frame_dim)
        return lat
