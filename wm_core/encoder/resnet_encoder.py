"""Frozen ResNet18 encoder with lightweight adapter layers.

Only the adapter parameters are trained; the ResNet backbone is frozen
to stay within the 8GB VRAM training budget.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """Frozen ResNet18 + trainable adapter.

    Args:
        adapter_dim: Hidden dimension of the adapter bottleneck.
        output_dim: Final latent dimension.
        pretrained: Use ImageNet-pretrained weights.
    """

    def __init__(self, adapter_dim: int = 64, output_dim: int = 64, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(512, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, output_dim),
        )

    @torch.no_grad()
    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).flatten(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(image)
        return self.adapter(features)
