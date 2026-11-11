#!/usr/bin/env python3
"""
Feature Extraction for PhicalNet (PhaseUNetFPP)

Author: [Your Name]
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Allow imports from UNet directory
import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent.parent))
_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet", _REPO / "mechanistic_interpretability"):
    sys.path.insert(0, str(_p))
from phase_unet import PhaseUNetFPP


class PhicalNetFeatureExtractor(nn.Module):
    """
    Wrapper around PhaseUNetFPP to extract intermediate activations.

    Args:
        pretrained_path: Path to pretrained checkpoint
        variant: PhaseUNetFPP variant ('A1', 'A2', etc.)
        K_max: Maximum fringe order
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        pretrained_path: str = None,
        variant: str = 'A1',
        K_max: int = 32,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.model = PhaseUNetFPP(
            variant=variant,
            in_channels=1,
            K_max=K_max,
            dropout_rate=dropout_rate
        )

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded pretrained PhicalNet from {pretrained_path}")

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate activations."""
        activations = {}

        skip1, x = self.model.down1(x)
        activations['skip1'] = skip1

        skip2, x = self.model.down2(x)
        activations['skip2'] = skip2

        skip3, x = self.model.down3(x)
        activations['skip3'] = skip3

        skip4, x = self.model.down4(x)
        activations['skip4'] = skip4

        x = self.model.bottleneck(x)
        activations['bottleneck'] = x
        x = self.model.dropout(x)

        x = self.model.up1(x, skip4)
        activations['up1'] = x

        x = self.model.up2(x, skip3)
        activations['up2'] = x

        x = self.model.up3(x, skip2)
        activations['up3'] = x

        x = self.model.up4(x, skip1)
        activations['up4'] = x

        phase_out = self.model.head_phase(x)
        sin_raw = phase_out[:, 0:1]
        cos_raw = phase_out[:, 1:2]
        norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw + 1e-8)
        activations['sin_phi_hat'] = sin_raw / norm
        activations['cos_phi_hat'] = cos_raw / norm

        return activations

    def get_layer_info(self) -> Dict[str, Tuple[int, int]]:
        """Returns (channels, downsample_factor) for each probing layer."""
        return {
            'skip1': (64, 1),
            'skip2': (128, 2),
            'skip3': (256, 4),
            'skip4': (512, 8),
            'bottleneck': (1024, 16),
            'up1': (512, 8),
            'up2': (256, 4),
            'up3': (128, 2),
            'up4': (64, 1),
        }


def extract_global_features(
    activations: Dict[str, torch.Tensor],
    pooling: str = 'avg'
) -> Dict[str, torch.Tensor]:
    """Apply global pooling to get fixed-size feature vectors."""
    pool_fn = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

    features = {}
    for name, act in activations.items():
        if name in ('sin_phi_hat', 'cos_phi_hat'):
            continue
        pooled = pool_fn(act)
        features[name] = pooled.view(pooled.size(0), -1)

    return features
