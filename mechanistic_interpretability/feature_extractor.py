#!/usr/bin/env python3
"""
Feature Extraction for UNet Intermediate Activations

Extracts intermediate activations from the UNet model at key layers
for mechanistic interpretability analysis.

Author: [Your Name]
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Allow imports from parent UNet directory
import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent))
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet"):
    sys.path.insert(0, str(_p))
from unet import UNetFPP


class UNetFeatureExtractor(nn.Module):
    """
    Wrapper around UNetFPP to extract intermediate activations.

    Extraction points:
        - Encoder: skip1, skip2, skip3, skip4 (after DoubleConv, before MaxPool)
        - Bottleneck: after bottleneck convolutions
        - Decoder: up1, up2, up3, up4 (after upsampling and concat)
        - Output: final depth prediction

    Args:
        pretrained_path: Path to pretrained checkpoint (optional)
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        pretrained_path: str = None,
        in_channels: int = 1,
        out_channels: int = 1
    ):
        super().__init__()

        self.model = UNetFPP(in_channels=in_channels, out_channels=out_channels)

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded pretrained UNet from {pretrained_path}")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass extracting intermediate activations.

        Args:
            x: Input fringe image (B, 1, H, W)

        Returns:
            Dictionary with activations at each layer:
                - skip1-4: Encoder outputs (B, C, H_i, W_i)
                - bottleneck: Bottleneck output (B, 1024, H/16, W/16)
                - up1-4: Decoder outputs (B, C, H_i, W_i)
                - output: Final depth prediction (B, 1, H, W)
        """
        activations = {}

        # Encoder
        skip1, x = self.model.down1(x)
        activations['skip1'] = skip1

        skip2, x = self.model.down2(x)
        activations['skip2'] = skip2

        skip3, x = self.model.down3(x)
        activations['skip3'] = skip3

        skip4, x = self.model.down4(x)
        activations['skip4'] = skip4

        # Bottleneck
        x = self.model.bottleneck(x)
        activations['bottleneck'] = x
        x = self.model.dropout(x)

        # Decoder
        x = self.model.up1(x, skip4)
        activations['up1'] = x

        x = self.model.up2(x, skip3)
        activations['up2'] = x

        x = self.model.up3(x, skip2)
        activations['up3'] = x

        x = self.model.up4(x, skip1)
        activations['up4'] = x

        # Output
        out = self.model.outconv(x)
        activations['output'] = out

        return activations

    def get_layer_info(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns information about each probing layer.

        Returns:
            Dictionary mapping layer name to (channels, downsample_factor)
        """
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
    """
    Apply global pooling to get fixed-size feature vectors.

    Args:
        activations: Dictionary of layer activations (B, C, H, W)
        pooling: 'avg' for global average pooling, 'max' for global max pooling

    Returns:
        Dictionary mapping layer name to feature vector (B, C)
    """
    pool_fn = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

    features = {}
    for name, act in activations.items():
        if name == 'output':
            continue
        pooled = pool_fn(act)
        features[name] = pooled.view(pooled.size(0), -1)

    return features


if __name__ == "__main__":
    print("Testing UNet Feature Extractor...")

    extractor = UNetFeatureExtractor(pretrained_path=None)

    print("\nLayer Information:")
    for name, (channels, downsample) in extractor.get_layer_info().items():
        print(f"  {name}: {channels} channels, 1/{downsample} resolution")

    x = torch.randn(2, 1, 256, 256)
    print(f"\nInput shape: {x.shape}")

    activations = extractor(x)
    print("\nActivation shapes:")
    for name, act in activations.items():
        print(f"  {name}: {act.shape}")

    global_features = extract_global_features(activations)
    print("\nGlobal features (after pooling):")
    for name, feat in global_features.items():
        print(f"  {name}: {feat.shape}")
