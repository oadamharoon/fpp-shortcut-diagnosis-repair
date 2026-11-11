"""
Phase-Intermediate UNet for FPP.

Same UNet backbone as baseline (encoder + decoder + skip connections) but
with the output head replaced. The network emits wrapped phase (as a
normalized sin/cos pair) and optionally fringe-order logits. Depth is
NOT produced by the network — it is computed downstream via the fixed
differentiable calibration layer.

This is the key architectural difference from PINN: PINN kept depth as
output and added physics to the loss. Phase-intermediate forces the
output to be phase, removing the shape-prior shortcut path.

Variants:
    A1: single phase head (sin, cos). Uses GT k at training time.
    A2: phase head + order classification head. Full single-shot pipeline.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse baseline building blocks
sys.path.insert(0, str(Path(__file__).parent.parent / "UNet"))
from unet import DoubleConv, DownSample, UpSample


class PhaseUNetFPP(nn.Module):
    """
    UNet with phase-intermediate output head.

    Architecture is identical to UNetFPP (baseline) except for the output
    layer. The encoder, decoder, skip connections, and final 64-channel
    feature map are unchanged.

    Args:
        variant: 'A1' or 'A2'
            A1: phase head only (2 channels for sin/cos)
            A2: phase head (2 ch) + order head (K_max channels)
        in_channels: input channels (1 for grayscale fringe)
        K_max: number of fringe order classes (32 for 5-bit gray code)
        dropout_rate: bottleneck dropout (0 to match baseline best)
    """

    def __init__(self, variant='A1', in_channels=1, K_max=32, dropout_rate=0.0, predict_ab=False):
        super().__init__()
        assert variant in ('A1', 'A2'), f"variant must be A1 or A2, got {variant}"
        self.variant = variant
        self.K_max = K_max

        # ENCODER (identical to baseline)
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # BOTTLENECK (identical to baseline)
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER (identical to baseline)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # OUTPUT HEADS (new — replaces baseline's Conv2d(64, 1, 1))
        # Phase head: 2 channels for (sin_phi_raw, cos_phi_raw)
        self.head_phase = nn.Conv2d(64, 2, kernel_size=1)
        # optional auxiliary head: per-pixel background A and modulation B,
        # supervised by the 18-frame DFT decomposition.  Off by default so
        # every existing script keeps its exact architecture and state_dict.
        self.predict_ab = predict_ab
        if predict_ab:
            self.head_ab = nn.Conv2d(64, 2, kernel_size=1)
        # Order head (A2 only): K_max channel logits for fringe order classification
        if variant == 'A2':
            self.head_order = nn.Conv2d(64, K_max, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input fringe image

        Returns:
            dict with:
                sin_phi_hat: (B, 1, H, W) normalized sin(wrapped phase)
                cos_phi_hat: (B, 1, H, W) normalized cos(wrapped phase)
                order_logits: (B, K_max, H, W) fringe order logits (A2 only)
        """
        # Encoder (same as baseline)
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)

        # Decoder (same as baseline)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Phase head: raw sin/cos then L2-normalize per-pixel
        phase_out = self.head_phase(x)  # (B, 2, H, W)
        sin_raw = phase_out[:, 0:1]  # (B, 1, H, W)
        cos_raw = phase_out[:, 1:2]  # (B, 1, H, W)

        # L2 normalize: enforce sin^2 + cos^2 = 1
        # Prevents trivial minimization via scaling; clamps for gradient safety
        norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw + 1e-8)
        sin_phi_hat = sin_raw / norm
        cos_phi_hat = cos_raw / norm

        out = {
            'sin_phi_hat': sin_phi_hat,
            'cos_phi_hat': cos_phi_hat,
        }

        # Order head (A2 only)
        if self.variant == 'A2':
            out['order_logits'] = self.head_order(x)  # (B, K_max, H, W)
        if self.predict_ab:
            ab = self.head_ab(x)
            out['A_hat'] = ab[:, 0:1]
            out['B_hat'] = ab[:, 1:2]

        return out
