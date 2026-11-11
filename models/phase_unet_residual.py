"""
Phase-intermediate UNet with residual depth correction head.

Same backbone as A1, but adds a second 1x1 output head that predicts a
per-pixel depth correction (in mm). Final depth = depth_from_phase + residual.

The network learns phase as the primary signal; the residual head only
activates where phase-based reconstruction has systematic errors (fringe
wrap boundaries, high-gradient regions, calibration artifacts).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "UNet"))
from unet import DoubleConv, DownSample, UpSample


class PhaseUNetResidual(nn.Module):
    """
    Phase-intermediate UNet with residual depth correction.

    Output heads:
      head_phase: 2 channels (sin, cos) -> normalized to unit circle
      head_residual: 1 channel (depth correction in mm, initialized near 0)
    """

    def __init__(self, in_channels=1, dropout_rate=0.0, residual_scale=50.0):
        """
        Args:
            residual_scale: output of residual head is multiplied by tanh(raw)*residual_scale,
                bounding correction to +/- residual_scale mm. Prevents runaway corrections.
        """
        super().__init__()
        self.residual_scale = residual_scale

        # Encoder (same as baseline)
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Decoder (same as baseline)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # Output heads
        self.head_phase = nn.Conv2d(64, 2, kernel_size=1)
        self.head_residual = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        # Initialize residual head bias to 0 so initial correction is 0
        nn.init.zeros_(self.head_residual.weight)
        nn.init.zeros_(self.head_residual.bias)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Phase head: raw sin/cos, then L2 normalize
        phase_out = self.head_phase(x)
        sin_raw = phase_out[:, 0:1]
        cos_raw = phase_out[:, 1:2]
        norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw + 1e-8)
        sin_phi_hat = sin_raw / norm
        cos_phi_hat = cos_raw / norm

        # Residual head: bounded in [-residual_scale, +residual_scale] via tanh
        residual_raw = self.head_residual(x)
        residual_mm = torch.tanh(residual_raw) * self.residual_scale

        return {
            'sin_phi_hat': sin_phi_hat,
            'cos_phi_hat': cos_phi_hat,
            'residual_mm': residual_mm,
        }
