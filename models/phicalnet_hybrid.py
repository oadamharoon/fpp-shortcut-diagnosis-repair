"""
PhiCalHybridNet — Variant 2 of the multi-frame ablation.

Hybrid analytical-phase / learned-k architecture:
  1. Analytical wrapped phase from N >= 3 phase-shifted frames via the
     closed-form N-step atan2 formula (no learnable parameters).
  2. Learned k-head predicts fringe order from (phi_hat, raw frames)
     using the same FringeOrderHead module as the existing FTP pipeline.
  3. Fixed unwrap:  Phi_hat = phi_hat + 2 pi k_hat.
  4. Fixed differentiable calibration layer -> depth.

Only the k-head has trainable parameters. The network's entire job is to
resolve the fringe-order ambiguity that analytical phase shifting cannot
handle on its own.

Phase shifts follow the generator convention used for the raw dataset:
    delta_i = 2 pi / 18 * (i - 2) + pi,  i = frame_index + 1  (1-indexed)
The default frame selection {F0, F6, F12} produces shifts {8pi/9, 14pi/9,
20pi/9}, pairwise 120 degrees apart, which is a valid 3-step set.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# Reuse the k-head + soft argmax from the FTP architecture to keep the
# learned component identical across variants.
from models.ftp_architecture import FringeOrderHead, soft_argmax


N_GENERATOR_STEPS = 18  # matches ComputeNStepPhase.m for the source dataset
DEFAULT_FRAMES = (0, 6, 12)


def generator_delta(frame_index: int, N: int = N_GENERATOR_STEPS) -> float:
    """Phase shift (radians) rendered into frame `frame_index` (0-indexed)
    by the dataset's N-step generator. 1-indexed i = frame_index + 1, and
    delta_i = 2*pi/N*(i-2) + pi."""
    i = frame_index + 1
    return 2.0 * math.pi / N * (i - 2) + math.pi


class ConfigurableFringeOrderHead(nn.Module):
    """FringeOrderHead with a configurable input-channel count.

    The original FringeOrderHead hardcodes 4 input channels (phi_hat,
    Re(c), Im(c), I). The hybrid model passes (phi_hat, F_0, ..., F_{N-1}),
    so we parameterize the first conv. Layer sizes and norm/activation
    choices match the FTP version byte-for-byte below the first conv so
    optimizer behaviour is the same.
    """

    def __init__(self, in_channels: int, K_max: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, K_max, 1),
        )

    def forward(self, x):
        return self.conv(x)


class WideFringeOrderHead(nn.Module):
    """Wider + deeper FringeOrderHead for the capacity-ablation study.

    Compared to ConfigurableFringeOrderHead:
      * channel width doubled (128 -> 256)
      * two extra intermediate 3x3 conv blocks (3 -> 5)

    Ends up at ~3.3 M head parameters (vs. ~0.38 M for the default head),
    a ~9x capacity increase. Input/output interface is identical so the
    rest of the forward graph is unchanged.
    """

    def __init__(self, in_channels: int, K_max: int = 32,
                 width: int = 256, n_middle: int = 5):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.InstanceNorm2d(width, affine=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_middle):
            layers.extend([
                nn.Conv2d(width, width, 3, padding=1),
                nn.InstanceNorm2d(width, affine=True),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.InstanceNorm2d(width // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, K_max, 1),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class PhiCalHybridNet(nn.Module):
    """Analytical-phase / learned-k PhiCalNet.

    Args:
        calibration_layer: an instance of CalibrationLayer (already built).
        frame_indices: 0-indexed frame numbers in the source 18-step grid
            that will be stacked as input channels. Must be at least 3
            and spaced so the corresponding delta_i values are linearly
            independent (equivalently: equally spaced modulo 2*pi).
        K_max: number of fringe-order classes.
        gumbel_temperature: softmax temperature used by the differentiable
            argmax on k_logits.
        N_steps: total steps of the generator that produced the raw dataset
            (18 in our case).
    """

    def __init__(
        self,
        calibration_layer,
        frame_indices: Sequence[int] = DEFAULT_FRAMES,
        K_max: int = 32,
        gumbel_temperature: float = 1.0,
        N_steps: int = N_GENERATOR_STEPS,
    ):
        super().__init__()
        if len(frame_indices) < 3:
            raise ValueError(
                "PhiCalHybridNet requires at least 3 phase-shifted frames "
                "for analytical wrapped-phase recovery; got "
                f"{len(frame_indices)}."
            )

        self.frame_indices = tuple(int(i) for i in frame_indices)
        self.K_max = K_max
        self.gumbel_temperature = gumbel_temperature
        self.calibration = calibration_layer

        # Precompute the phase-shift deltas for each stacked channel.
        deltas = torch.tensor(
            [generator_delta(i, N=N_steps) for i in self.frame_indices],
            dtype=torch.float32,
        )
        self.register_buffer("sin_deltas", torch.sin(deltas))
        self.register_buffer("cos_deltas", torch.cos(deltas))

        # k-head input: phi_hat + N raw frames = 1 + N channels.
        self.k_head = ConfigurableFringeOrderHead(
            in_channels=1 + len(self.frame_indices), K_max=K_max
        )

    def analytical_wrapped_phase(
        self, frames_stacked: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form N-step wrapped phase from N phase-shifted frames.

        Args:
            frames_stacked: (B, N, H, W)
        Returns:
            phi_hat: (B, 1, H, W) in (-pi, pi]
        """
        B, N, _, _ = frames_stacked.shape
        assert N == self.sin_deltas.numel(), (
            f"Input channel count ({N}) does not match model's configured "
            f"frame count ({self.sin_deltas.numel()})."
        )
        s_d = self.sin_deltas.view(1, N, 1, 1)
        c_d = self.cos_deltas.view(1, N, 1, 1)
        S = (frames_stacked * s_d).sum(dim=1, keepdim=True)
        C = (frames_stacked * c_d).sum(dim=1, keepdim=True)
        # Minus sign matches ComputeNStepPhase.m and the rest of the repo.
        return -torch.atan2(S, C)

    def forward(self, frames_stacked: torch.Tensor) -> dict:
        """Full forward pass.

        Args:
            frames_stacked: (B, N, H, W) phase-shifted fringe frames, with
                channel order matching `frame_indices`.
        Returns:
            dict with:
                phi_hat:   (B, 1, H, W) analytical wrapped phase.
                k_logits:  (B, K_max, H, W) per-pixel fringe-order logits.
                k_soft:    (B, 1, H, W) differentiable fringe-order estimate.
                Phi_hat:   (B, 1, H, W) unwrapped phase (pre-calibration).
                depth_hat: (B, 1, H, W) predicted depth in mm.
                sin_phi_hat / cos_phi_hat: convenient aliases so the
                    downstream evaluation/visualization code that expects
                    (sin, cos) outputs (as in PhiCalNet) still works.
        """
        phi_hat = self.analytical_wrapped_phase(frames_stacked)

        k_input = torch.cat([phi_hat, frames_stacked], dim=1)  # (B, 1+N, H, W)
        k_logits = self.k_head(k_input)
        # Safety against rare NaNs on GPU, mirroring FTPUNet.
        k_logits = torch.nan_to_num(k_logits, nan=0.0)

        k_soft = soft_argmax(k_logits, temperature=self.gumbel_temperature)
        Phi_hat = phi_hat + 2.0 * math.pi * k_soft
        depth_hat = self.calibration(Phi_hat)

        return {
            "phi_hat": phi_hat,
            "k_logits": k_logits,
            "k_soft": k_soft,
            "Phi_hat": Phi_hat,
            "depth_hat": depth_hat,
            "sin_phi_hat": torch.sin(phi_hat),
            "cos_phi_hat": torch.cos(phi_hat),
        }
