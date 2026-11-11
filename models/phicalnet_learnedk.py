"""
PhiCalNetLearnedK — Variant 1 of the multi-frame ablation.

Learned-phase / learned-k PhiCalNet: the existing PhaseUNetFPP backbone
predicts (sin phi, cos phi) as before, and a FringeOrderHead grafted onto
the input side predicts the fringe-order k. Both heads are trained
simultaneously (phase loss + k cross-entropy + depth loss).

Works at any input frame count N >= 1 by passing `in_channels=N` to the
UNet's first conv and using N raw frames as part of the k-head input.

Trainable parameters:
    PhaseUNetFPP backbone + ConfigurableFringeOrderHead.

Everything else (calibration layer, soft argmax, unwrap formula) is
shared with PhiCalHybridNet.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from models.phase_unet import PhaseUNetFPP
from models.ftp_architecture import soft_argmax
from models.phicalnet_hybrid import (
    ConfigurableFringeOrderHead, WideFringeOrderHead,
)


class PhiCalNetLearnedK(nn.Module):
    """PhiCalNet backbone + learned k-head.

    Args:
        calibration_layer: CalibrationLayer instance (fixed, differentiable).
        in_channels: number of stacked input fringe frames (1 for single
            shot, 3 for the classical 3-step set, etc.).
        K_max: number of fringe-order classes.
        gumbel_temperature: softmax temperature on k_logits for the
            differentiable argmax.
        dropout_rate: bottleneck dropout on the backbone (default 0).
        k_head_variant: "default" (128-channel, 3 middle blocks, ~0.38M
            params) or "wide" (256-channel, 5 middle blocks, ~3.3M params)
            -- used for the capacity-ablation study.
    """

    def __init__(
        self,
        calibration_layer,
        in_channels: int = 1,
        K_max: int = 32,
        gumbel_temperature: float = 1.0,
        dropout_rate: float = 0.0,
        k_head_variant: str = "default",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.K_max = K_max
        self.gumbel_temperature = gumbel_temperature
        self.calibration = calibration_layer
        self.k_head_variant = k_head_variant

        # Phase backbone: A1 head (no built-in order head; we use our own).
        self.phase_net = PhaseUNetFPP(
            variant="A1",
            in_channels=self.in_channels,
            K_max=K_max,
            dropout_rate=dropout_rate,
        )

        # k-head input: (phi_hat, sin_hat, cos_hat, *input_frames)
        # = 3 + in_channels channels.
        k_in = 3 + self.in_channels
        if k_head_variant == "default":
            self.k_head = ConfigurableFringeOrderHead(
                in_channels=k_in, K_max=K_max
            )
        elif k_head_variant == "wide":
            self.k_head = WideFringeOrderHead(in_channels=k_in, K_max=K_max)
        else:
            raise ValueError(
                f"Unknown k_head_variant={k_head_variant!r}; "
                "expected 'default' or 'wide'."
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, in_channels, H, W) stacked fringe frames.
        Returns:
            dict containing phi_hat, sin/cos_phi_hat, k_logits, k_soft,
            Phi_hat, depth_hat.
        """
        phase_out = self.phase_net(x)
        sin_hat = phase_out["sin_phi_hat"]
        cos_hat = phase_out["cos_phi_hat"]
        phi_hat = torch.atan2(sin_hat, cos_hat)  # (B, 1, H, W), (-pi, pi]

        k_input = torch.cat([phi_hat, sin_hat, cos_hat, x], dim=1)
        k_logits = self.k_head(k_input)
        k_logits = torch.nan_to_num(k_logits, nan=0.0)

        k_soft = soft_argmax(k_logits, temperature=self.gumbel_temperature)
        Phi_hat = phi_hat + 2.0 * math.pi * k_soft
        depth_hat = self.calibration(Phi_hat)

        return {
            "phi_hat": phi_hat,
            "sin_phi_hat": sin_hat,
            "cos_phi_hat": cos_hat,
            "k_logits": k_logits,
            "k_soft": k_soft,
            "Phi_hat": Phi_hat,
            "depth_hat": depth_hat,
        }
