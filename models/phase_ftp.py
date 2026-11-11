"""
PhaseFTP_FPP: physics-constrained FTP-as-Architecture phase backbone
for the PhiCalNet pipeline.

Reuses the phase-extraction component of models/ftp_architecture.py:
    DC removal (small UNet)
    -> Hann window (fixed)
    -> 2D FFT (fixed)
    -> learnable bandpass filter
    -> 2D IFFT (fixed)
    -> atan2 -> wrapped phase

Compared to the existing FTPUNet in ftp_architecture.py, this class:
  - Drops the FringeOrderHead (we use oracle k like everywhere else in
    the paper).
  - Drops the internal CalibrationLayer (handled externally by the
    trainer / eval pipeline, matching PhiCalNet's interface).
  - Outputs (sin_phi_hat, cos_phi_hat) in the same dict format as
    PhaseUNetFPP / PhaseFNO_FPP, so it can swap in directly.

This is a true physics-constrained operator network: the FFT/IFFT
structure is fixed (we KNOW the right operator for FTP), and only the
DC-removal subnet (~120k params) and the bandpass filter (4 params for
parametric / ~921k for flexible) are learned. Total trainable params
~120k-1M, vs ~31M for the UNet baseline. If accuracy is comparable, the
physics constraint is doing real work.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.ftp_architecture import (
    DCRemovalNet, ParametricGaussianFilter, FlexibleFilter,
)


class PhaseFTP_FPP(nn.Module):
    """Physics-constrained FTP backbone with PhiCalNet-compatible output.

    Args:
        in_channels: input channel count. Must be 1 -- the FFT path
            assumes a single-frame fringe image. (For multi-frame inputs,
            you'd average channels first or build a multi-frame variant
            separately.)
        H, W: input height/width (used to precompute Hann window and
            filter grids; default 960x960 to match the dataset).
        filter_type: 'parametric' (learnable Gaussian; ~4 params) or
            'flexible' (free HxW logits; ~921k params).
        fu0, fv0: initial Fourier-bin offsets for the bandpass filter
            (centered, fftshift convention). Defaults match the existing
            FTPUNet's empirically-tuned values for this dataset's
            horizontal-fringe carrier.
        K_max: kept for parity with PhaseUNetFPP / PhaseFNO_FPP; unused.
    """

    def __init__(self, in_channels: int = 1, H: int = 960, W: int = 960,
                 filter_type: str = "parametric",
                 fu0: float = -21, fv0: float = 0,
                 K_max: int = 32):
        super().__init__()
        if in_channels != 1:
            raise NotImplementedError(
                "PhaseFTP_FPP currently assumes a 1-channel fringe input "
                "(the FFT path expects a single image)."
            )
        self.H = H
        self.W = W
        self.K_max = K_max
        self.filter_type = filter_type

        # Step 1: DC removal subnet (small UNet, ~120k params)
        self.pre_processor = DCRemovalNet()

        # Step 2: Hann window (fixed buffer)
        hann_h = 0.5 * (1 - torch.cos(
            2 * math.pi * torch.arange(H, dtype=torch.float32) / (H - 1)))
        hann_w = 0.5 * (1 - torch.cos(
            2 * math.pi * torch.arange(W, dtype=torch.float32) / (W - 1)))
        self.register_buffer(
            "hann_window", hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        )

        # Step 4: Learnable band-pass filter (steps 3 and 5 are fixed FFT/IFFT)
        if filter_type == "parametric":
            self.bp_filter = ParametricGaussianFilter(H, W, fu0, fv0)
        elif filter_type == "flexible":
            self.bp_filter = FlexibleFilter(H, W, fu0, fv0)
        else:
            raise ValueError(f"Unknown filter_type: {filter_type!r}")

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, 1, H, W)

        # 1. DC removal
        I_prime = self.pre_processor(x)
        I_clean = x - I_prime

        # 2. Hann windowing
        I_windowed = I_clean * self.hann_window

        # 3. FFT (float64 for stability on GPU, matching existing FTPUNet)
        Iw = I_windowed.squeeze(1).double()
        F_complex = torch.fft.fft2(Iw)
        F_shifted = torch.fft.fftshift(F_complex, dim=(-2, -1))

        # 4. Learnable bandpass filter
        filt = self.bp_filter().double().unsqueeze(0)  # (1, H, W)
        F_filtered = F_shifted * filt

        # 5. IFFT
        F_unshifted = torch.fft.ifftshift(F_filtered, dim=(-2, -1))
        c = torch.fft.ifft2(F_unshifted)

        # 6. Wrapped phase via atan2 (back to float32, with NaN safety)
        c_real = torch.nan_to_num(c.real.float(), nan=0.0)
        c_imag = torch.nan_to_num(c.imag.float(), nan=0.0)
        phi_hat = torch.atan2(c_imag, c_real).unsqueeze(1)  # (B, 1, H, W)

        # 7. Convert to (sin, cos) so the interface matches PhaseUNetFPP
        # (both are unit-norm by construction; no L2 projection needed).
        sin_phi_hat = torch.sin(phi_hat)
        cos_phi_hat = torch.cos(phi_hat)

        return {
            "sin_phi_hat": sin_phi_hat,
            "cos_phi_hat": cos_phi_hat,
            # Extras (useful for diagnostics, not used by the loss/eval)
            "phi_hat": phi_hat,
            "I_prime_predicted": I_prime,
            "filter_snapshot": filt[0].real.float().detach(),
        }
