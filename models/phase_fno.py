"""
PhaseFNO_FPP: Fourier Neural Operator backbone for the phase-intermediate
PhiCalNet pipeline.

Drop-in replacement for PhaseUNetFPP. Same I/O contract:
    Input:  (B, in_channels, H, W) fringe image(s)
    Output: dict with 'sin_phi_hat' and 'cos_phi_hat' (each (B, 1, H, W)),
            both L2-normalized so sin^2 + cos^2 = 1 per pixel.

Why FNO here? Phase recovery from a sinusoidal fringe pattern is a
Fourier-domain operation by physics: classical FTP literally takes the
FFT, applies a band-pass around the fringe carrier, IFFTs, and reads
phase via atan2. An FNO generalizes that by learning richer per-mode
complex weights and stacking multiple Fourier-mixing layers, while
keeping the right inductive bias (spectral convolutions instead of
purely spatial convolutions).

Architecture follows Li et al. 2020 (Fourier Neural Operator for
Parametric PDEs):
    1. Lift: 1x1 conv from (in_channels + 2 positional channels) -> width.
    2. K spectral blocks: each is SpectralConv2d (mode-truncated FFT
       with learnable complex weights) added to a 1x1 spatial conv,
       followed by GELU. The final block omits the activation.
    3. Project: 1x1 conv stack from width -> 128 -> 2 (raw sin/cos).
    4. L2-normalize the (sin, cos) pair per pixel.

Defaults (width=64, modes=32, 4 blocks) put the parameter count near
the existing PhiCalNet UNet backbone (~31M), so any accuracy difference
is attributable to inductive bias rather than capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution: mode-truncated FFT with learnable
    complex weights.

    For an input of shape (B, C_in, H, W):
        - rfft2 gives (B, C_in, H, W//2+1) complex tensor.
        - We keep only the lowest `modes1` rows and lowest `modes2` cols
          (and the symmetric high-frequency rows due to real-input rfft).
        - Multiply by learnable complex weights (in_ch -> out_ch).
        - irfft2 back to (B, C_out, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # Two weight banks: one for the low-frequency rows (positive),
        # one for the symmetric high-frequency rows (negative wrap).
        self.weight_lo = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat)
        )
        self.weight_hi = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, m1, m2)  w: (in_ch, out_ch, m1, m2)
        # -> (B, out_ch, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # rfft2 over last two dims; output shape (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)
        # Low-frequency block (rows 0..m1, cols 0..m2)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight_lo[:, :, :m1, :m2]
        )
        # High-frequency block (last m1 rows, cols 0..m2)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weight_hi[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNOBlock(nn.Module):
    """One FNO block: SpectralConv + parallel 1x1 spatial conv, residual sum,
    optional GELU."""
    def __init__(self, width: int, modes1: int, modes2: int,
                 activate: bool = True):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.spat = nn.Conv2d(width, width, 1)
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.spat(x)
        if self.activate:
            y = F.gelu(y)
        return y


class PhaseFNO_FPP(nn.Module):
    """Fourier Neural Operator backbone with the PhiCalNet phase head.

    Args:
        in_channels: number of stacked input fringe frames (1 for single
            shot, 3 for the 3-step set, etc.).
        width: channel width of the FNO interior.
        modes: number of Fourier modes to keep along each axis.
        n_blocks: number of FNO blocks (each is SpectralConv + 1x1 conv +
            GELU, with the last block omitting activation).
        K_max: kept for interface compatibility with PhaseUNetFPP; not
            used by the FNO directly.
    """

    def __init__(self, in_channels: int = 1, width: int = 64,
                 modes: int = 32, n_blocks: int = 4, K_max: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.modes = modes
        self.n_blocks = n_blocks
        self.K_max = K_max  # unused; here for parity with PhaseUNetFPP

        # Lift: input + 2 positional channels -> width
        self.lift = nn.Conv2d(in_channels + 2, width, 1)

        # FNO blocks (last block: no activation, matching Li et al. 2020)
        self.blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, activate=(i < n_blocks - 1))
            for i in range(n_blocks)
        ])

        # Project to (sin_raw, cos_raw)
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, 2, 1)

        self._initialize_weights()
        self._cached_grid = None

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _positional_grid(self, B: int, H: int, W: int,
                         device, dtype) -> torch.Tensor:
        """Return (B, 2, H, W) positional channels in [0, 1]. Cached so we
        don't allocate every forward call at the same resolution."""
        if (self._cached_grid is not None
                and self._cached_grid.shape[2:] == (H, W)
                and self._cached_grid.device == device):
            grid = self._cached_grid
        else:
            ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
            xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
            yg, xg = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([yg, xg], dim=0).unsqueeze(0)  # (1, 2, H, W)
            self._cached_grid = grid
        return grid.expand(B, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        pos = self._positional_grid(B, H, W, x.device, x.dtype)
        x = torch.cat([x, pos], dim=1)            # (B, in_ch+2, H, W)
        x = self.lift(x)                          # (B, width,    H, W)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))                 # (B, 128, H, W)
        out = self.proj2(x)                       # (B, 2,   H, W)

        sin_raw, cos_raw = out[:, 0:1], out[:, 1:2]
        norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw + 1e-8)
        sin_phi_hat = sin_raw / norm
        cos_phi_hat = cos_raw / norm

        return {
            "sin_phi_hat": sin_phi_hat,
            "cos_phi_hat": cos_phi_hat,
        }
