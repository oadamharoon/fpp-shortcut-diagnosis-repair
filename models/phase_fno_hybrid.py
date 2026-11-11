"""
PhaseFNOHybrid: a spectral-local operator matched to the FPP inductive bias.

A plain FNO underperforms a UNet for phase-intermediate FPP because its mode-truncated
spectral convolution (a low-pass operator) cannot represent the HIGH spatial frequency
of the wrapped-phase output -- sin(phi)/cos(phi) oscillate once per fringe, far above
the retained modes -- and its only spatial mixing is 1x1 (point-wise), so it has no
local receptive field. FTP's own logic confirms the mismatch: the phase information
lives around the high-frequency carrier, not in the lowest modes.

This model keeps the global spectral path (useful for the carrier/demodulation and
global phase consistency) but adds a LOCAL convolutional branch to every block and to
the output head, restoring the high-frequency, locally-mixed capacity that the phase
field needs. The two branches are summed inside each block:

    block(x) = GELU( SpectralConv_lowmode(x) + LocalConv_3x3x3x3(x) )

Parameters stay near the UNet/FNO (~33M, dominated by the spectral weights), so any
accuracy change is attributable to the added local capacity, not to more parameters.
Drop-in I/O contract identical to PhaseUNetFPP / PhaseFNO_FPP.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.phase_fno import SpectralConv2d


class HybridBlock(nn.Module):
    """SpectralConv (global, low-frequency) + a local 3x3->3x3 conv branch
    (local, high-frequency), summed, then GELU (except the last block)."""

    def __init__(self, width: int, modes1: int, modes2: int, activate: bool = True):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.local(x)
        return F.gelu(y) if self.activate else y


class PhaseFNOHybrid(nn.Module):
    """Spectral-local hybrid operator with the PhiCalNet phase head.

    Args mirror PhaseFNO_FPP; K_max is kept only for interface parity."""

    def __init__(self, in_channels: int = 1, width: int = 64,
                 modes: int = 32, n_blocks: int = 4, K_max: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.modes = modes
        self.n_blocks = n_blocks
        self.K_max = K_max

        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.blocks = nn.ModuleList([
            HybridBlock(width, modes, modes, activate=(i < n_blocks - 1))
            for i in range(n_blocks)
        ])
        # projection head with a local 3x3 conv so the OUTPUT can carry high frequency
        self.proj1 = nn.Conv2d(width, 128, 3, padding=1)
        self.proj2 = nn.Conv2d(128, 2, 1)

        self._initialize_weights()
        self._cached_grid = None

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _positional_grid(self, B, H, W, device, dtype):
        if (self._cached_grid is not None and self._cached_grid.shape[2:] == (H, W)
                and self._cached_grid.device == device):
            grid = self._cached_grid
        else:
            ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
            xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
            yg, xg = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([yg, xg], dim=0).unsqueeze(0)
            self._cached_grid = grid
        return grid.expand(B, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        pos = self._positional_grid(B, H, W, x.device, x.dtype)
        x = torch.cat([x, pos], dim=1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.proj1(x))
        out = self.proj2(x)

        sin_raw, cos_raw = out[:, 0:1], out[:, 1:2]
        norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw + 1e-8)
        return {"sin_phi_hat": sin_raw / norm, "cos_phi_hat": cos_raw / norm}
