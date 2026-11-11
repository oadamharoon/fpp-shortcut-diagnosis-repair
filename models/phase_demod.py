"""
DemodPhaseNet: a phase-recovery network whose information flow IS the physics of
Fourier-transform profilometry, learnable end to end.

Design principle. A fringe image is I = a + b*cos(2*pi*f0.x + phi). The phase lives
around the spatial CARRIER f0. FTP recovers it by (1) band-passing the analytic lobe
around f0, (2) demodulating (shifting f0 to baseband) to get b*exp(i*phi), and
(3) reading phi = angle(.). We make that pipeline the network skeleton and learn only
its physical knobs plus a small LOCAL refinement:

    learnable carriers f_k  --band-pass-->  analytic_k  --demodulate-->  baseband_k
    stack{ Re, Im of baseband_k }  --local conv refinement-->  (sin phi, cos phi)

This is physics-constrained by construction, and it dissolves the UNet/FNO puzzle:
 - it demodulates first, so the signal the learned part sees is BASEBAND (low
   frequency) -- exactly where local convs (or even a spectral operator) are apt;
 - the carrier band-pass and quadrature demodulation, the operations a plain FNO
   lacks and a UNet has to learn implicitly, are baked in and start from the physics.

The demodulation (FFT, complex window, IFFT, heterodyne) is fully differentiable, so
the carrier frequencies and band-pass widths train by backprop. Parameter count is a
small fraction of the UNet because the physics does the heavy lifting.

Drop-in I/O contract identical to PhaseUNetFPP.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# carriers observed in the benchmark: oriented, |f| ~ 0.05-0.07 cyc/px, varying with depth
CARRIER_INIT = [(0.0, 0.060), (0.0, 0.050), (0.0, 0.070), (0.060, 0.0)]


class DemodPhaseNet(nn.Module):
    def __init__(self, in_channels: int = 1, carriers=CARRIER_INIT, bw_init: float = 0.02,
                 w: int = 64, K_max: int = 32, learn_carrier: bool = False):
        super().__init__()
        self.K_max = K_max
        c = torch.tensor(carriers, dtype=torch.float32)                # (K,2) cyc/px
        b = torch.full((len(carriers),), float(np.log(bw_init)))
        # The loss is hyper-sensitive to the carrier frequency (a small error becomes a
        # full-cycle ramp that decorrelates the output), which destabilizes training. We
        # therefore FIX the carriers as physical buffers by default and let only the
        # refinement learn; the per-scene carrier variation is absorbed by the carrier
        # bank plus the refinement's low-frequency correction.
        if learn_carrier:
            self.carrier = nn.Parameter(c); self.log_bw = nn.Parameter(b)
        else:
            self.register_buffer("carrier", c); self.register_buffer("log_bw", b)
        # Per carrier we expose physically meaningful features: the normalized quadrature
        # (cos phi_k, sin phi_k) -- a scale-free per-carrier phase estimate in the same
        # form as the output -- and the response magnitude |bb_k| -- a confidence that says
        # how well that carrier matches the local fringe. The refinement learns to fuse the
        # per-carrier estimates by confidence (soft carrier selection), the way multi-
        # frequency FTP does. The raw fringe is appended so ML can recover what the fixed
        # demodulation discards (albedo, fine detail).
        C = 3 * len(carriers) + in_channels                     # [cos, sin, mag] per carrier + raw

        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GELU())
        # small 2-scale U-Net refinement -- takes what works from UNet (local multi-scale
        # feature extraction) at a fraction of its size, since demodulation already did
        # the global carrier work an FNO could not.
        self.e0 = blk(C, w); self.e1 = blk(w, 2 * w); self.e2 = blk(2 * w, 4 * w)
        self.d1 = blk(4 * w + 2 * w, 2 * w); self.d0 = blk(2 * w + w, w)
        self.head = nn.Conv2d(w, 2, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _refine(self, z):
        x0 = self.e0(z)
        x1 = self.e1(self.pool(x0))
        x2 = self.e2(self.pool(x1))
        y1 = self.d1(torch.cat([self.up(x2), x1], 1))
        y0 = self.d0(torch.cat([self.up(y1), x0], 1))
        return self.head(y0)

    def _freq_space(self, H, W, device, dtype):
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        yy = torch.arange(H, device=device, dtype=dtype)
        xx = torch.arange(W, device=device, dtype=dtype)
        YY, XX = torch.meshgrid(yy, xx, indexing="ij")
        return FX, FY, XX, YY

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        FX, FY, XX, YY = self._freq_space(H, W, x.device, torch.float32)
        I_ft = torch.fft.fft2(x.float())                        # (B,1,H,W) complex, non-shifted
        feats = []
        for k in range(self.carrier.shape[0]):
            cx, cy = self.carrier[k, 0], self.carrier[k, 1]
            bw = torch.exp(self.log_bw[k]).clamp(1e-3, 0.5)
            win = torch.exp(-((FX - cx) ** 2 + (FY - cy) ** 2) / (2 * bw ** 2))     # analytic lobe
            analytic = torch.fft.ifft2(I_ft * win)              # complex baseband-around-carrier
            ramp = 2 * np.pi * (cx * XX + cy * YY)              # heterodyne to baseband
            bb = analytic * torch.exp(-1j * ramp)               # (B,1,H,W) complex ~ b exp(i psi)
            mag = torch.sqrt(bb.real ** 2 + bb.imag ** 2 + 1e-8)
            feats.append(bb.real / mag)                         # cos psi_k (scale-free deviation)
            feats.append(bb.imag / mag)                         # sin psi_k
            feats.append(mag / (mag.mean(dim=(-2, -1), keepdim=True) + 1e-8))   # confidence
        z = torch.cat(feats + [x.float()], dim=1)              # per-carrier estimates + confidence + raw
        # the refinement predicts only the LOW-frequency deviation quadrature exp(i psi):
        u = self._refine(z)                                     # (B, 2, H, W) = (u_re, u_im)
        u_re, u_im = u[:, 0:1], u[:, 1:2]
        # physics RE-MODULATES with the primary carrier to synthesize the high-frequency
        # output sin/cos(Phi = carrier.x + psi). The network never generates the carrier.
        cx0, cy0 = self.carrier[0, 0], self.carrier[0, 1]
        ramp0 = 2 * np.pi * (cx0 * XX + cy0 * YY)
        cr, sr = torch.cos(ramp0), torch.sin(ramp0)
        out_re = u_re * cr - u_im * sr                          # Re(u * e^{i ramp0}) = |u| cos(Phi)
        out_im = u_re * sr + u_im * cr                          # Im(...)            = |u| sin(Phi)
        norm = torch.sqrt(out_re ** 2 + out_im ** 2 + 1e-8)
        return {"sin_phi_hat": out_im / norm, "cos_phi_hat": out_re / norm}
