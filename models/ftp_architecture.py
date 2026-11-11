"""
FTP-as-Architecture for Single-Shot FPP.

Wires Fourier Transform Profilometry into the network as fixed
differentiable layers: DC removal -> windowing -> FFT -> learned
band-pass filter -> IFFT -> atan2 -> fringe order head -> absolute
phase -> calibration layer -> depth.

No depth-regression head: depth can only be produced by decoding phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# =============================================================================
# Shallow UNet for DC removal (pre-processor)
# =============================================================================

class ShallowConvBlock(nn.Module):
    """Two 3x3 convs with InstanceNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DCRemovalNet(nn.Module):
    """
    2-level shallow UNet to estimate background intensity I'(u,v).
    Output is subtracted from input to remove DC component before FFT.
    ~5x fewer params than the main UNet.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ShallowConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ShallowConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ShallowConvBlock(32, 64)

        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ShallowConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ShallowConvBlock(32, 16)

        # Output: single channel (estimated DC)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        # Decoder
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# =============================================================================
# Learned band-pass filters
# =============================================================================

class ParametricGaussianFilter(nn.Module):
    """
    2D Gaussian filter with learnable center and width.
    Applies to the +carrier sideband; mirror sideband is zeroed.
    """
    def __init__(self, H, W, fu0_init, fv0_init, sigma_u_init=None, sigma_v_init=None):
        super().__init__()
        # Learnable parameters (in frequency-bin units)
        self.fu0 = nn.Parameter(torch.tensor(float(fu0_init)))
        self.fv0 = nn.Parameter(torch.tensor(float(fv0_init)))
        self.log_sigma_u = nn.Parameter(torch.tensor(
            math.log(sigma_u_init if sigma_u_init else H * 0.1)
        ))
        self.log_sigma_v = nn.Parameter(torch.tensor(
            math.log(sigma_v_init if sigma_v_init else W * 0.1)
        ))
        self.H = H
        self.W = W

        # Precompute frequency grids (centered, matching fftshift)
        fu = torch.arange(H, dtype=torch.float32) - H // 2
        fv = torch.arange(W, dtype=torch.float32) - W // 2
        self.register_buffer('fu_grid', fu.unsqueeze(1).expand(H, W).clone())
        self.register_buffer('fv_grid', fv.unsqueeze(0).expand(H, W).clone())

    def forward(self):
        """Returns (H, W) filter mask."""
        # Clamp log_sigma to prevent overflow/underflow
        sigma_u = self.log_sigma_u.clamp(-2, 8).exp()
        sigma_v = self.log_sigma_v.clamp(-2, 8).exp()

        du = self.fu_grid - self.fu0
        dv = self.fv_grid - self.fv0
        # Clamp exponent to prevent exp overflow
        exponent = -0.5 * (du ** 2 / sigma_u ** 2 + dv ** 2 / sigma_v ** 2)
        gauss = torch.exp(exponent.clamp(min=-50))

        return gauss


class FlexibleFilter(nn.Module):
    """
    Free HxW learnable filter passed through sigmoid.
    Initialized to approximate a Gaussian at the expected carrier.
    """
    def __init__(self, H, W, fu0_init, fv0_init, sigma_u_init=None, sigma_v_init=None):
        super().__init__()
        sigma_u = sigma_u_init if sigma_u_init else H * 0.1
        sigma_v = sigma_v_init if sigma_v_init else W * 0.1

        # Initialize logits so sigmoid gives a Gaussian
        fu = torch.arange(H, dtype=torch.float32) - H // 2
        fv = torch.arange(W, dtype=torch.float32) - W // 2
        fu_grid = fu.unsqueeze(1).expand(H, W)
        fv_grid = fv.unsqueeze(0).expand(H, W)

        du = fu_grid - fu0_init
        dv = fv_grid - fv0_init
        gauss = torch.exp(-0.5 * (du ** 2 / sigma_u ** 2 + dv ** 2 / sigma_v ** 2))

        # Inverse sigmoid to initialize logits
        gauss_clamped = gauss.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(gauss_clamped / (1 - gauss_clamped))

        self.logits = nn.Parameter(logits)

    def forward(self):
        return torch.sigmoid(self.logits)


# =============================================================================
# Fringe order classification head
# =============================================================================

class FringeOrderHead(nn.Module):
    """
    Classifies fringe order k from phase features.
    Input: 4 channels (phi_hat, Re(c), Im(c), original I)
    Output: K_max channels of logits per pixel.
    """
    def __init__(self, K_max=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 128, 3, padding=1),
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


# =============================================================================
# Soft argmax with Gumbel-Softmax
# =============================================================================

def soft_argmax(logits, temperature=1.0):
    """
    Differentiable argmax using softmax with straight-through hard argmax.

    Uses soft probabilities for gradient but hard argmax for forward value.
    Avoids Gumbel noise which can cause NaN on GPU with large tensors.

    Args:
        logits: (B, K, H, W) class logits
        temperature: softmax temperature

    Returns:
        k_soft: (B, 1, H, W) fringe order (float)
    """
    B, K, H, W = logits.shape

    # Soft probabilities (for gradient)
    probs = F.softmax(logits / temperature, dim=1)  # (B, K, H, W)

    # Weighted sum = soft expected value
    indices = torch.arange(K, device=logits.device, dtype=torch.float32)
    indices = indices.view(1, K, 1, 1)
    k_soft = (probs * indices).sum(dim=1, keepdim=True)  # (B, 1, H, W)

    # Straight-through: use hard argmax in forward, soft gradient in backward
    k_hard = logits.argmax(dim=1, keepdim=True).float()  # (B, 1, H, W)
    k_out = k_hard + (k_soft - k_soft.detach())  # straight-through

    return k_out


# =============================================================================
# Main FTP Architecture
# =============================================================================

class FTPUNet(nn.Module):
    """
    FTP-as-Architecture: differentiable FPP pipeline.

    Forward pass:
      1. DC removal (learned shallow UNet)
      2. Hann windowing (fixed)
      3. FFT (fixed)
      4. Band-pass filter (learned)
      5. IFFT (fixed)
      6. Wrapped phase via atan2 (fixed)
      7. Fringe order classification (learned)
      8. Absolute phase via soft argmax (fixed)
      9. Calibration to depth (fixed)
    """

    def __init__(self, calibration_layer, filter_type='parametric',
                 fu0=-21, fv0=0, K_max=32, gumbel_temperature=1.0):
        """
        Args:
            calibration_layer: CalibrationLayer instance
            filter_type: 'parametric' or 'flexible'
            fu0, fv0: expected carrier frequency (frequency-bin offset from center)
            K_max: number of fringe order classes (32 for 5-bit effective range)
            gumbel_temperature: initial temperature for Gumbel-Softmax
        """
        super().__init__()
        H, W = 960, 960

        # Step 1: DC removal
        self.pre_processor = DCRemovalNet()

        # Step 2: Hann window (fixed buffer)
        hann_h = 0.5 * (1 - torch.cos(2 * torch.pi * torch.arange(H, dtype=torch.float32) / (H - 1)))
        hann_w = 0.5 * (1 - torch.cos(2 * torch.pi * torch.arange(W, dtype=torch.float32) / (W - 1)))
        hann_2d = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        self.register_buffer('hann_window', hann_2d)

        # Step 4: Band-pass filter
        self.filter_type = filter_type
        if filter_type == 'parametric':
            self.bp_filter = ParametricGaussianFilter(H, W, fu0, fv0)
        elif filter_type == 'flexible':
            self.bp_filter = FlexibleFilter(H, W, fu0, fv0)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Step 7: Fringe order head
        self.k_head = FringeOrderHead(K_max=K_max)
        self.K_max = K_max

        # Step 9: Calibration (fixed)
        self.calibration = calibration_layer

        # Temperature for Gumbel-Softmax
        self.gumbel_temperature = gumbel_temperature

    def forward(self, I):
        """
        Args:
            I: (B, 1, H, W) input fringe image

        Returns:
            dict with:
                phi_hat: (B, 1, H, W) wrapped phase estimate
                k_logits: (B, K_max, H, W) fringe order logits
                Phi_hat: (B, 1, H, W) absolute phase (before offset)
                depth_hat: (B, 1, H, W) reconstructed depth in mm
                filter_snapshot: (H, W) current filter mask
                I_prime_predicted: (B, 1, H, W) estimated DC
        """
        B = I.shape[0]

        # Step 1: DC removal
        I_prime = self.pre_processor(I)  # (B, 1, H, W)
        I_clean = I - I_prime

        # Step 2: Hann windowing
        I_windowed = I_clean * self.hann_window  # (B, 1, H, W)

        # Step 3: FFT (use float64 for numerical stability on GPU)
        I_windowed_d = I_windowed.squeeze(1).double()
        F_complex = torch.fft.fft2(I_windowed_d)  # (B, H, W) complex128
        F_shifted = torch.fft.fftshift(F_complex, dim=(-2, -1))

        # Step 4: Band-pass filter
        filter_mask = self.bp_filter()  # (H, W)
        F_filtered = F_shifted * filter_mask.double().unsqueeze(0)  # (B, H, W)

        # Step 5: IFFT
        F_unshifted = torch.fft.ifftshift(F_filtered, dim=(-2, -1))
        c = torch.fft.ifft2(F_unshifted)  # (B, H, W) complex128

        # Step 6: Wrapped phase via atan2 (back to float32)
        c_real = c.real.float()
        c_imag = c.imag.float()
        # Replace any NaN from FFT path
        c_real = torch.nan_to_num(c_real, nan=0.0)
        c_imag = torch.nan_to_num(c_imag, nan=0.0)
        phi_hat = torch.atan2(c_imag, c_real)  # (B, H, W), range (-pi, pi]
        phi_hat = phi_hat.unsqueeze(1)  # (B, 1, H, W)

        # Step 7: Fringe order classification
        k_input = torch.cat([
            phi_hat,
            c_real.unsqueeze(1),
            c_imag.unsqueeze(1),
            I,
        ], dim=1)  # (B, 4, H, W)

        k_logits = self.k_head(k_input)  # (B, K_max, H, W)
        k_logits = torch.nan_to_num(k_logits, nan=0.0)  # safety

        # Step 8: Absolute phase
        k_soft = soft_argmax(k_logits, temperature=self.gumbel_temperature)
        Phi_hat = phi_hat + 2 * torch.pi * k_soft  # (B, 1, H, W)

        # Step 9: Calibration to depth
        # CalibrationLayer adds phase_offset internally
        depth_hat = self.calibration(Phi_hat)  # (B, 1, H, W)

        return {
            'phi_hat': phi_hat,
            'k_logits': k_logits,
            'Phi_hat': Phi_hat,
            'depth_hat': depth_hat,
            'filter_snapshot': filter_mask.detach(),
            'I_prime_predicted': I_prime,
        }
