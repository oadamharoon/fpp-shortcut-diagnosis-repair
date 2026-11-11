"""
Differentiable calibration layer for FPP.

Converts absolute phase Phi to depth (mm) via triangulation using
pre-loaded camera and projector projection matrices.

Pipeline: Phi -> v_p -> triangulate (x,y,z) -> z/scale -> depth_mm

All matrices and constants are registered as non-trainable buffers.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class CalibrationLayer(nn.Module):
    """
    Fixed (non-learned) differentiable layer that converts absolute phase
    to depth in mm via the FPP triangulation equations.

    Buffers (non-trainable):
        M_c:  (3, 4) camera projection matrix
        M_p:  (3, 4) projector projection matrix
        u_cam: (H, W) camera column coordinates (1-based)
        v_cam: (H, W) camera row coordinates (1-based)
        T_p:  scalar, projector fringe period in pixels
        scale: scalar, proportional scaling factor
    """

    def __init__(self, calibration_dir, image_size=(960, 960), T_p=36,
                 scale=1.446, phase_offset=2.9):
        super().__init__()

        cal_dir = Path(calibration_dir)

        # Load calibration matrices
        A_c = np.loadtxt(cal_dir / "CamIntrinsicMatrix.txt")
        A_p = np.loadtxt(cal_dir / "ProjIntrinsicMatrix.txt")
        R_p = np.loadtxt(cal_dir / "ProjRotationMatrix.txt")
        T_p_vec = np.loadtxt(cal_dir / "ProjTranslationVector.txt").reshape(3, 1)

        R_c = np.eye(3)
        T_c = np.zeros((3, 1))

        M_c = A_c @ np.hstack([R_c, T_c])  # (3, 4)
        M_p = A_p @ np.hstack([R_p, T_p_vec])  # (3, 4)

        # Register as buffers (move to device with model, not trainable)
        self.register_buffer('M_c', torch.from_numpy(M_c).double())
        self.register_buffer('M_p', torch.from_numpy(M_p).double())
        self.register_buffer('T_p', torch.tensor(float(T_p), dtype=torch.float64))
        self.register_buffer('scale_factor', torch.tensor(float(scale), dtype=torch.float64))
        self.register_buffer('phase_offset', torch.tensor(float(phase_offset), dtype=torch.float64))

        # Precompute camera pixel coordinate grids (1-based, matching MATLAB)
        H, W = image_size
        row_grid, col_grid = torch.meshgrid(
            torch.arange(1, H + 1, dtype=torch.float64),
            torch.arange(1, W + 1, dtype=torch.float64),
            indexing='ij'
        )
        self.register_buffer('u_cam', col_grid.clone())  # (H, W)
        self.register_buffer('v_cam', row_grid.clone())   # (H, W)

    def forward(self, Phi):
        """
        Convert absolute phase to depth.

        Args:
            Phi: (B, 1, H, W) absolute phase (after unwrapping + offset)

        Returns:
            depth: (B, 1, H, W) depth in mm
        """
        B = Phi.shape[0]

        # Work in float64 for numerical precision in triangulation
        Phi_d = Phi.squeeze(1).double()  # (B, H, W)

        # Add phase offset (matches MATLAB: uphH = unwrPhase + 2.9)
        Phi_offset = Phi_d + self.phase_offset

        # Convert to projector coordinate: v_p = Phi * T_p / (2*pi)
        up = Phi_offset * self.T_p / (2.0 * torch.pi)  # (B, H, W)

        mc = self.M_c  # (3, 4)
        mp = self.M_p  # (3, 4)
        u = self.u_cam  # (H, W)
        v = self.v_cam  # (H, W)

        # Build A matrix rows and b vector (vectorized over batch and pixels)
        # Row 0: camera u equation
        A00 = mc[0, 0] - u * mc[2, 0]  # (H, W)
        A01 = mc[0, 1] - u * mc[2, 1]
        A02 = mc[0, 2] - u * mc[2, 2]

        # Row 1: camera v equation
        A10 = mc[1, 0] - v * mc[2, 0]
        A11 = mc[1, 1] - v * mc[2, 1]
        A12 = mc[1, 2] - v * mc[2, 2]

        # Row 2: projector v_p equation (depends on up, so batch-dependent)
        A20 = mp[1, 0] - up * mp[2, 0]  # (B, H, W)
        A21 = mp[1, 1] - up * mp[2, 1]
        A22 = mp[1, 2] - up * mp[2, 2]

        b0 = u * mc[2, 3] - mc[0, 3]  # (H, W)
        b1 = v * mc[2, 3] - mc[1, 3]
        b2 = up * mp[2, 3] - mp[1, 3]  # (B, H, W)

        # Cramer's rule for z (vectorized)
        det = (A00 * (A11 * A22 - A12 * A21)
             - A01 * (A10 * A22 - A12 * A20)
             + A02 * (A10 * A21 - A11 * A20))

        det_z = (A00 * (A11 * b2 - b1 * A21)
               - A01 * (A10 * b2 - b1 * A20)
               + b0 * (A10 * A21 - A11 * A20))

        # Avoid division by zero
        z = det_z / (det + 1e-12)

        # Apply scale factor
        depth = z / self.scale_factor

        # Clamp to reasonable range to prevent NaN from bad phase predictions
        # Valid depth range is ~1000-3000mm for this setup
        depth = depth.clamp(-5000.0, 5000.0)

        # Convert back to float32 and add channel dim
        return depth.float().unsqueeze(1)  # (B, 1, H, W)
