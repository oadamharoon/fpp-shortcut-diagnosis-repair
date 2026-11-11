"""
Differentiable depth-to-phase layer for PINN loss.

Given predicted depth (mm), back-projects to 3D, projects through
the projector matrix, and computes the expected wrapped phase.

This is the inverse direction of the calibration layer:
  CalibrationLayer:  phase → depth
  DepthToPhaseLayer: depth → phase

Used as a physics constraint: the predicted depth should produce
a phase consistent with the observed fringe pattern.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class DepthToPhaseLayer(nn.Module):
    """
    Fixed (non-learned) differentiable layer that converts depth (mm)
    to wrapped phase via camera back-projection and projector projection.

    depth_mm → 3D point → project through M_p → v_p → wrapped phase
    """

    def __init__(self, calibration_dir, image_size=(960, 960), T_p=36,
                 scale=1.446, phase_offset=2.9):
        super().__init__()

        cal_dir = Path(calibration_dir)

        # Load intrinsics
        A_c = np.loadtxt(cal_dir / "CamIntrinsicMatrix.txt")
        A_p = np.loadtxt(cal_dir / "ProjIntrinsicMatrix.txt")
        R_p = np.loadtxt(cal_dir / "ProjRotationMatrix.txt")
        T_p_vec = np.loadtxt(cal_dir / "ProjTranslationVector.txt").reshape(3, 1)

        M_p = A_p @ np.hstack([R_p, T_p_vec])

        # Camera intrinsics for back-projection
        self.register_buffer('fx', torch.tensor(A_c[0, 0], dtype=torch.float64))
        self.register_buffer('fy', torch.tensor(A_c[1, 1], dtype=torch.float64))
        self.register_buffer('cx', torch.tensor(A_c[0, 2], dtype=torch.float64))
        self.register_buffer('cy', torch.tensor(A_c[1, 2], dtype=torch.float64))

        # Projector matrix
        self.register_buffer('M_p', torch.from_numpy(M_p).double())

        # Constants
        self.register_buffer('T_p', torch.tensor(float(T_p), dtype=torch.float64))
        self.register_buffer('scale_factor', torch.tensor(float(scale), dtype=torch.float64))
        self.register_buffer('phase_offset_val', torch.tensor(float(phase_offset), dtype=torch.float64))

        # Pixel grids (1-based to match MATLAB)
        H, W = image_size
        row_grid, col_grid = torch.meshgrid(
            torch.arange(1, H + 1, dtype=torch.float64),
            torch.arange(1, W + 1, dtype=torch.float64),
            indexing='ij'
        )
        self.register_buffer('u_cam', col_grid.clone())
        self.register_buffer('v_cam', row_grid.clone())

    def forward(self, depth_mm):
        """
        Convert depth (mm) to absolute (unwrapped) phase.

        Args:
            depth_mm: (B, 1, H, W) depth in mm (after denormalization)

        Returns:
            abs_phase: (B, 1, H, W) absolute phase = 2*pi*v_p/T_p (radians,
                       monotonic in v_p, no wrapping)
        """
        # Work in float64 for precision
        z_stored = depth_mm.squeeze(1).double()

        # Undo the scale to get world z
        z_world = z_stored * self.scale_factor

        # Back-project camera pixels to 3D
        x_3d = (self.u_cam - self.cx) * z_world / self.fx
        y_3d = (self.v_cam - self.cy) * z_world / self.fy

        # Project through M_p to get v_p
        mp = self.M_p
        proj_v = mp[1, 0] * x_3d + mp[1, 1] * y_3d + mp[1, 2] * z_world + mp[1, 3]
        proj_w = mp[2, 0] * x_3d + mp[2, 1] * y_3d + mp[2, 2] * z_world + mp[2, 3]
        v_p = proj_v / (proj_w + 1e-12)

        # Absolute phase (NOT wrapped) — bijective in depth, no multi-modality
        # Subtract phase_offset to match the convention used to compute GT phase
        abs_phase = v_p * 2.0 * torch.pi / self.T_p - self.phase_offset_val

        return abs_phase.float().unsqueeze(1)
