"""
Differentiable calibration layer for the VERTICAL fringe direction (u_p).

Clone of CalibrationLayer with three changes, mirroring the vertical-target
pipeline (scripts/precompute_phase_targets_vertical.py):
  1. The projector constraint row is the u_p equation (M_p row 0), not v_p.
  2. Fringe period T_p = 18.0291 px and phase offset 2.5461 rad, both fitted
     globally against GT-projected u_p (the horizontal 36 / 2.9 analogs).
  3. Everything else (matrices, scale, pixel grids) is identical.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

T_P_VERTICAL = 18.0291
PHASE_OFFSET_VERTICAL = 2.5461


class CalibrationLayerVertical(nn.Module):
    def __init__(self, calibration_dir, image_size=(960, 960),
                 T_p=T_P_VERTICAL, scale=1.446, phase_offset=PHASE_OFFSET_VERTICAL):
        super().__init__()
        cal_dir = Path(calibration_dir)
        A_c = np.loadtxt(cal_dir / "CamIntrinsicMatrix.txt")
        A_p = np.loadtxt(cal_dir / "ProjIntrinsicMatrix.txt")
        R_p = np.loadtxt(cal_dir / "ProjRotationMatrix.txt")
        T_p_vec = np.loadtxt(cal_dir / "ProjTranslationVector.txt").reshape(3, 1)

        M_c = A_c @ np.hstack([np.eye(3), np.zeros((3, 1))])
        M_p = A_p @ np.hstack([R_p, T_p_vec])

        self.register_buffer('M_c', torch.from_numpy(M_c).double())
        self.register_buffer('M_p', torch.from_numpy(M_p).double())
        self.register_buffer('T_p', torch.tensor(float(T_p), dtype=torch.float64))
        self.register_buffer('scale_factor', torch.tensor(float(scale), dtype=torch.float64))
        self.register_buffer('phase_offset', torch.tensor(float(phase_offset), dtype=torch.float64))

        H, W = image_size
        row_grid, col_grid = torch.meshgrid(
            torch.arange(1, H + 1, dtype=torch.float64),
            torch.arange(1, W + 1, dtype=torch.float64),
            indexing='ij')
        self.register_buffer('u_cam', col_grid.clone())
        self.register_buffer('v_cam', row_grid.clone())

    def forward(self, Phi):
        Phi_d = Phi.squeeze(1).double()
        Phi_offset = Phi_d + self.phase_offset
        up = Phi_offset * self.T_p / (2.0 * torch.pi)

        mc = self.M_c
        mp = self.M_p
        u = self.u_cam
        v = self.v_cam

        A00 = mc[0, 0] - u * mc[2, 0]
        A01 = mc[0, 1] - u * mc[2, 1]
        A02 = mc[0, 2] - u * mc[2, 2]
        A10 = mc[1, 0] - v * mc[2, 0]
        A11 = mc[1, 1] - v * mc[2, 1]
        A12 = mc[1, 2] - v * mc[2, 2]
        # projector u_p equation (row 0)
        A20 = mp[0, 0] - up * mp[2, 0]
        A21 = mp[0, 1] - up * mp[2, 1]
        A22 = mp[0, 2] - up * mp[2, 2]

        b0 = u * mc[2, 3] - mc[0, 3]
        b1 = v * mc[2, 3] - mc[1, 3]
        b2 = up * mp[2, 3] - mp[0, 3]

        det = (A00 * (A11 * A22 - A12 * A21)
             - A01 * (A10 * A22 - A12 * A20)
             + A02 * (A10 * A21 - A11 * A20))
        det_z = (A00 * (A11 * b2 - b1 * A21)
               - A01 * (A10 * b2 - b1 * A20)
               + b0 * (A10 * A21 - A11 * A20))

        z = det_z / (det + 1e-12)
        depth = z / self.scale_factor
        depth = depth.clamp(-5000.0, 5000.0)
        return depth.float().unsqueeze(1)  # (B, 1, H, W)
