"""
Dataset for Phase-Intermediate UNet (variants A1 and A2).

Per sample:
    fringe:     (1, H, W) fp32 — same first-frame input as baseline
    sin_phi_gt: (1, H, W) fp32
    cos_phi_gt: (1, H, W) fp32
    k_gt:       (H, W) long — values in [0, K_max-1], 0 where no object
    mask_gt:    (1, H, W) fp32 — 1 for object pixels, 0 background
    depth_gt:   (1, H, W) fp32 — in mm (already denormalized)
    d_min:      scalar — per-sample normalization parameter
    d_max:      scalar — per-sample normalization parameter
    name:       sample stem

Phase targets are loaded from precomputed .npz files in phase_targets/
(validated to reproduce depth.mat to <1e-3 mm via the calibration layer).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from scipy.ndimage import median_filter


class PhaseDataset(Dataset):
    def __init__(self, fringe_dir, phase_targets_dir, K_max=32):
        """
        Args:
            fringe_dir: Path to fringe PNGs (e.g., .../train/fringe/)
            phase_targets_dir: Path to precomputed .npz files for this split
            K_max: number of fringe order classes (clamp k values to [0, K_max-1])
        """
        self.fringe_dir = Path(fringe_dir)
        self.phase_dir = Path(phase_targets_dir)
        self.K_max = K_max

        self.samples = []
        for fringe_path in sorted(self.fringe_dir.glob("*.png")):
            stem = fringe_path.stem
            phase_path = self.phase_dir / f"{stem}.npz"
            if phase_path.exists():
                self.samples.append((fringe_path, phase_path))

        if not self.samples:
            raise ValueError(
                f"No matching samples.\n"
                f"  Fringe: {fringe_dir}\n"
                f"  Phase:  {phase_targets_dir}"
            )

        print(f"PhaseDataset: {len(self.samples)} samples from {fringe_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fringe_path, phase_path = self.samples[idx]

        # Fringe image
        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # (1, H, W)

        # Precomputed phase targets
        npz = np.load(phase_path)
        sin_phi_gt = torch.from_numpy(npz['sin_phi']).unsqueeze(0)  # (1, H, W)
        cos_phi_gt = torch.from_numpy(npz['cos_phi']).unsqueeze(0)  # (1, H, W)
        # k was stored as int32 with values in [0, 127] (7-bit gray code).
        # Apply MATLAB-style 7x7 median filter to clean up gray code boundary
        # noise (matches the unwrap pipeline that reproduces depth.mat exactly).
        # Then clamp to [0, K_max-1] since we use 5 effective bits.
        k_raw = npz['k'].astype(np.float64)
        k_filtered = median_filter(k_raw, size=7)
        k_clipped = np.clip(k_filtered, 0, self.K_max - 1).astype(np.int64)
        k_gt = torch.from_numpy(k_clipped)  # (H, W)
        mask_gt = torch.from_numpy(npz['mask']).unsqueeze(0)  # (1, H, W)
        depth_gt_mm = torch.from_numpy(npz['depth_mm']).unsqueeze(0)  # (1, H, W) in mm
        d_min = float(npz['d_min'])
        d_max = float(npz['d_max'])

        return {
            'fringe': fringe,
            'sin_phi_gt': sin_phi_gt,
            'cos_phi_gt': cos_phi_gt,
            'k_gt': k_gt,
            'mask_gt': mask_gt,
            'depth_gt': depth_gt_mm,
            'd_min': d_min,
            'd_max': d_max,
            'name': fringe_path.stem,
        }
