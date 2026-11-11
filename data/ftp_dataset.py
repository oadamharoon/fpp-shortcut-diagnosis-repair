"""
Dataset for FTP architecture training.

Loads per-sample:
  - fringe: (1, 960, 960) grayscale fringe image (frame 0)
  - phi_gt: (1, 960, 960) wrapped phase from 18-step
  - sin_phi_gt: (1, 960, 960)
  - cos_phi_gt: (1, 960, 960)
  - k_gt: (960, 960) long, fringe order [0, 127]
  - mask_gt: (1, 960, 960) binary object mask
  - depth_gt: (1, 960, 960) depth in mm
  - d_min, d_max: normalization parameters
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class FTPDataset(Dataset):
    """Dataset for FTP architecture with phase targets."""

    def __init__(self, fringe_dir, phase_targets_dir):
        """
        Args:
            fringe_dir: Path to fringe PNGs (e.g., .../train/fringe/)
            phase_targets_dir: Path to precomputed .npz files (e.g., .../train/)
        """
        self.fringe_dir = Path(fringe_dir)
        self.phase_dir = Path(phase_targets_dir)

        # Find all samples that have both fringe and phase targets
        self.samples = []
        for fringe_path in sorted(self.fringe_dir.glob("*.png")):
            stem = fringe_path.stem
            phase_path = self.phase_dir / f"{stem}.npz"
            if phase_path.exists():
                self.samples.append((fringe_path, phase_path))

        if not self.samples:
            raise ValueError(
                f"No matching samples found.\n"
                f"  Fringe dir: {fringe_dir}\n"
                f"  Phase dir: {phase_targets_dir}"
            )

        print(f"FTPDataset: loaded {len(self.samples)} samples from {fringe_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fringe_path, phase_path = self.samples[idx]

        # Load fringe image
        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32)
        fringe = fringe / 255.0
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # (1, H, W)

        # Load precomputed phase targets
        npz = np.load(phase_path)

        sin_phi = torch.from_numpy(npz['sin_phi']).unsqueeze(0)  # (1, H, W)
        cos_phi = torch.from_numpy(npz['cos_phi']).unsqueeze(0)  # (1, H, W)
        k_gt = torch.from_numpy(npz['k'].astype(np.int64)).clamp(0, 31)  # (H, W)
        mask_gt = torch.from_numpy(npz['mask']).unsqueeze(0)      # (1, H, W)
        depth_mm = torch.from_numpy(npz['depth_mm']).unsqueeze(0) # (1, H, W)
        d_min = float(npz['d_min'])
        d_max = float(npz['d_max'])

        # Wrapped phase from sin/cos
        phi_gt = torch.atan2(sin_phi, cos_phi)  # (1, H, W)

        return {
            'fringe': fringe,
            'phi_gt': phi_gt,
            'sin_phi_gt': sin_phi,
            'cos_phi_gt': cos_phi,
            'k_gt': k_gt,
            'mask_gt': mask_gt,
            'depth_gt': depth_mm,
            'd_min': d_min,
            'd_max': d_max,
            'name': fringe_path.stem,
        }
