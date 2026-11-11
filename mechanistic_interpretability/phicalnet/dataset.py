#!/usr/bin/env python3
"""
Dataset for PhicalNet Analysis

Author: [Your Name]
"""

from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import torch
from PIL import Image
from torch.utils.data import Dataset


def compute_edges(depth: np.ndarray) -> np.ndarray:
    """Compute edge map using Sobel filter."""
    sx = ndimage.sobel(depth, axis=1)
    sy = ndimage.sobel(depth, axis=0)
    edges = np.hypot(sx, sy)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    return edges.astype(np.float32)


class PhicalNetDataset(Dataset):
    """
    Dataset for PhicalNet probing with phase targets.

    Args:
        fringe_dir: Directory containing fringe .png files
        depth_dir: Directory containing depth .mat files
        phase_dir: Directory containing phase target .npz files
        depth_key: Key for depth in .mat files
    """

    def __init__(
        self,
        fringe_dir: str,
        depth_dir: str,
        phase_dir: str,
        depth_key: str = 'depthMapNormalized'
    ):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.phase_dir = Path(phase_dir)
        self.depth_key = depth_key

        fringe_files = sorted(self.fringe_dir.glob("*.png"))
        self.samples = []

        for f in fringe_files:
            stem = f.stem
            depth_path = self.depth_dir / f"{stem}.mat"
            phase_path = self.phase_dir / f"{stem}.npz"

            if depth_path.exists() and phase_path.exists():
                self.samples.append((f, depth_path, phase_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No matching samples found in {fringe_dir}")

        print(f"PhicalNetDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fringe_path, depth_path, phase_path = self.samples[idx]

        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0
        mat = sio.loadmat(depth_path)
        depth = mat[self.depth_key].astype(np.float32)
        edges = compute_edges(depth)

        npz = np.load(phase_path)
        sin_phi = npz['sin_phi'].astype(np.float32)
        cos_phi = npz['cos_phi'].astype(np.float32)
        k_gt = npz['k'].astype(np.int32)
        mask = npz['mask'].astype(np.float32)
        d_min = float(npz['d_min'])
        d_max = float(npz['d_max'])
        depth_mm = npz['depth_mm'].astype(np.float32)

        return {
            'fringe': torch.from_numpy(fringe).unsqueeze(0),
            'depth': torch.from_numpy(depth).unsqueeze(0),
            'edges': torch.from_numpy(edges).unsqueeze(0),
            'sin_phi': torch.from_numpy(sin_phi).unsqueeze(0),
            'cos_phi': torch.from_numpy(cos_phi).unsqueeze(0),
            'k_gt': torch.from_numpy(k_gt).long(),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'depth_mm': torch.from_numpy(depth_mm).unsqueeze(0),
            'd_min': torch.tensor(d_min),
            'd_max': torch.tensor(d_max),
            'filename': fringe_path.name,
        }
