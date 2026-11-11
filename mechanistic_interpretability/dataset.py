#!/usr/bin/env python3
"""
Datasets for Mechanistic Interpretability Analysis

Provides datasets that load fringe images with associated shape labels,
phase information, and depth ground truth for probing and MI estimation.

Author: [Your Name]
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import torch
from PIL import Image
from torch.utils.data import Dataset


# =============================================================================
# Shape Categories
# =============================================================================

SHAPE_CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

SHAPE_TO_IDX = {shape: idx for idx, shape in enumerate(SHAPE_CATEGORIES)}


def extract_shape_from_filename(filename: str) -> Optional[str]:
    """
    Extract shape category from filename.

    Expected format: shapename_XXXX.png or similar

    Args:
        filename: Name of the file

    Returns:
        Shape category name or None if not found
    """
    stem = Path(filename).stem
    match = re.match(r'^([a-z_]+)_\d+', stem)
    if match:
        shape = match.group(1)
        if shape in SHAPE_TO_IDX:
            return shape
    return None


# =============================================================================
# Edge Computation
# =============================================================================

def compute_edges(depth: np.ndarray) -> np.ndarray:
    """
    Compute edge map using Sobel filter, normalized to [0, 1].

    Args:
        depth: 2D depth map array

    Returns:
        Normalized edge magnitude map
    """
    sx = ndimage.sobel(depth, axis=1)
    sy = ndimage.sobel(depth, axis=0)
    edges = np.hypot(sx, sy)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    return edges.astype(np.float32)


# =============================================================================
# UNet Dataset with Labels
# =============================================================================

class FPPDatasetWithLabels(Dataset):
    """
    FPP dataset with shape labels for probing and MI analysis.

    Loads fringe images, depth ground truth, and extracts shape labels
    from filenames. Also computes phase representation for probing.

    Args:
        fringe_dir: Directory containing fringe .png files
        depth_dir: Directory containing depth .mat files
        depth_key: Key for depth in .mat files
    """

    def __init__(
        self,
        fringe_dir: str,
        depth_dir: str,
        depth_key: str = 'depthMap'
    ):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.depth_key = depth_key

        # Find all fringe files
        fringe_files = sorted(self.fringe_dir.glob("*.png"))

        self.samples = []
        for f in fringe_files:
            depth_path = self.depth_dir / f"{f.stem}.mat"
            if depth_path.exists():
                shape = extract_shape_from_filename(f.name)
                shape_idx = SHAPE_TO_IDX.get(shape, -1) if shape else -1
                self.samples.append((f, depth_path, shape_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No matching samples found in {fringe_dir}")

        # Count shape distribution
        shape_counts = {}
        for _, _, idx in self.samples:
            if idx >= 0:
                shape = SHAPE_CATEGORIES[idx]
                shape_counts[shape] = shape_counts.get(shape, 0) + 1

        print(f"FPPDatasetWithLabels: {len(self.samples)} samples, "
              f"{len(shape_counts)} shape categories")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fringe_path, depth_path, shape_idx = self.samples[idx]

        # Load fringe
        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0

        # Load depth
        mat = sio.loadmat(depth_path)
        depth = mat[self.depth_key].astype(np.float32)

        # Normalize depth to [0, 1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Compute phase representation (wrapped phase approximation)
        # This is a simplified representation; actual phase requires FPP processing
        phase = np.arctan2(np.sin(fringe * 2 * np.pi), np.cos(fringe * 2 * np.pi))
        phase = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

        # Compute depth from unwrapped phase (simplified)
        phase_depth = depth_norm  # In practice, this would use calibration

        return {
            'fringe': torch.from_numpy(fringe).unsqueeze(0),
            'depth': torch.from_numpy(depth_norm).unsqueeze(0),
            'phase': torch.from_numpy(phase.astype(np.float32)).unsqueeze(0),
            'phase_depth': torch.from_numpy(phase_depth).unsqueeze(0),
            'shape_label': torch.tensor(shape_idx, dtype=torch.long),
            'filename': fringe_path.name,
        }


# =============================================================================
# PhicalNet Dataset with Phase Targets
# =============================================================================

class PhicalNetDataset(Dataset):
    """
    Dataset for PhicalNet probing with phase targets.

    Loads fringe images, depth GT, phase targets (sin/cos), and edges.

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

        # Find matching samples
        fringe_files = sorted(self.fringe_dir.glob("*.png"))
        self.samples = []

        for f in fringe_files:
            stem = f.stem
            depth_path = self.depth_dir / f"{stem}.mat"
            phase_path = self.phase_dir / f"{stem}.npz"

            if depth_path.exists() and phase_path.exists():
                self.samples.append((f, depth_path, phase_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching samples found.\n"
                f"  Fringe dir: {fringe_dir}\n"
                f"  Depth dir: {depth_dir}\n"
                f"  Phase dir: {phase_dir}"
            )

        print(f"PhicalNetDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fringe_path, depth_path, phase_path = self.samples[idx]

        # Load fringe image
        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0

        # Load depth
        mat = sio.loadmat(depth_path)
        depth = mat[self.depth_key].astype(np.float32)

        # Compute edges from depth
        edges = compute_edges(depth)

        # Load phase targets
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


if __name__ == "__main__":
    print("Testing datasets...")
    print(f"\nShape categories ({len(SHAPE_CATEGORIES)}):")
    for i, shape in enumerate(SHAPE_CATEGORIES):
        print(f"  {i}: {shape}")
