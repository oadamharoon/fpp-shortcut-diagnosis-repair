"""
Multi-frame variant of PhaseDataset: stacks three phase-shifted sinusoidal
frames as a 3-channel input so the network can resolve phase analytically
rather than approximating the discontinuous atan2.

Frames are loaded from the preprocessed
`training_data_depth_individual_normalized_multiframe_horizontal_only`
directory, where each sample is stored as
    {stem}_F{0..17}.png
with per-sample normalization already applied. We pick F0, F6, F12 -- the
classical 3-step phase-shift set, spaced exactly 120 degrees apart under
the 18-step generator.

Phase targets (sin_phi, cos_phi, k, mask, depth) are shared with the
single-frame pipeline and loaded from the unchanged `phase_targets/`
directory: the extra frames only change the input, not the supervision.

Sample output:
    fringe:     (C, H, W) fp32  -- len(frame_indices) stacked channels
    sin_phi_gt: (1, H, W) fp32
    cos_phi_gt: (1, H, W) fp32
    k_gt:       (H, W)    long
    mask_gt:    (1, H, W) fp32
    depth_gt:   (1, H, W) fp32  (mm)
    d_min, d_max: scalars
    name:       sample stem
"""

from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

# 0-indexed frame numbers picked for the 3-step phase-shift set.
# Under the 18-step generator delta_i = 2*pi/18 * (i-2) + pi, picking every
# sixth frame gives shifts of 120 degrees exactly.
DEFAULT_FRAMES = (0, 6, 12)


class PhaseMultiFrameDataset(Dataset):
    """Drop-in replacement for PhaseDataset that returns multi-channel input.

    `fringe_dir` points at the horizontal-only multi-frame directory (e.g.
    training_datasets/training_data_depth_individual_normalized_multiframe_horizontal_only/train/fringe/),
    which contains files named {stem}_F{i}.png.
    """

    def __init__(self, fringe_dir, phase_targets_dir,
                 K_max=32, frame_indices=DEFAULT_FRAMES):
        self.fringe_dir = Path(fringe_dir)
        self.phase_dir = Path(phase_targets_dir)
        self.K_max = K_max
        self.frame_indices = tuple(frame_indices)

        # Enumerate unique sample stems by looking at the F0 frames; that
        # gives one entry per sample even though the directory contains 18
        # frames per sample.
        stem_re = re.compile(r"(?P<stem>.+)_F0$")
        self.samples = []
        for p in sorted(self.fringe_dir.glob("*_F0.png")):
            m = stem_re.match(p.stem)
            if not m:
                continue
            stem = m.group("stem")
            phase_path = self.phase_dir / f"{stem}.npz"
            frame_paths = [self.fringe_dir / f"{stem}_F{i}.png"
                           for i in self.frame_indices]
            if phase_path.exists() and all(fp.exists() for fp in frame_paths):
                self.samples.append((stem, phase_path, frame_paths))

        if not self.samples:
            raise ValueError(
                f"No matching samples.\n"
                f"  fringe_dir: {fringe_dir}\n"
                f"  phase_dir:  {phase_targets_dir}\n"
                f"  frames:     {self.frame_indices}"
            )

        print(f"PhaseMultiFrameDataset: {len(self.samples)} samples "
              f"(stacked frames {self.frame_indices}) from {fringe_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, phase_path, frame_paths = self.samples[idx]

        channels = []
        for fp in frame_paths:
            arr = np.array(Image.open(fp).convert('L'), dtype=np.float32) / 255.0
            channels.append(arr)
        fringe = torch.from_numpy(np.stack(channels, axis=0))  # (C, H, W)

        # Precomputed phase targets (same as single-frame dataset)
        npz = np.load(phase_path)
        sin_phi_gt = torch.from_numpy(npz['sin_phi']).unsqueeze(0)
        cos_phi_gt = torch.from_numpy(npz['cos_phi']).unsqueeze(0)

        k_raw = npz['k'].astype(np.float64)
        k_filtered = median_filter(k_raw, size=7)
        k_clipped = np.clip(k_filtered, 0, self.K_max - 1).astype(np.int64)
        k_gt = torch.from_numpy(k_clipped)

        mask_gt = torch.from_numpy(npz['mask']).unsqueeze(0)
        depth_gt_mm = torch.from_numpy(npz['depth_mm']).unsqueeze(0)
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
            'name': stem,
        }
