"""Sum/difference-encoded variant of PhaseMultiFrameDataset for 2-frame pairs.

For the 180-degree pair, (I_0 - I_9)/2 = B*cos(phi) cancels the background
exactly and (I_0 + I_9)/2 = A isolates it (Section 6.2 of the manuscript).
This wrapper feeds the network those physical channels instead of raw frames,
so the background separation is performed by arithmetic rather than learned.
Applies to any pair; the encoding is exact background cancellation only at
180 degrees.
"""
import torch

from data.phase_dataset_multiframe import PhaseMultiFrameDataset


class PhaseMultiFrameSumDiffDataset(PhaseMultiFrameDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        f = sample["fringe"]
        assert f.shape[0] == 2, "sum/diff encoding is defined for 2-frame input"
        sample["fringe"] = torch.stack(
            [(f[0] - f[1]) * 0.5, (f[0] + f[1]) * 0.5], dim=0)
        return sample
