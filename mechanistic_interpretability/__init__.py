"""
Mechanistic Interpretability Analysis for Depth Prediction Networks

This package provides tools for analyzing what UNet and PhicalNet learn
when predicting depth from fringe projection profilometry (FPP) images.

Analysis Methods:
    - GradCAM: Visualize spatial attention patterns
    - Neural Probing: Test information decodability at each layer
    - Mutual Information: Quantify information about shape vs. physics

Subpackages:
    - phicalnet: Analysis tools specific to PhicalNet (PhaseUNetFPP)

Author: [Your Name]
"""

from .utils import (
    compute_edges,
    compute_fringe_frequency_map,
    compute_correlation,
    disable_inplace_relu,
    load_checkpoint,
)

__all__ = [
    'compute_edges',
    'compute_fringe_frequency_map',
    'compute_correlation',
    'disable_inplace_relu',
    'load_checkpoint',
]

__version__ = '1.0.0'
