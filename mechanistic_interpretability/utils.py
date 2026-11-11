"""
Shared Utilities for Mechanistic Interpretability Analysis

This module provides common functions used across GradCAM, neural probing,
and mutual information analyses for investigating shape priors vs physics
learning in UNet-based depth prediction networks.

Author: [Your Name]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
from scipy.ndimage import uniform_filter
from typing import Dict, Tuple, Optional


# =============================================================================
# Image Processing Utilities
# =============================================================================

def compute_edges(depth: np.ndarray) -> np.ndarray:
    """
    Compute edge map using Sobel filter.

    Args:
        depth: 2D depth map array

    Returns:
        Normalized edge map in [0, 1]
    """
    sx = ndimage.sobel(depth, axis=1)
    sy = ndimage.sobel(depth, axis=0)
    edges = np.hypot(sx, sy)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
    return edges.astype(np.float32)


def compute_fringe_frequency_map(fringe: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute local fringe frequency/intensity variation map.

    High values indicate strong fringe patterns. Uses local standard
    deviation to capture fringe intensity variation.

    Args:
        fringe: 2D fringe pattern array, values in [0, 1]
        window_size: Size of the local window for computing statistics

    Returns:
        Normalized fringe frequency map in [0, 1]
    """
    mean = uniform_filter(fringe, size=window_size)
    mean_sq = uniform_filter(fringe ** 2, size=window_size)
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)
    return local_std.astype(np.float32)


def compute_correlation(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Compute Pearson correlation between two spatial maps.

    Args:
        map1: First 2D array
        map2: Second 2D array

    Returns:
        Pearson correlation coefficient, or 0.0 if computation fails
    """
    map1_flat = map1.flatten()
    map2_flat = map2.flatten()

    if np.std(map1_flat) < 1e-8 or np.std(map2_flat) < 1e-8:
        return 0.0

    correlation = np.corrcoef(map1_flat, map2_flat)[0, 1]
    return 0.0 if np.isnan(correlation) else float(correlation)


# =============================================================================
# Model Utilities
# =============================================================================

def disable_inplace_relu(model: nn.Module) -> None:
    """
    Disable inplace operations in ReLU layers.

    Required for GradCAM to work correctly, as inplace operations
    can cause gradient computation issues.

    Args:
        model: PyTorch model to modify
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    device: str = 'cpu') -> nn.Module:
    """
    Load model checkpoint with flexible state dict handling.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


# =============================================================================
# Feature Extraction Utilities
# =============================================================================

def extract_global_features(activations: Dict[str, torch.Tensor],
                            pooling: str = 'avg',
                            exclude_keys: Optional[Tuple[str, ...]] = None
                            ) -> Dict[str, torch.Tensor]:
    """
    Apply global pooling to get fixed-size feature vectors from activations.

    Args:
        activations: Dictionary of layer activations (B, C, H, W)
        pooling: 'avg' for global average pooling, 'max' for global max pooling
        exclude_keys: Keys to exclude from processing (e.g., 'output')

    Returns:
        Dictionary mapping layer name to feature vector (B, C)
    """
    if exclude_keys is None:
        exclude_keys = ('output', 'sin_phi_hat', 'cos_phi_hat')

    pool_fn = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

    features = {}
    for name, act in activations.items():
        if name in exclude_keys:
            continue
        pooled = pool_fn(act)
        features[name] = pooled.view(pooled.size(0), -1)

    return features


def extract_spatial_features(activations: Dict[str, torch.Tensor],
                             target_size: Tuple[int, int] = (30, 30),
                             exclude_keys: Optional[Tuple[str, ...]] = None
                             ) -> Dict[str, torch.Tensor]:
    """
    Resize all activations to a common spatial size for comparison.

    Args:
        activations: Dictionary of layer activations
        target_size: Target spatial dimensions (H, W)
        exclude_keys: Keys to exclude from processing

    Returns:
        Dictionary mapping layer name to resized activations (B, C, H, W)
    """
    if exclude_keys is None:
        exclude_keys = ('output', 'sin_phi_hat', 'cos_phi_hat')

    features = {}
    for name, act in activations.items():
        if name in exclude_keys:
            continue
        resized = F.interpolate(act, size=target_size, mode='bilinear', align_corners=False)
        features[name] = resized

    return features


# =============================================================================
# Data Processing Utilities
# =============================================================================

DEPTH_KEYS = {
    "_raw": "depthMap",
    "_global_normalized": "depthMapMeters",
    "_individual_normalized": "depthMapNormalized"
}


def get_depth_key(dataset_type: str) -> str:
    """
    Get the appropriate depth key for a dataset type.

    Args:
        dataset_type: One of '_raw', '_global_normalized', '_individual_normalized'

    Returns:
        Key to use when loading depth from .mat files
    """
    return DEPTH_KEYS.get(dataset_type, "depthMapNormalized")


def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize tensor to zero mean and unit variance.

    Args:
        tensor: Input tensor
        eps: Small value for numerical stability

    Returns:
        Normalized tensor
    """
    return (tensor - tensor.mean()) / (tensor.std() + eps)


# =============================================================================
# JSON Serialization Utilities
# =============================================================================

def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects for JSON.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj
