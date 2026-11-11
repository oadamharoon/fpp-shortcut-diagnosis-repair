#!/usr/bin/env python3
"""
GradCAM Analysis for UNet Depth Prediction

Visualizes what regions of the input the network focuses on when predicting
depth using Gradient-weighted Class Activation Mapping.

Hypothesis Testing:
    If the network learns shape priors:
    - GradCAM heatmaps should highlight edges/boundaries (geometric features)

    If the network learns physical fringe-to-depth relationships:
    - GradCAM should correlate with fringe pattern variations

Usage:
    python gradcam_analysis.py --pretrained /path/to/model.pth \\
                               --data_root /path/to/data \\
                               --output_dir ./gradcam_results

Author: [Your Name]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt

# Setup paths
import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent))
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet"):
    sys.path.insert(0, str(_p))
from unet import UNetFPP

from utils import (
    compute_edges,
    compute_fringe_frequency_map,
    compute_correlation,
    disable_inplace_relu,
    load_checkpoint,
    get_depth_key,
)


# =============================================================================
# GradCAM Implementation
# =============================================================================

class GradCAM:
    """
    GradCAM implementation for regression tasks.

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization", ICCV 2017
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        output = self.model(input_tensor)
        target = output.mean()

        self.model.zero_grad()
        target.backward(retain_graph=True)

        if self.gradients is None:
            return np.zeros(input_tensor.shape[2:])

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                           mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


# =============================================================================
# Layer Selection
# =============================================================================

def get_target_layers(model: UNetFPP) -> Dict[str, nn.Module]:
    """Get target layers for GradCAM visualization."""
    return {
        'enc1': model.down1.conv.conv[-1],
        'enc2': model.down2.conv.conv[-1],
        'enc3': model.down3.conv.conv[-1],
        'enc4': model.down4.conv.conv[-1],
        'bottleneck': model.bottleneck.conv[-1],
        'dec1': model.up1.conv.conv[-1],
        'dec2': model.up2.conv.conv[-1],
        'dec3': model.up3.conv.conv[-1],
        'dec4': model.up4.conv.conv[-1],
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_gradcam_single(
    fringe: np.ndarray,
    gradcam_maps: Dict[str, np.ndarray],
    output_path: str,
    sample_name: str = ""
) -> None:
    """Create visualization of GradCAM overlays at each layer."""
    n_layers = len(gradcam_maps)
    fig, axes = plt.subplots(1, n_layers + 1, figsize=(3 * (n_layers + 1), 3))

    axes[0].imshow(fringe, cmap='gray')
    axes[0].set_title('Input', fontsize=10)
    axes[0].axis('off')

    for i, (layer_name, cam) in enumerate(gradcam_maps.items()):
        axes[i + 1].imshow(fringe, cmap='gray')
        axes[i + 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(layer_name, fontsize=10)
        axes[i + 1].axis('off')

    plt.suptitle(f'GradCAM: {sample_name}', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_correlation_summary(
    all_correlations: Dict[str, Dict[str, List[float]]],
    output_path: str
) -> None:
    """Create summary plot of correlations."""
    layers = list(all_correlations.keys())

    edge_means = [np.mean(all_correlations[l]['edge']) for l in layers]
    edge_stds = [np.std(all_correlations[l]['edge']) for l in layers]
    fringe_means = [np.mean(all_correlations[l]['fringe']) for l in layers]
    fringe_stds = [np.std(all_correlations[l]['fringe']) for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - width/2, edge_means, width, yerr=edge_stds,
            label='Correlation with Edges', color='#e74c3c', capsize=3)
    ax1.bar(x + width/2, fringe_means, width, yerr=fringe_stds,
            label='Correlation with Fringe', color='#3498db', capsize=3)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_title('GradCAM Correlation: Edges vs Fringe Patterns', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-0.2, 1.0)

    ratios = [e / max(f, 0.01) for e, f in zip(edge_means, fringe_means)]
    colors = ['#e74c3c' if r > 1 else '#3498db' for r in ratios]
    ax2.bar(x, ratios, color=colors, edgecolor='black')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal')
    ax2.set_ylabel('Edge Corr / Fringe Corr', fontsize=12)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_title('Correlation Ratio (>1 = Attends More to Edges)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_text_report(
    all_correlations: Dict[str, Dict[str, List[float]]],
    output_path: str
) -> None:
    """Generate text report summarizing results."""
    layers = list(all_correlations.keys())

    report = []
    report.append("=" * 70)
    report.append("GRADCAM ANALYSIS REPORT - UNET")
    report.append("=" * 70)
    report.append("")
    report.append("If network learns SHAPE PRIORS:")
    report.append("  - GradCAM should correlate highly with EDGES")
    report.append("")
    report.append("If network learns PHYSICS:")
    report.append("  - GradCAM should correlate with FRINGE PATTERNS")
    report.append("")
    report.append("-" * 70)
    report.append(f"{'Layer':<15} {'r(CAM,Edge)':<15} {'r(CAM,Fringe)':<15} {'Ratio':<10}")
    report.append("-" * 70)

    total_edge, total_fringe, valid_layers = 0, 0, 0

    for layer in layers:
        edge_corr = np.mean(all_correlations[layer]['edge'])
        fringe_corr = np.mean(all_correlations[layer]['fringe'])

        if np.isnan(edge_corr) or np.isnan(fringe_corr):
            continue

        ratio = edge_corr / max(fringe_corr, 0.01)
        total_edge += edge_corr
        total_fringe += fringe_corr
        valid_layers += 1
        report.append(f"{layer:<15} {edge_corr:<15.3f} {fringe_corr:<15.3f} {ratio:<10.2f}")

    report.append("-" * 70)

    if valid_layers > 0:
        avg_edge = total_edge / valid_layers
        avg_fringe = total_fringe / valid_layers
        avg_ratio = avg_edge / max(avg_fringe, 0.01)
        report.append(f"{'AVERAGE':<15} {avg_edge:<15.3f} {avg_fringe:<15.3f} {avg_ratio:<10.2f}")

        report.append("")
        report.append("=" * 70)
        report.append("CONCLUSION")
        report.append("=" * 70)

        if avg_ratio > 1.5:
            report.append("STRONG EVIDENCE FOR SHAPE PRIOR LEARNING")
        elif avg_ratio > 1.0:
            report.append("MODERATE EVIDENCE FOR SHAPE PRIOR LEARNING")
        else:
            report.append("WEAK/NO EVIDENCE FOR SHAPE PRIOR LEARNING")

    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_gradcam_analysis(
    pretrained_path: str,
    data_root: str,
    output_dir: str,
    dataset_type: str = "_individual_normalized",
    num_samples: int = 10,
    device: str = 'cuda',
    layers_to_analyze: Optional[List[str]] = None
) -> Dict:
    """Run GradCAM analysis on UNet depth prediction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {pretrained_path}")
    model = UNetFPP(in_channels=1, out_channels=1)
    model = load_checkpoint(model, pretrained_path, device)
    model.to(device).eval()
    disable_inplace_relu(model)

    # Get target layers
    all_layers = get_target_layers(model)
    if layers_to_analyze is None:
        layers_to_analyze = ['enc3', 'enc4', 'bottleneck', 'dec1', 'dec3', 'dec4']
    target_layers = {k: v for k, v in all_layers.items() if k in layers_to_analyze}
    print(f"Analyzing layers: {list(target_layers.keys())}")

    # Setup data paths
    depth_key = get_depth_key(dataset_type)
    base_path = Path(data_root) / f"training_data_depth{dataset_type}"
    fringe_dir = base_path / "test" / "fringe"
    depth_dir = base_path / "test" / "depth"

    fringe_files = sorted(list(fringe_dir.glob("*.png")))[:num_samples]
    print(f"Processing {len(fringe_files)} samples")

    # Initialize correlation storage
    all_correlations = {layer: {'edge': [], 'fringe': []} for layer in target_layers.keys()}

    # Process samples
    for idx, fringe_path in enumerate(fringe_files):
        print(f"\nSample {idx+1}/{len(fringe_files)}: {fringe_path.name}")

        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0

        depth_path = depth_dir / f"{fringe_path.stem}.mat"
        if not depth_path.exists():
            continue

        mat = sio.loadmat(depth_path)
        depth_gt = mat[depth_key].astype(np.float32)
        depth_gt_norm = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-8)

        edges = compute_edges(depth_gt_norm)
        fringe_freq = compute_fringe_frequency_map(fringe)

        input_tensor = torch.from_numpy(fringe).unsqueeze(0).unsqueeze(0).to(device)

        gradcam_maps = {}
        for layer_name, layer_module in target_layers.items():
            gradcam = GradCAM(model, layer_module)
            cam = gradcam(input_tensor.clone())
            gradcam_maps[layer_name] = cam

            corr_edge = compute_correlation(cam, edges)
            corr_fringe = compute_correlation(cam, fringe_freq)
            all_correlations[layer_name]['edge'].append(corr_edge)
            all_correlations[layer_name]['fringe'].append(corr_fringe)

            gradcam.remove_hooks()

        visualize_gradcam_single(
            fringe, gradcam_maps,
            str(output_dir / f"gradcam_sample_{idx:03d}.png"),
            sample_name=fringe_path.stem
        )

    # Generate outputs
    visualize_correlation_summary(all_correlations, str(output_dir / "gradcam_correlation_summary.png"))
    generate_text_report(all_correlations, str(output_dir / "gradcam_report.txt"))

    print(f"\nResults saved to: {output_dir}")
    return all_correlations


def main():
    parser = argparse.ArgumentParser(description="GradCAM Analysis for UNet")

    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained UNet checkpoint")
    parser.add_argument("--data_root", type=str, default=str(_ROOT),
                        help="Path to data root directory")
    parser.add_argument("--output_dir", type=str, default="gradcam_results",
                        help="Output directory")
    parser.add_argument("--dataset_type", type=str, default="_individual_normalized",
                        choices=["_raw", "_global_normalized", "_individual_normalized"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--layers", type=str, nargs='+', default=None)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_gradcam_analysis(
        pretrained_path=args.pretrained,
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        num_samples=args.num_samples,
        device=device,
        layers_to_analyze=args.layers
    )


if __name__ == "__main__":
    main()
