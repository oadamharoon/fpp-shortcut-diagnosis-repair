#!/usr/bin/env python3
"""
GradCAM Analysis for PhicalNet (PhaseUNetFPP)

Two analysis modes:
    1. PHASE mode: gradient w.r.t. phase prediction
    2. DEPTH mode: gradient w.r.t. depth prediction

Usage:
    python gradcam_analysis.py --pretrained /path/to/model.pth \\
                               --output_dir ./phicalnet_gradcam_results

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

import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent.parent))
_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet", _REPO / "mechanistic_interpretability"):
    sys.path.insert(0, str(_p))
from phase_unet import PhaseUNetFPP
from calibration_layer import CalibrationLayer

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import compute_edges, compute_fringe_frequency_map, compute_correlation, disable_inplace_relu


def compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer, T_p=36.0):
    """Compute depth from predicted sin/cos and integer fringe order k."""
    shift = -torch.pi / T_p
    cos_s = torch.cos(torch.tensor(shift, dtype=sin_hat.dtype, device=sin_hat.device))
    sin_s = torch.sin(torch.tensor(shift, dtype=sin_hat.dtype, device=sin_hat.device))

    shifted_sin = sin_hat * cos_s + cos_hat * sin_s
    shifted_cos = cos_hat * cos_s - sin_hat * sin_s

    wph_shifted = torch.atan2(shifted_sin, shifted_cos)
    Phi = wph_shifted + 2.0 * torch.pi * k_tensor.float() - shift

    return cal_layer(Phi)


class GradCAM:
    """GradCAM implementation for PhicalNet."""

    def __init__(self, target_layer: nn.Module):
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

    def compute_cam(self, input_size: Tuple[int, int]) -> np.ndarray:
        if self.gradients is None or self.activations is None:
            return np.zeros(input_size)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


def get_target_layers(model: PhaseUNetFPP) -> Dict[str, nn.Module]:
    """Get target layers for GradCAM."""
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


def gradcam_phase_mode(model, fringe_tensor, target_layers, input_size):
    """Compute GradCAM using phase output as target."""
    cams = {}
    for layer_name, layer_module in target_layers.items():
        gc = GradCAM(layer_module)
        model.zero_grad()
        x = fringe_tensor.clone().detach().requires_grad_(True)
        out = model(x)
        target = out['sin_phi_hat'].mean() + out['cos_phi_hat'].mean()
        target.backward()
        cams[layer_name] = gc.compute_cam(input_size)
        gc.remove_hooks()
    return cams


def gradcam_depth_mode(model, cal_layer, fringe_tensor, k_gt_tensor, target_layers, input_size):
    """Compute GradCAM using depth output as target."""
    cams = {}
    for layer_name, layer_module in target_layers.items():
        gc = GradCAM(layer_module)
        model.zero_grad()
        x = fringe_tensor.clone().detach().requires_grad_(True)
        out = model(x)
        depth_mm = compute_depth_from_phase(
            out['sin_phi_hat'], out['cos_phi_hat'], k_gt_tensor, cal_layer
        )
        target = depth_mm.mean()
        target.backward()
        cams[layer_name] = gc.compute_cam(input_size)
        gc.remove_hooks()
    return cams


def visualize_sample(fringe, gradcam_maps, output_path, sample_name=""):
    """Create visualization of GradCAM overlays."""
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


def visualize_correlation_summary(all_correlations, output_path, mode_label='Phase'):
    """Create correlation summary bar charts."""
    layers = list(all_correlations.keys())
    edge_means = [np.mean(all_correlations[l]['edge']) for l in layers]
    fringe_means = [np.mean(all_correlations[l]['fringe']) for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, edge_means, width, label='Edge Corr', color='#e74c3c')
    ax.bar(x + width/2, fringe_means, width, label='Fringe Corr', color='#3498db')
    ax.set_ylabel('Pearson Correlation')
    ax.set_xlabel('Layer')
    ax.set_title(f'GradCAM ({mode_label}) Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_gradcam_analysis(
    pretrained_path: str,
    data_root: str,
    phase_root: str,
    cal_dir: str,
    output_dir: str,
    num_samples: int = 30,
    device: str = 'cuda',
    layers_to_analyze: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
) -> Dict:
    """Run GradCAM analysis on PhicalNet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if modes is None:
        modes = ['phase', 'depth']

    print(f"Loading PhicalNet from {pretrained_path}")
    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=32, dropout_rate=0.0)
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    disable_inplace_relu(model)

    cal_layer = CalibrationLayer(cal_dir).to(device)

    all_layers = get_target_layers(model)
    if layers_to_analyze is None:
        layers_to_analyze = ['enc3', 'enc4', 'bottleneck', 'dec1', 'dec3', 'dec4']
    target_layers = {k: v for k, v in all_layers.items() if k in layers_to_analyze}

    base_path = Path(data_root) / "training_data_depth_individual_normalized"
    fringe_dir = base_path / "test" / "fringe"
    depth_dir = base_path / "test" / "depth"
    phase_dir = Path(phase_root) / "test"

    fringe_files = sorted(fringe_dir.glob("*.png"))[:num_samples]
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"GRADCAM MODE: {mode.upper()}")
        print('='*60)

        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)

        all_correlations = {layer: {'edge': [], 'fringe': []} for layer in target_layers.keys()}

        for idx, fringe_path in enumerate(fringe_files):
            print(f"[{mode}] Sample {idx+1}/{len(fringe_files)}: {fringe_path.name}")

            fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0

            depth_path = depth_dir / f"{fringe_path.stem}.mat"
            if not depth_path.exists():
                continue

            mat = sio.loadmat(depth_path)
            depth_gt = mat['depthMapNormalized'].astype(np.float32)

            edges = compute_edges(depth_gt)
            fringe_freq = compute_fringe_frequency_map(fringe)

            fringe_tensor = torch.from_numpy(fringe).unsqueeze(0).unsqueeze(0).to(device)
            input_size = (fringe.shape[0], fringe.shape[1])

            if mode == 'phase':
                gradcam_maps = gradcam_phase_mode(model, fringe_tensor, target_layers, input_size)
            elif mode == 'depth':
                phase_path = phase_dir / f"{fringe_path.stem}.npz"
                if not phase_path.exists():
                    continue
                npz = np.load(phase_path)
                k_gt = npz['k'].astype(np.int32)
                k_gt_tensor = torch.from_numpy(k_gt).unsqueeze(0).unsqueeze(0).to(device)
                gradcam_maps = gradcam_depth_mode(model, cal_layer, fringe_tensor, k_gt_tensor, target_layers, input_size)

            for layer_name, cam in gradcam_maps.items():
                ce = compute_correlation(cam, edges)
                cf = compute_correlation(cam, fringe_freq)
                all_correlations[layer_name]['edge'].append(ce)
                all_correlations[layer_name]['fringe'].append(cf)

            visualize_sample(fringe, gradcam_maps, str(mode_dir / f"gradcam_sample_{idx:03d}.png"), fringe_path.stem)

        visualize_correlation_summary(all_correlations, str(mode_dir / "gradcam_correlation_summary.png"), mode.capitalize())
        results[mode] = all_correlations

    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="GradCAM Analysis for PhicalNet")

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(_ROOT))
    parser.add_argument("--phase_root", type=str, default=str(_ROOT / "training_datasets" / "phase_targets"))
    parser.add_argument("--cal_dir", type=str, default=str(_REPO / "calibration_matrices"))
    parser.add_argument("--output_dir", type=str, default="phicalnet_gradcam_results")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--layers", type=str, nargs='+', default=None)
    parser.add_argument("--modes", type=str, nargs='+', default=['phase', 'depth'], choices=['phase', 'depth'])

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_gradcam_analysis(
        pretrained_path=args.pretrained,
        data_root=args.data_root,
        phase_root=args.phase_root,
        cal_dir=args.cal_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=device,
        layers_to_analyze=args.layers,
        modes=args.modes,
    )


if __name__ == "__main__":
    main()
