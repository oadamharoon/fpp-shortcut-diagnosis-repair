"""
Visualization for Phase-Intermediate variants.

For each test sample, produces a grid showing:
  fringe | depth_gt | depth_pred | abs_error |
  sin_gt | sin_pred | cos_gt | cos_pred |
  (k_gt | k_pred for A2)

Errors clipped at 95th percentile for visibility; outlier pixels shown
separately with a mask overlay.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import median_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer import CalibrationLayer
from data.phase_dataset import PhaseDataset

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

T_P = 36.0
K_MAX = 32


def full_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def visualize_sample(sample, model, cal_layer, variant, save_dir, device):
    name = sample['name']
    fringe_np = sample['fringe'].squeeze().numpy()
    sin_gt = sample['sin_phi_gt'].squeeze().numpy()
    cos_gt = sample['cos_phi_gt'].squeeze().numpy()
    k_gt = sample['k_gt'].numpy()
    mask = sample['mask_gt'].squeeze().numpy() > 0
    depth_gt = sample['depth_gt'].squeeze().numpy()

    fringe_t = sample['fringe'].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(fringe_t)

    sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
    cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)

    if variant == 'A1':
        k_use = k_gt
    else:
        k_use = out['order_logits'].argmax(dim=1).squeeze().cpu().numpy()

    uph = full_unwrap(sin_hat, cos_hat, k_use)
    uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
    depth_pred = cal_layer(uph_t).squeeze().cpu().numpy()

    error = np.abs(depth_pred - depth_gt)

    # Per-sample object error stats
    obj_err = error[mask]
    obj_mae = float(obj_err.mean()) if len(obj_err) > 0 else 0.0
    obj_rmse = float(np.sqrt((obj_err ** 2).mean())) if len(obj_err) > 0 else 0.0
    # Outlier threshold: errors > 100mm are "fringe wrap misclassifications"
    outlier_mask = mask & (error > 100)
    n_outlier = outlier_mask.sum()
    outlier_pct = (n_outlier / max(mask.sum(), 1)) * 100

    # Mask visualizations: show only object pixels for depth/error
    depth_gt_vis = np.where(mask, depth_gt, np.nan)
    depth_pred_vis = np.where(mask, depth_pred, np.nan)
    error_vis = np.where(mask, error, np.nan)
    sin_gt_vis = np.where(mask, sin_gt, np.nan)
    cos_gt_vis = np.where(mask, cos_gt, np.nan)
    sin_pred_vis = np.where(mask, sin_hat, np.nan)
    cos_pred_vis = np.where(mask, cos_hat, np.nan)

    # Depth colormap: use same range for GT and pred
    depth_vmin = float(np.nanmin([depth_gt_vis, depth_pred_vis]))
    depth_vmax = float(np.nanmax([depth_gt_vis, depth_pred_vis]))
    # Error colormap: cap at 95th percentile for visibility
    err_vmax = float(np.percentile(obj_err, 95)) if len(obj_err) > 0 else 1.0

    # Layout: 2 rows (fringe/depth row and sin/cos row), + k row for A2
    n_rows = 3 if variant == 'A2' else 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))

    # Row 1: fringe, depth_gt, depth_pred, error
    axes[0, 0].imshow(fringe_np, cmap='gray')
    axes[0, 0].set_title('Input fringe')
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(depth_gt_vis, cmap='viridis', vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 1].set_title(f'Depth GT (mm)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(depth_pred_vis, cmap='viridis', vmin=depth_vmin, vmax=depth_vmax)
    axes[0, 2].set_title(f'Depth Pred (mm)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[0, 3].imshow(error_vis, cmap='hot', vmin=0, vmax=err_vmax)
    axes[0, 3].set_title(f'|error| (mm, clip 95%)')
    axes[0, 3].axis('off')
    # Overlay outlier pixels in cyan
    if n_outlier > 0:
        outlier_overlay = np.zeros((*outlier_mask.shape, 4))
        outlier_overlay[outlier_mask] = [0, 1, 1, 1]  # cyan
        axes[0, 3].imshow(outlier_overlay)
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 2: sin_gt, sin_pred, cos_gt, cos_pred
    axes[1, 0].imshow(sin_gt_vis, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_title('sin(phi) GT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sin_pred_vis, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('sin(phi) Pred')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cos_gt_vis, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('cos(phi) GT')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(cos_pred_vis, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 3].set_title('cos(phi) Pred')
    axes[1, 3].axis('off')

    # Row 3 (A2 only): k_gt, k_pred, k_error, k_correct_mask
    if variant == 'A2':
        k_gt_vis = np.where(mask, k_gt.astype(float), np.nan)
        k_pred_vis = np.where(mask, k_use.astype(float), np.nan)
        k_diff = (k_use - k_gt).astype(float)
        k_diff_vis = np.where(mask, k_diff, np.nan)
        k_correct_vis = np.where(mask, (k_use == k_gt).astype(float), np.nan)

        vmax_k = max(float(np.nanmax(k_gt_vis)), float(np.nanmax(k_pred_vis)))
        vmin_k = min(float(np.nanmin(k_gt_vis)), float(np.nanmin(k_pred_vis)))

        im = axes[2, 0].imshow(k_gt_vis, cmap='tab20', vmin=vmin_k, vmax=vmax_k)
        axes[2, 0].set_title('k GT')
        axes[2, 0].axis('off')
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

        im = axes[2, 1].imshow(k_pred_vis, cmap='tab20', vmin=vmin_k, vmax=vmax_k)
        axes[2, 1].set_title('k Pred')
        axes[2, 1].axis('off')
        plt.colorbar(im, ax=axes[2, 1], fraction=0.046)

        im = axes[2, 2].imshow(k_diff_vis, cmap='seismic', vmin=-5, vmax=5)
        axes[2, 2].set_title('k Pred - k GT')
        axes[2, 2].axis('off')
        plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

        axes[2, 3].imshow(k_correct_vis, cmap='RdYlGn', vmin=0, vmax=1)
        k_acc = float((k_use == k_gt)[mask].mean()) * 100 if mask.sum() > 0 else 0.0
        axes[2, 3].set_title(f'k Correct (acc={k_acc:.1f}%)')
        axes[2, 3].axis('off')

    plt.suptitle(
        f'{name}  |  Obj MAE: {obj_mae:.2f} mm, Obj RMSE: {obj_rmse:.2f} mm  |  '
        f'Outliers (err>100mm): {n_outlier} ({outlier_pct:.2f}%)',
        fontsize=14, y=0.99
    )
    plt.tight_layout()
    save_path = save_dir / f'{name}_viz.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = PhaseUNetFPP(variant=args.variant, in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(test_ds)), desc="Visualizing"):
        visualize_sample(test_ds[idx], model, cal_layer, args.variant, save_dir, device)

    print(f"\n{len(test_ds)} visualizations saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=['A1', 'A2'])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
