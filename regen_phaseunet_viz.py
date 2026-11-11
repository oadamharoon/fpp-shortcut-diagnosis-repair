"""
Regenerate PhaseUNet test-sample visualizations as two row-based PNGs per
sample:

  {prefix}_{sample}_depth_row.png -- 1x4 row:
      Fringe Pattern | Ground Truth Depth (mm) | Predicted Depth (mm) |
      Absolute Error (mm)

  {prefix}_{sample}_phase_row.png -- 1x4 row:
      sin(phi) GT | sin(phi) Predicted | cos(phi) GT | cos(phi) Predicted

Styling matches vis_pred.py: bold titles at fontsize 16, colorbar ticks at
14, colorbar labels at 16.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

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

# Font sizes — match vis_pred.py conventions for paper figures.
TITLE_FS       = 16
CBAR_LABEL_FS  = 16
CBAR_TICK_FS   = 14


def full_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def _draw_panel(ax, data, cmap, vmin, vmax, title,
                cbar_label=None, show_cbar=True, bold_title=False,
                cbar_label_fs=None):
    """Draw an image panel with a top title and (optionally) a colorbar."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontsize=TITLE_FS,
                 fontweight=('bold' if bold_title else 'normal'))
    ax.axis('off')
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
        if cbar_label is not None:
            cbar.set_label(cbar_label,
                           fontsize=cbar_label_fs or CBAR_LABEL_FS)
    return im


def regen_sample(sample, model, cal_layer, save_dir, prefix, device):
    name = sample['name']
    fringe = sample['fringe'].squeeze().numpy()
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

    uph = full_unwrap(sin_hat, cos_hat, k_gt)
    uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
    depth_pred = cal_layer(uph_t).squeeze().cpu().numpy()
    error = np.abs(depth_pred - depth_gt)

    # Mask out background for object-centric panels
    nan_mask = np.where(mask, 1.0, np.nan)
    depth_gt_vis = depth_gt * nan_mask
    depth_pred_vis = depth_pred * nan_mask
    error_vis = error * nan_mask
    sin_gt_vis = sin_gt * nan_mask
    cos_gt_vis = cos_gt * nan_mask
    sin_pred_vis = sin_hat * nan_mask
    cos_pred_vis = cos_hat * nan_mask

    # Per-sample metrics (used in titles / console summary)
    obj_err = error[mask]
    mae = float(obj_err.mean())
    rmse = float(np.sqrt((obj_err ** 2).mean()))
    err_vmax = float(np.percentile(obj_err, 95)) if obj_err.size > 0 else 1.0

    # Shared depth colormap range across GT and Pred
    depth_vmin = float(np.nanmin([depth_gt_vis, depth_pred_vis]))
    depth_vmax = float(np.nanmax([depth_gt_vis, depth_pred_vis]))

    # ------- Row 1: depth set (Fringe | GT Depth | Pred Depth | |error|)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _draw_panel(axes[0], fringe, cmap='gray',
                vmin=float(np.min(fringe)), vmax=float(np.max(fringe)),
                title='Fringe Pattern')
    _draw_panel(axes[1], depth_gt_vis, cmap='viridis',
                vmin=depth_vmin, vmax=depth_vmax,
                title='Ground Truth Depth (mm)')
    _draw_panel(axes[2], depth_pred_vis, cmap='viridis',
                vmin=depth_vmin, vmax=depth_vmax,
                title='Predicted Depth (mm)')
    _draw_panel(axes[3], error_vis, cmap='hot',
                vmin=0.0, vmax=err_vmax,
                title=(f'Absolute Error (mm)\n'
                       f'Object MAE: {mae:.2f}, RMSE: {rmse:.2f}'),
                cbar_label=f'Error (mm, 95th %ile: {err_vmax:.1f})')
    plt.tight_layout()
    depth_path = save_dir / f"{prefix}_{name}_depth_row.png"
    plt.savefig(depth_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------- Row 2: wrapped-phase set with per-component error maps
    # (sin GT | sin Pred | sin error | cos GT | cos Pred | cos error)
    # Error = pred - gt on a diverging 'seismic' cmap with a fixed
    # symmetric range [-1, +1]. Tiny errors render near-white (honestly
    # matching the visual GT vs Pred comparison); wrap-boundary sign flips
    # stand out as deep red/blue. The colorbar label reports the min and
    # max error over the object mask for reference.
    sin_err = sin_hat - sin_gt
    cos_err = cos_hat - cos_gt
    sin_err_vis = sin_err * nan_mask
    cos_err_vis = cos_err * nan_mask
    sin_err_obj = sin_err[mask]
    cos_err_obj = cos_err[mask]
    sin_lo = float(sin_err_obj.min()) if sin_err_obj.size > 0 else 0.0
    sin_hi = float(sin_err_obj.max()) if sin_err_obj.size > 0 else 0.0
    cos_lo = float(cos_err_obj.min()) if cos_err_obj.size > 0 else 0.0
    cos_hi = float(cos_err_obj.max()) if cos_err_obj.size > 0 else 0.0
    SC_ERR_RNG = 1.0  # symmetric scale: |error| = 1 already implies sign flip

    # figsize matches the depth row (20" x 5") for side-by-side consistency
    # in the paper layout. With 6 panels in 20" each slot is ~3.33" wide, so
    # the error-colorbar label gets a smaller font to stay inside bounds.
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    CBAR_ERR_FS = 11  # smaller than CBAR_LABEL_FS (16) for the tight layout
    _draw_panel(axes[0], sin_gt_vis,   cmap='RdBu', vmin=-1, vmax=1,
                title=r'$\sin\phi$ Ground Truth',
                show_cbar=False)
    _draw_panel(axes[1], sin_pred_vis, cmap='RdBu', vmin=-1, vmax=1,
                title=r'$\sin\phi$ Predicted',
                show_cbar=False)
    _draw_panel(axes[2], sin_err_vis,  cmap='seismic',
                vmin=-SC_ERR_RNG, vmax=SC_ERR_RNG,
                title=r'$\sin\phi$ Error (pred $-$ gt)',
                cbar_label=f'Error (min: {sin_lo:+.3f}, max: {sin_hi:+.3f})',
                cbar_label_fs=CBAR_ERR_FS)
    _draw_panel(axes[3], cos_gt_vis,   cmap='RdBu', vmin=-1, vmax=1,
                title=r'$\cos\phi$ Ground Truth',
                show_cbar=False)
    _draw_panel(axes[4], cos_pred_vis, cmap='RdBu', vmin=-1, vmax=1,
                title=r'$\cos\phi$ Predicted',
                show_cbar=False)
    _draw_panel(axes[5], cos_err_vis,  cmap='seismic',
                vmin=-SC_ERR_RNG, vmax=SC_ERR_RNG,
                title=r'$\cos\phi$ Error (pred $-$ gt)',
                cbar_label=f'Error (min: {cos_lo:+.3f}, max: {cos_hi:+.3f})',
                cbar_label_fs=CBAR_ERR_FS)
    plt.tight_layout()
    phase_path = save_dir / f"{prefix}_{name}_phase_row.png"
    plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    n_out100 = int((obj_err > 100).sum())
    pct_out100 = 100 * n_out100 / max(obj_err.size, 1)
    print(f"{name}: Obj MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  "
          f"outliers>100mm={n_out100} ({pct_out100:.2f}%)  "
          f"-> {depth_path.name}, {phase_path.name}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    # Build name -> index lookup
    name_to_idx = {}
    for i in range(len(test_ds)):
        name_to_idx[test_ds[i]['name']] = i

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    targets = args.samples.split(',')
    for target in targets:
        target = target.strip()
        if target not in name_to_idx:
            print(f"WARN: sample '{target}' not in test split; skipping")
            continue
        regen_sample(test_ds[name_to_idx[target]], model, cal_layer,
                     save_dir, args.prefix, device)
    print(f"\nPanels saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_phase_A1_refined_v2/best_model.pth")
    parser.add_argument("--save_dir", type=str,
                        default="visualizations_phase_A1_refined_v2/paper_panels")
    parser.add_argument("--samples", type=str,
                        default="container_bottle_A180,power_drill_A0",
                        help="Comma-separated list of sample names to render")
    parser.add_argument("--prefix", type=str, default="phaseunet",
                        help="Filename prefix for each panel")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
