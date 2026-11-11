"""
Generate a pixel-wise error histogram for PhaseUNet aggregated over the
test split. The distribution spans 0 to ~1500 mm with 99.9th percentile
at 103 mm, so we use a log-x axis to show both the bulk and the tail.
Vertical markers indicate the 100 mm and 500 mm outlier thresholds
reported in the paper.
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


def full_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def collect_errors(model, cal_layer, test_ds, device):
    all_errs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            fringe = s['fringe'].unsqueeze(0).to(device)
            out = model(fringe)
            sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            k_gt = s['k_gt'].numpy()
            mask = s['mask_gt'].squeeze().bool().numpy()
            depth_gt = s['depth_gt'].squeeze().numpy()
            uph = full_unwrap(sin_hat, cos_hat, k_gt)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth_pred = cal_layer(uph_t).squeeze().cpu().numpy()
            all_errs.append(np.abs(depth_pred - depth_gt)[mask])
    return np.concatenate(all_errs)


def plot_histogram(errs, save_path):
    # Bin edges: log-spaced from 0.01 to max, clipped
    eps = 0.01
    errs_clipped = np.clip(errs, eps, None)
    max_err = float(np.max(errs))
    bins = np.logspace(np.log10(eps), np.log10(max_err), 80)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(errs_clipped, bins=bins, color='#2b6cb0', edgecolor='white',
            linewidth=0.3, alpha=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Per-pixel absolute error (mm)', fontsize=11)
    ax.set_ylabel('Pixel count (log)', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)

    # Annotate thresholds
    for thr, color, label in [
        (100.0, '#dd6b20', '100 mm'),
        (500.0, '#c53030', '500 mm'),
    ]:
        ax.axvline(thr, color=color, linestyle='--', linewidth=1.2,
                   label=f'{label} threshold')

    # Annotate percentiles
    p50 = np.percentile(errs, 50)
    p90 = np.percentile(errs, 90)
    p999 = np.percentile(errs, 99.9)
    ax.axvline(p50,  color='gray', linestyle=':', linewidth=1.0)
    ax.axvline(p90,  color='gray', linestyle=':', linewidth=1.0)
    ax.axvline(p999, color='gray', linestyle=':', linewidth=1.0)

    # Annotations (text placement at top of plot)
    ymax = ax.get_ylim()[1]
    for p, label, ha in [
        (p50,  f'median\n{p50:.2f} mm',  'left'),
        (p90,  f'p90\n{p90:.2f} mm',     'left'),
        (p999, f'p99.9\n{p999:.1f} mm',  'right'),
    ]:
        ax.text(p, ymax * 0.5, label, fontsize=8, color='gray',
                ha=ha, va='top', rotation=90)

    ax.legend(fontsize=9, loc='upper right', frameon=True)
    ax.grid(which='both', axis='both', alpha=0.15)
    ax.set_xlim(eps, max_err * 1.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also print tail statistics
    print(f"\nPhaseUNet per-pixel error distribution summary:")
    print(f"  Total pixels: {len(errs):,}")
    print(f"  MAE  (pixel-wise): {errs.mean():.3f} mm")
    print(f"  RMSE (pixel-wise): {np.sqrt((errs**2).mean()):.3f} mm")
    print(f"  50th percentile:   {np.percentile(errs, 50):.3f} mm")
    print(f"  90th percentile:   {np.percentile(errs, 90):.3f} mm")
    print(f"  99th percentile:   {np.percentile(errs, 99):.3f} mm")
    print(f"  99.9th percentile: {np.percentile(errs, 99.9):.3f} mm")
    print(f"  Max:               {errs.max():.3f} mm")
    for thr in [10, 50, 100, 500]:
        n = int((errs > thr).sum())
        pct = 100 * n / len(errs)
        print(f"  err > {thr:>3} mm: {n:>8,} ({pct:.3f}%)")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)
    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    errs = collect_errors(model, cal_layer, test_ds, device)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_histogram(errs, save_path)
    print(f"\nHistogram saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_phase_A1_refined_v2/best_model.pth")
    parser.add_argument("--save_path", type=str,
                        default="visualizations_phase_A1_refined_v2/phaseunet_error_histogram.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
