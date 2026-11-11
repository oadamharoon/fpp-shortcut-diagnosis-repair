"""
Plot a CDF of per-pixel absolute error for PhaseUNet on the test split.

Reads the same data/model as plot_phaseunet_error_histogram.py, but presents
the distribution as: x = error threshold (mm), y = fraction of pixels with
absolute error <= threshold. Easier to read than the log-log histogram for
the paper's oracle-exclusion story.

Two views are saved: the full distribution and a zoom on the bulk (<= 50 mm)
so the knee of the curve is legible.
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


def _draw_cdf(ax, sorted_errs, cdf, xlim, thresholds, annotate_percentiles):
    ax.plot(sorted_errs, cdf, color='#2b6cb0', linewidth=1.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel('Error threshold $T$ (mm)', fontsize=11)
    ax.set_ylabel(r'Fraction of pixels with $|\mathrm{error}| \leq T$',
                  fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.2)

    # Threshold vertical lines with the fraction of pixels below each
    for thr, color, label in thresholds:
        frac = float(np.searchsorted(sorted_errs, thr,
                                     side='right')) / len(sorted_errs)
        ax.axvline(thr, color=color, linestyle='--', linewidth=1.2,
                   label=f'{label}: {frac * 100:.3f}% below')

    # Percentile horizontal lines for a read-at-a-glance summary
    if annotate_percentiles:
        for pct, color in [(0.50, '#555'), (0.90, '#555'), (0.99, '#555')]:
            ax.axhline(pct, color=color, linestyle=':', linewidth=0.9,
                       alpha=0.6)
            idx = int(pct * (len(sorted_errs) - 1))
            x_at_p = sorted_errs[idx]
            label = f'p{int(pct * 100)}: {x_at_p:.2f} mm'
            ax.text(xlim[1] * 0.99, pct + 0.01, label, fontsize=8,
                    color='gray', ha='right', va='bottom')

    ax.legend(fontsize=9, loc='lower right', frameon=True)


def plot_cdf(errs, save_path):
    sorted_errs = np.sort(errs)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    # Left: zoom on the bulk (<= 50 mm) so the knee is visible
    _draw_cdf(
        axes[0], sorted_errs, cdf,
        xlim=(0.0, 50.0),
        thresholds=[],
        annotate_percentiles=True,
    )
    axes[0].set_title('Bulk distribution (0--50 mm)', fontsize=11)

    # Right: full range including the oracle-exclusion thresholds
    _draw_cdf(
        axes[1], sorted_errs, cdf,
        xlim=(0.0, 600.0),
        thresholds=[
            (100.0, '#dd6b20', '100 mm threshold'),
            (500.0, '#c53030', '500 mm threshold'),
        ],
        annotate_percentiles=False,
    )
    axes[1].set_title('Full range with oracle thresholds', fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Text summary for the caption
    print(f"\nPhaseUNet per-pixel error CDF summary:")
    print(f"  Total pixels:   {len(errs):,}")
    print(f"  MAE:            {errs.mean():.3f} mm")
    print(f"  RMSE:           {np.sqrt((errs ** 2).mean()):.3f} mm")
    for pct in [50, 90, 95, 99, 99.9]:
        print(f"  p{pct:>4}:          {np.percentile(errs, pct):.3f} mm")
    print(f"  max:            {errs.max():.3f} mm")
    for thr in [10, 50, 100, 200, 500]:
        n_below = int((errs <= thr).sum())
        print(f"  |err| <= {thr:>3} mm: {n_below:>9,}  "
              f"({100 * n_below / len(errs):.3f}%)")


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
    plot_cdf(errs, save_path)
    print(f"\nCDF saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_phase_A1_refined_v2/best_model.pth")
    parser.add_argument("--save_path", type=str,
                        default="visualizations_phase_A1_refined_v2/phaseunet_error_cdf.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
