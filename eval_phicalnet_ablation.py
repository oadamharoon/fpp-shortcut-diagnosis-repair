"""
Evaluate a PhiCalNet checkpoint (reference or ablation) on the 30-sample test
split and emit the metrics needed for the ablation table.

The depth pipeline and metric definitions are identical to
evaluate_phase_A1_v2_outliers.py (the script behind the paper's 4.46 mm /
Table 14 numbers): single-shot model -> sin/cos -> full_unwrap with oracle k
-> calibration layer -> raw depth (no post-processing).

Metrics produced:
  - Object MAE / RMSE (mm): mean over the per-sample object MAE / RMSE values
    (matches Table 8).
  - Pixel MAE / RMSE and the 50 / 90 / 99 / 99.9 percentile pixel errors over
    all ~2 M object pixels concatenated across the 30 samples (matches Table 14).
  - Count and percentage of object pixels with |error| > 100 mm (Section 5.6).

Outputs:
  <out_dir>/<label>_metrics.json        all scalar metrics
  <out_dir>/<label>_pixel_errors.npy    concatenated per-pixel |error| (mm)

Usage:
    python eval_phicalnet_ablation.py --label reference \
        --checkpoint checkpoints_phase_A1_refined_v2/best_model.pth
    python eval_phicalnet_ablation.py --label drop_depth \
        --checkpoint checkpoints_phicalnet_ablation_depth/best_model.pth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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

K_MAX = 32
T_P = 36.0
PERCENTILES = [50, 90, 99, 99.9]


def full_unwrap(sin_np, cos_np, k_np):
    """Identical to evaluate_phase_A1_v2_outliers.full_unwrap."""
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[{args.label}] loading {args.checkpoint}")
    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)

    per_sample_mae, per_sample_rmse = [], []
    all_errs = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc=f"Eval {args.label}"):
            sample = test_ds[idx]
            fringe = sample['fringe'].unsqueeze(0).to(device)
            k_gt_np = sample['k_gt'].numpy()
            mask = sample['mask_gt'].squeeze().numpy() > 0
            depth_gt = sample['depth_gt'].numpy().squeeze()

            out = model(fringe)
            sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)

            uph = full_unwrap(sin_hat, cos_hat, k_gt_np)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth = cal_layer(uph_t).squeeze().cpu().numpy()

            err = np.abs(depth - depth_gt)[mask]
            per_sample_mae.append(float(err.mean()))
            per_sample_rmse.append(float(np.sqrt((err ** 2).mean())))
            all_errs.append(err)

    all_errs = np.concatenate(all_errs)
    n_pix = int(all_errs.size)
    n_over = int((all_errs > 100).sum())

    metrics = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "n_samples": len(test_ds),
        "n_object_pixels": n_pix,
        # Table 8: mean over per-sample values
        "object_mae_mm": float(np.mean(per_sample_mae)),
        "object_rmse_mm": float(np.mean(per_sample_rmse)),
        # Table 14: pixel-wise over all object pixels
        "pixel_mae_mm": float(all_errs.mean()),
        "pixel_rmse_mm": float(np.sqrt((all_errs ** 2).mean())),
        "pixel_pctl_mm": {str(p): float(np.percentile(all_errs, p)) for p in PERCENTILES},
        "pixel_max_mm": float(all_errs.max()),
        # Section 5.6: wrap-boundary tail
        "n_err_gt_100mm": n_over,
        "pct_err_gt_100mm": float(100.0 * n_over / max(n_pix, 1)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}_metrics.json"
    npy_path = out_dir / f"{args.label}_pixel_errors.npy"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(npy_path, all_errs.astype(np.float32))

    print(f"[{args.label}] Obj MAE={metrics['object_mae_mm']:.2f} mm  "
          f"Obj RMSE={metrics['object_rmse_mm']:.2f} mm | "
          f"Pix MAE={metrics['pixel_mae_mm']:.2f}  "
          f"p99.9={metrics['pixel_pctl_mm']['99.9']:.1f} | "
          f">100mm: {n_over:,} ({metrics['pct_err_gt_100mm']:.3f}%)")
    print(f"[{args.label}] wrote {json_path}  and  {npy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--label", required=True, type=str,
                        help="Row label, e.g. reference / drop_circular / drop_grad / drop_depth")
    parser.add_argument("--out_dir", type=str, default="ablation_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
