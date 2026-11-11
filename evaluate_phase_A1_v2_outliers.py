"""
Post-hoc outlier removal sweep for A1 refined v2.

The Obj RMSE (17-18 mm) is much larger than Obj MAE (4.46 mm), which tells us
the error distribution has a long tail — most likely wrap-boundary pixels
where the predicted fringe order k disagrees with neighbors. This script
sweeps a few cheap post-processing strategies on the *unwrapped-and-calibrated*
depth map and reports the effect on Obj MAE / RMSE.

Strategies:
  1. Plain median filter on depth (k=3, 5, 7) — baseline smoothing
  2. MAD-based outlier replacement (robust): flag pixels where
     |depth - local_median| > k * local_MAD; replace those with local median
  3. Absolute threshold: flag pixels where |depth - local_median| > T mm;
     replace with local median
"""
import argparse
import csv
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


def full_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def mad_outlier_replace(depth, mask, win=5, k_mad=5.0):
    """Replace pixels where |depth - local_median| > k_mad * local_MAD."""
    d = depth.copy()
    local_med = median_filter(d, size=win)
    abs_dev = np.abs(d - local_med)
    local_mad = median_filter(abs_dev, size=win) + 1e-6
    outlier = mask & (abs_dev > k_mad * local_mad)
    d[outlier] = local_med[outlier]
    return d, int(outlier.sum())


def abs_outlier_replace(depth, mask, win=5, thresh_mm=25.0):
    """Replace pixels where |depth - local_median| > thresh_mm."""
    d = depth.copy()
    local_med = median_filter(d, size=win)
    outlier = mask & (np.abs(d - local_med) > thresh_mm)
    d[outlier] = local_med[outlier]
    return d, int(outlier.sum())


def metrics(depth_pred, depth_gt, mask):
    err = np.abs(depth_pred - depth_gt)
    obj = err[mask]
    return float(obj.mean()), float(np.sqrt((obj ** 2).mean()))


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Loading A1 model from {args.checkpoint}")
    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)

    # Strategies to sweep
    strategies = [
        ("raw", None),
        ("median_k3", lambda d, m: (median_filter(d, size=3), 0)),
        ("median_k5", lambda d, m: (median_filter(d, size=5), 0)),
        ("median_k7", lambda d, m: (median_filter(d, size=7), 0)),
        ("mad_k3.0_w5", lambda d, m: mad_outlier_replace(d, m, win=5, k_mad=3.0)),
        ("mad_k5.0_w5", lambda d, m: mad_outlier_replace(d, m, win=5, k_mad=5.0)),
        ("abs_10mm_w5", lambda d, m: abs_outlier_replace(d, m, win=5, thresh_mm=10.0)),
        ("abs_25mm_w5", lambda d, m: abs_outlier_replace(d, m, win=5, thresh_mm=25.0)),
        ("abs_50mm_w5", lambda d, m: abs_outlier_replace(d, m, win=5, thresh_mm=50.0)),
    ]

    results = {name: {'mae': [], 'rmse': [], 'n_replaced': []} for name, _ in strategies}
    per_sample_rows = []
    all_errs_raw = []  # pixel-wise concat for percentile / exclusion analysis

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
            sample = test_ds[idx]
            fringe = sample['fringe'].unsqueeze(0).to(device)
            k_gt_np = sample['k_gt'].numpy()
            mask = sample['mask_gt'].squeeze().numpy() > 0
            depth_gt = sample['depth_gt'].numpy().squeeze()
            name = sample['name']

            out = model(fringe)
            sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)

            uph = full_unwrap(sin_hat, cos_hat, k_gt_np)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth_raw = cal_layer(uph_t).squeeze().cpu().numpy()

            all_errs_raw.append(np.abs(depth_raw - depth_gt)[mask])

            row = {'name': name}
            for sname, fn in strategies:
                if fn is None:
                    d_clean, n_rep = depth_raw, 0
                else:
                    d_clean, n_rep = fn(depth_raw, mask)
                mae, rmse = metrics(d_clean, depth_gt, mask)
                results[sname]['mae'].append(mae)
                results[sname]['rmse'].append(rmse)
                results[sname]['n_replaced'].append(n_rep)
                row[f'{sname}_mae'] = mae
                row[f'{sname}_rmse'] = rmse
                row[f'{sname}_n_repl'] = n_rep
            per_sample_rows.append(row)

    # -------- Pixel-wise error distribution (raw, no post-processing) --------
    all_errs_raw = np.concatenate(all_errs_raw)
    print("\n" + "=" * 78)
    print("Pixel-wise Error Distribution (raw output, no post-processing)")
    print("=" * 78)
    print(f"Total object pixels: {len(all_errs_raw):,}")
    print(f"MAE:  {all_errs_raw.mean():.3f} mm")
    print(f"RMSE: {np.sqrt((all_errs_raw ** 2).mean()):.3f} mm")
    print()
    print("Percentiles of per-pixel error:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]:
        print(f"  {p:5.2f}%: {np.percentile(all_errs_raw, p):.3f} mm")
    print(f"  Max:     {all_errs_raw.max():.3f} mm")
    print()
    for t in [10, 50, 100, 500]:
        n = int((all_errs_raw > t).sum())
        pct = float((all_errs_raw > t).mean()) * 100
        print(f"Pixels with err > {t:>3} mm: {n:>8,} ({pct:.3f}%)")
    print()
    print("Oracle exclusion (ceiling — shows what's achievable if we knew which px to drop):")
    for t in [100, 200, 500]:
        keep = all_errs_raw <= t
        if keep.sum() > 0:
            kept_mae = all_errs_raw[keep].mean()
            kept_rmse = np.sqrt((all_errs_raw[keep] ** 2).mean())
            n_rem = int((~keep).sum())
            pct_rem = float((~keep).mean()) * 100
            print(f"  Exclude err > {t:>3} mm ({n_rem:,} px, {pct_rem:.3f}%): "
                  f"MAE {kept_mae:.3f} | RMSE {kept_rmse:.3f}")

    print("\n" + "=" * 78)
    print("A1 Refined v2 — Post-hoc Outlier Removal Sweep")
    print("=" * 78)
    print(f"{'Strategy':<18} {'Obj MAE':>10} {'Obj RMSE':>10} {'ΔMAE':>8} {'ΔRMSE':>8} {'AvgRepl':>10}")
    print("-" * 78)
    base_mae = np.mean(results['raw']['mae'])
    base_rmse = np.mean(results['raw']['rmse'])
    for sname, _ in strategies:
        mae = np.mean(results[sname]['mae'])
        rmse = np.mean(results[sname]['rmse'])
        n_rep = np.mean(results[sname]['n_replaced'])
        d_mae = mae - base_mae
        d_rmse = rmse - base_rmse
        print(f"{sname:<18} {mae:>10.4f} {rmse:>10.4f} {d_mae:>+8.4f} {d_rmse:>+8.4f} {n_rep:>10.0f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "outlier_sweep.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"\nPer-sample results: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_phase_A1_refined_v2/best_model.pth")
    parser.add_argument("--save_dir", type=str, default="visualizations_phase_A1_refined_v2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
