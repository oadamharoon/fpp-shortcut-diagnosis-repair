"""
Snapshot-ensemble conformal UQ for the UNet baseline.

Mirror of conformal_snapshot.py but for the UNet depth-regression
baseline (UNet/checkpoints_hybrid_l1_a07_depth_individual_normalized).
Same conformal procedure, same conformity-score family (depth std
across N=5 training snapshots), so any difference vs the PhiCalNet
result is attributable to the architecture, not the UQ pipeline.

Differences vs conformal_snapshot.py:
  * Model is UNetFPP (depth regression head, single-channel output).
  * Per-pixel depth from each snapshot = network output (in [0,1]
    normalised space) denormalised by d_min, d_max from the dataset.
  * No phase-unwrap, no calibration layer (UNet predicts depth directly).

Usage:
    python -u conformal_snapshot_unet.py
    python -u conformal_snapshot_unet.py --epochs 100,110,120,130,140
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset

# UNet model lives one level deeper
sys.path.insert(0, str(Path(__file__).parent / "UNet"))
from UNet.unet import UNetFPP

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CKPT_DIR = (BASE / "FPP-ML-Benchmarking/UNet"
                 / "checkpoints_hybrid_l1_a07_depth_individual_normalized")

K_MAX = 32  # not used here, just for PhaseDataset signature


def load_snapshots(epochs, device, dropout_rate=0.5):
    """Load a list of UNetFPP snapshots from CKPT_DIR.

    Note on dropout: the existing UNet baseline was trained with
    dropout_rate=0.5 on the bottleneck. We load it but call .eval(), so
    dropout is OFF at inference -- matching how `evaluate_*.py` would use
    these checkpoints. This keeps the snapshot-std purely epistemic
    (cross-snapshot disagreement), not stochastic from dropout."""
    models, paths = [], []
    for ep in epochs:
        p = CKPT_DIR / f"checkpoint_epoch_{ep:04d}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Snapshot not found: {p}")
        m = UNetFPP(in_channels=1, out_channels=1, dropout_rate=dropout_rate)
        ckpt = torch.load(p, map_location=device)
        # The UNet trainer saves under 'model_state_dict' or sometimes the raw
        # state_dict; try both.
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        m.load_state_dict(sd)
        m = m.to(device).eval()
        models.append(m)
        paths.append(p)
    print(f"Loaded {len(models)} UNet snapshots: {[p.name for p in paths]}")
    return models


def collect_snapshot_predictions(models, dataset, device, desc):
    """For each sample run all snapshots, denormalise to mm, return
    (residual, depth_std, mask, name) per sample. depth_mean from the
    snapshot ensemble is used as the point prediction."""
    residuals, depth_stds, masks, names = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc):
            s = dataset[i]
            fringe = s["fringe"].unsqueeze(0).to(device)
            mask = s["mask_gt"].squeeze().bool().numpy()
            depth_gt = s["depth_gt"].squeeze().numpy()
            d_min = float(s["d_min"])
            d_max = float(s["d_max"])

            depths = []
            for m in models:
                pred_norm = m(fringe).squeeze().cpu().numpy()  # in [0, 1] approx
                depth_mm = pred_norm * (d_max - d_min) + d_min
                depths.append(depth_mm)
            depths = np.stack(depths, axis=0)  # (N, H, W)

            depth_mean = depths.mean(axis=0)
            depth_std = depths.std(axis=0)

            residuals.append(np.abs(depth_mean - depth_gt))
            depth_stds.append(depth_std)
            masks.append(mask)
            names.append(s["name"])
    return residuals, depth_stds, masks, names


def conformity_score(depth_std_pixel, mode, scale_floor=0.05):
    if mode == "std":
        return np.maximum(depth_std_pixel, scale_floor)
    if mode == "std_log":
        return np.log1p(depth_std_pixel / max(scale_floor, 1e-6))
    raise ValueError(mode)


def calibrate(residuals, depth_stds, masks, mode, alphas, **kw):
    rs = []
    for r, d, m in zip(residuals, depth_stds, masks):
        s = conformity_score(d[m], mode, **kw)
        rs.append(r[m] / s)
    rs = np.concatenate(rs)
    n = len(rs)
    quantiles = {}
    for alpha in alphas:
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(k, n)
        q = float(np.partition(rs, k - 1)[k - 1])
        quantiles[alpha] = q
    return quantiles, n


def evaluate_coverage(residuals, depth_stds, masks, mode, q_alpha, **kw):
    covered_total, n_total = 0, 0
    widths_per_sample = []
    for r, d, m in zip(residuals, depth_stds, masks):
        s = conformity_score(d[m], mode, **kw)
        widths_per_sample.append(2.0 * q_alpha * s)
        covered_total += int((r[m] <= q_alpha * s).sum())
        n_total += int(m.sum())
    widths = np.concatenate(widths_per_sample)
    return {
        "coverage": covered_total / max(n_total, 1),
        "median_width": float(np.median(widths)),
        "mean_width": float(widths.mean()),
        "p90_width": float(np.percentile(widths, 90)),
        "p99_width": float(np.percentile(widths, 99)),
        "n": n_total,
    }


def uncertainty_rejected_metrics(residuals, depth_stds, masks, mode,
                                 fractions, **kw):
    all_r, all_s = [], []
    for r, d, m in zip(residuals, depth_stds, masks):
        s = conformity_score(d[m], mode, **kw)
        all_r.append(r[m])
        all_s.append(s)
    all_r = np.concatenate(all_r)
    all_s = np.concatenate(all_s)
    out = []
    for frac in fractions:
        if frac <= 0:
            kept = np.ones_like(all_r, dtype=bool)
        else:
            thr = np.quantile(all_s, 1.0 - frac, method="higher")
            kept = all_s < thr
        if kept.sum() == 0:
            out.append({"frac": frac, "mae": float("nan"),
                        "rmse": float("nan"), "n_kept": 0})
            continue
        kr = all_r[kept]
        out.append({
            "frac": frac, "mae": float(kr.mean()),
            "rmse": float(np.sqrt((kr ** 2).mean())),
            "n_kept": int(kept.sum()),
        })
    return out


def compute_spearman(residuals, depth_stds, masks, mode, **kw):
    all_r, all_s = [], []
    for r, d, m in zip(residuals, depth_stds, masks):
        s = conformity_score(d[m], mode, **kw)
        all_r.append(r[m])
        all_s.append(s)
    all_r = np.concatenate(all_r)
    all_s = np.concatenate(all_s)
    if np.allclose(all_s, all_s[0]):
        return float("nan")
    rho, _ = spearmanr(all_s, all_r)
    return float(rho)


def oracle_rejected_metrics(residuals, masks, thresholds_mm):
    all_r = np.concatenate([r[m] for r, m in zip(residuals, masks)])
    out = []
    for thr in thresholds_mm:
        kept = all_r < thr
        kr = all_r[kept]
        out.append({
            "thr_mm": thr,
            "mae":  float(kr.mean()) if kr.size else float("nan"),
            "rmse": float(np.sqrt((kr ** 2).mean())) if kr.size else float("nan"),
            "n_kept": int(kept.sum()),
            "frac_kept": float(kept.mean()),
        })
    return out


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = [int(e) for e in args.epochs.split(",") if e.strip()]
    print(f"UNet snapshot epochs: {epochs}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    models = load_snapshots(epochs, device, dropout_rate=args.dropout)

    val_ds = PhaseDataset(DATA_ROOT / "val/fringe",  PHASE_ROOT / "val",
                          K_max=K_MAX)
    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test",
                           K_max=K_MAX)

    val_r, val_d, val_m, _ = collect_snapshot_predictions(
        models, val_ds, device, "calibration (val)")
    test_r, test_d, test_m, _ = collect_snapshot_predictions(
        models, test_ds, device, "test")

    print("\n" + "=" * 78)
    print(f"  UNet snapshot ensemble of {len(models)} models, depth-std as score")
    print("=" * 78)
    print("\nSpearman rank corr (depth_std, |residual|) on test:")
    for mode in ("std", "std_log"):
        rho = compute_spearman(test_r, test_d, test_m, mode,
                               scale_floor=args.scale_floor)
        print(f"  {mode:<8s}: {rho:+.4f}")

    alphas = [0.10, 0.05, 0.01]
    summary = {}

    for mode in ("std", "std_log"):
        kw = {"scale_floor": args.scale_floor}
        print("\n" + "=" * 78)
        print(f"  Conformal prediction (UNet, mode='{mode}')")
        print("=" * 78)

        quantiles, n_cal = calibrate(val_r, val_d, val_m, mode, alphas, **kw)
        print(f"Calibration set: {n_cal:,} object pixels (val)")
        for alpha in alphas:
            print(f"  q_(1-alpha)={1 - alpha:.2f}: {quantiles[alpha]:.4f}")

        rho = compute_spearman(test_r, test_d, test_m, mode, **kw)
        print(f"Spearman rank corr: {rho:+.4f}")

        summary[mode] = {}
        for alpha in alphas:
            q = quantiles[alpha]
            metrics = evaluate_coverage(test_r, test_d, test_m, mode, q, **kw)
            summary[mode][alpha] = metrics
            print(f"\nTest @ target {1 - alpha:.2f} (q={q:.4f}):")
            print(f"  empirical coverage : {100 * metrics['coverage']:7.3f} %  "
                  f"(target: {100 * (1 - alpha):.1f} %)")
            print(f"  median / mean width: {metrics['median_width']:7.3f} / "
                  f"{metrics['mean_width']:7.3f} mm")
            print(f"  p90 / p99 width    : {metrics['p90_width']:7.3f} / "
                  f"{metrics['p99_width']:7.3f} mm")

        print("\nUncertainty-based rejection (drop top-k% by snapshot-std):")
        for r in uncertainty_rejected_metrics(
                test_r, test_d, test_m, mode,
                [0.0, 0.001, 0.005, 0.01, 0.05, 0.103 / 100], **kw):
            label = ("no rejection      " if r["frac"] == 0
                     else f"drop top {100 * r['frac']:5.3f}%")
            if np.isnan(r["mae"]):
                print(f"  {label}: (no pixels remaining)")
            else:
                print(f"  {label}: MAE={r['mae']:7.3f} mm   "
                      f"RMSE={r['rmse']:7.3f} mm   (n={r['n_kept']:,})")

    print("\n" + "=" * 78)
    print("  Oracle outlier exclusion (uses GT |error|, NOT a deployable signal)")
    print("=" * 78)
    for r in oracle_rejected_metrics(test_r, test_m, [1e9, 500, 100, 50, 10]):
        if r["thr_mm"] >= 1e9:
            label = "no rejection         "
        else:
            label = f"drop |err| > {r['thr_mm']:>3.0f} mm "
        print(f"  {label}: MAE={r['mae']:7.3f} mm   RMSE={r['rmse']:7.3f} mm   "
              f"({100 * r['frac_kept']:.3f}% kept)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=str, default="100,110,120,130,140",
                   help="Snapshot epochs (last 5 by default)")
    p.add_argument("--scale_floor", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5,
                   help="dropout_rate the UNet was trained with (just so "
                        "state_dict loads; eval mode disables it)")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    main(args)
