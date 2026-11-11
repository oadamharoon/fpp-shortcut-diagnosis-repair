"""
Two diagnostics for the multi-frame ablation:

(1) learned-k sanity check
    For each learned-k checkpoint, pull out the per-pixel argmax k_hat and
    compare against k_gt on the object mask. Reports classification accuracy,
    confusion with off-by-one errors, and per-sample depth MAE distribution.
    This isolates whether the catastrophic depth numbers are driven by
    a soft-argmax bug at inference time, by genuinely-wrong k predictions,
    or by a small number of dominating samples.

(2) noisy-k sensitivity sweep
    Using the existing 3-frame PhiCalNet (learned phi, oracle k), corrupt
    a fraction f of object-pixel k values before unwrap/calibration and
    report the resulting depth MAE/RMSE. Two corruption modes:
        * off-by-one: k_hat(p) = k_gt(p) + delta, delta ~ +/-1 with equal prob
        * uniform:    k_hat(p) = random int in [0, K_max-1]
    Sweeps f over {0.01, 0.05, 0.10, 0.25}. Characterises how much upstream
    k noise the pipeline can absorb without going off a cliff.

Usage:
    python diag_learnedk_and_noisy_k.py                   # both parts
    python diag_learnedk_and_noisy_k.py --only sanity
    python diag_learnedk_and_noisy_k.py --only noisy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from data.phase_dataset_multiframe import PhaseMultiFrameDataset
from models.calibration_layer import CalibrationLayer
from models.phase_unet import PhaseUNetFPP
from models.phicalnet_hybrid import PhiCalHybridNet
from models.phicalnet_learnedk import PhiCalNetLearnedK

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
MULTIFRAME_ROOT = (
    BASE / "training_datasets"
         / "training_data_depth_individual_normalized_multiframe_horizontal_only"
)
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


# ---------------------------------------------------------------------------
# (1) learned-k sanity check
# ---------------------------------------------------------------------------
def _load_hybrid(device):
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    m = PhiCalHybridNet(calibration_layer=cal, frame_indices=(0, 6, 12),
                       K_max=K_MAX, gumbel_temperature=1.0)
    ckpt = torch.load("checkpoints_phicalnet_hybrid/best_model.pth",
                      map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), (0, 6, 12)


def _load_learnedk(device, frames, tag, k_head_variant="default"):
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    m = PhiCalNetLearnedK(calibration_layer=cal, in_channels=len(frames),
                         K_max=K_MAX, k_head_variant=k_head_variant)
    suffix = "" if k_head_variant == "default" else f"_{k_head_variant}"
    ckpt = torch.load(
        f"checkpoints_phicalnet_learnedk_{tag}{suffix}/best_model.pth",
        map_location=device,
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), frames


def sanity_check_model(model, frames, device, name):
    """Diagnose one learned-k checkpoint."""
    ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "test/fringe", PHASE_ROOT / "test",
        K_max=K_MAX, frame_indices=frames,
    )
    per_sample = []
    total_correct = 0
    total_off1 = 0
    total_pixels = 0
    max_abs_err = []

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f"sanity {name}"):
            s = ds[i]
            fringe = s["fringe"].unsqueeze(0).to(device)
            mask = s["mask_gt"].squeeze().bool().numpy()
            k_gt = s["k_gt"].numpy()
            depth_gt = s["depth_gt"].squeeze().numpy()

            out = model(fringe)
            k_hat = out["k_logits"].argmax(dim=1).squeeze().cpu().numpy()
            depth_pred = out["depth_hat"].squeeze().cpu().numpy()

            err = np.abs(depth_pred - depth_gt)[mask]
            per_sample.append({
                "name": s["name"],
                "n_obj": int(mask.sum()),
                "mae": float(err.mean()),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "max_err": float(err.max()),
                "k_acc": float((k_hat[mask] == k_gt[mask]).mean()),
                "k_off1": float((np.abs(k_hat[mask] - k_gt[mask]) == 1).mean()),
                "k_mean_abs_off": float(np.abs(k_hat[mask].astype(int) - k_gt[mask].astype(int)).mean()),
            })
            total_correct += int((k_hat[mask] == k_gt[mask]).sum())
            total_off1 += int((np.abs(k_hat[mask].astype(int) - k_gt[mask].astype(int)) == 1).sum())
            total_pixels += int(mask.sum())
            max_abs_err.append(np.abs(k_hat[mask].astype(int) - k_gt[mask].astype(int)).max())

    print(f"\n=== {name} ===")
    print(f"  Overall k accuracy:       {100 * total_correct / total_pixels:6.2f}%")
    print(f"  Overall k off-by-one:     {100 * total_off1 / total_pixels:6.2f}%")
    print(f"  Max k error across all:   {max(max_abs_err)}")
    per_sample.sort(key=lambda r: r["mae"])
    print(f"\n  Top 5 best samples (lowest MAE):")
    for r in per_sample[:5]:
        print(f"    {r['name']:<30s}  MAE={r['mae']:>8.2f}  k_acc={100*r['k_acc']:5.1f}%  "
              f"off1={100*r['k_off1']:5.1f}%  mean|k_err|={r['k_mean_abs_off']:.2f}")
    print(f"\n  Top 5 worst samples (highest MAE):")
    for r in per_sample[-5:]:
        print(f"    {r['name']:<30s}  MAE={r['mae']:>8.2f}  k_acc={100*r['k_acc']:5.1f}%  "
              f"off1={100*r['k_off1']:5.1f}%  mean|k_err|={r['k_mean_abs_off']:.2f}")
    mae_all = np.array([r["mae"] for r in per_sample])
    print(f"\n  Per-sample MAE distribution (mm):")
    print(f"    min/p25/median/p75/max: {mae_all.min():.2f} / "
          f"{np.percentile(mae_all, 25):.2f} / {np.median(mae_all):.2f} / "
          f"{np.percentile(mae_all, 75):.2f} / {mae_all.max():.2f}")


def run_sanity(device):
    for loader, label in [
        (lambda: _load_learnedk(device, frames=(0,), tag="1f"), "learnedk_1f"),
        (lambda: _load_learnedk(device, frames=(0, 6, 12), tag="3f"), "learnedk_3f"),
        (lambda: _load_learnedk(device, frames=(0, 6, 12), tag="3f",
                                k_head_variant="wide"), "learnedk_3f_wide"),
        (lambda: _load_hybrid(device), "hybrid_3f"),
    ]:
        try:
            model, frames = loader()
        except FileNotFoundError as e:
            print(f"[skip {label}] {e}")
            continue
        sanity_check_model(model, frames, device, label)


# ---------------------------------------------------------------------------
# (2) noisy-k sensitivity sweep (runs on the oracle-k 3-frame PhiCalNet)
# ---------------------------------------------------------------------------
def corrupt_k(k_gt, mask, fraction, mode, rng):
    """Return a corrupted copy of k_gt; corruption applied only on mask."""
    k = k_gt.copy()
    if fraction <= 0:
        return k
    # Pick a random subset of object pixels to corrupt
    obj_idx = np.flatnonzero(mask.ravel())
    n_corrupt = int(round(fraction * obj_idx.size))
    if n_corrupt == 0:
        return k
    chosen = rng.choice(obj_idx, size=n_corrupt, replace=False)
    flat = k.ravel()
    if mode == "off1":
        # +1 or -1 (clamped to [0, K_MAX-1])
        delta = rng.choice([-1, 1], size=n_corrupt)
        new_vals = np.clip(flat[chosen] + delta, 0, K_MAX - 1)
    elif mode == "uniform":
        new_vals = rng.integers(0, K_MAX, size=n_corrupt)
    else:
        raise ValueError(mode)
    flat[chosen] = new_vals
    return flat.reshape(k.shape)


def _noisy_k_sweep_for_model(model, dataset, cal, device, label, rng):
    # Precompute model phase on all test samples; reuse across corruption levels.
    print(f"\nPrecomputing model phase for {label} ({len(dataset)} samples)...")
    cache = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=label):
            s = dataset[i]
            fringe = s["fringe"].unsqueeze(0).to(device)
            out = model(fringe)
            sin_hat = out["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            cache.append({
                "sin_hat": sin_hat, "cos_hat": cos_hat,
                "k_gt": s["k_gt"].numpy(),
                "mask": s["mask_gt"].squeeze().bool().numpy(),
                "depth_gt": s["depth_gt"].squeeze().numpy(),
            })

    fractions = [0.0, 0.01, 0.05, 0.10, 0.25]
    modes = ["off1", "uniform"]
    results = []

    for mode in modes:
        for f in fractions:
            all_errs = []
            per_sample = []
            for c in cache:
                k_noisy = corrupt_k(c["k_gt"], c["mask"], f, mode, rng)
                uph = full_unwrap(c["sin_hat"], c["cos_hat"], k_noisy)
                uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
                depth_pred = cal(uph_t).squeeze().cpu().numpy()
                err = np.abs(depth_pred - c["depth_gt"])[c["mask"]]
                all_errs.append(err)
                per_sample.append(float(err.mean()))
            errs = np.concatenate(all_errs)
            per_sample_rmse = [np.sqrt(np.mean(e ** 2)) for e in all_errs]
            results.append({
                "mode": mode, "fraction": f,
                "pixel_mae": float(errs.mean()),
                "pixel_rmse": float(np.sqrt((errs ** 2).mean())),
                "object_mae": float(np.mean(per_sample)),
                "object_rmse": float(np.mean(per_sample_rmse)),
            })

    print(f"\n=== {label} ===")
    print("=" * 84)
    print(f"{'mode':<10s}  {'fraction':>8s}  {'Obj MAE':>10s}  {'Obj RMSE':>10s}  "
          f"{'Pix MAE':>10s}  {'Pix RMSE':>10s}")
    print("-" * 84)
    for r in results:
        print(f"{r['mode']:<10s}  {r['fraction']:>8.2%}  "
              f"{r['object_mae']:>10.3f}  {r['object_rmse']:>10.3f}  "
              f"{r['pixel_mae']:>10.3f}  {r['pixel_rmse']:>10.3f}")
    print("=" * 84)


def run_noisy_k(device):
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    rng = np.random.default_rng(0)

    # --- 1-frame PhiCalNet baseline ---
    ckpt_path_1f = Path("checkpoints_phase_A1_refined_v2/best_model.pth")
    if ckpt_path_1f.exists():
        model_1f = PhaseUNetFPP(variant="A1", in_channels=1, K_max=K_MAX,
                                dropout_rate=0.0)
        model_1f.load_state_dict(
            torch.load(ckpt_path_1f, map_location=device)["model_state_dict"])
        model_1f = model_1f.to(device).eval()
        ds_1f = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test",
                             K_max=K_MAX)
        _noisy_k_sweep_for_model(model_1f, ds_1f, cal, device,
                                  "1-frame PhiCalNet", rng)
    else:
        print(f"[skip 1-frame sweep] checkpoint not found: {ckpt_path_1f}")

    # --- 3-frame PhiCalNet multiframe ---
    ckpt_path_3f = Path("checkpoints_phase_A1_multiframe/best_model.pth")
    if ckpt_path_3f.exists():
        model_3f = PhaseUNetFPP(variant="A1", in_channels=3, K_max=K_MAX,
                                dropout_rate=0.0)
        model_3f.load_state_dict(
            torch.load(ckpt_path_3f, map_location=device)["model_state_dict"])
        model_3f = model_3f.to(device).eval()
        ds_3f = PhaseMultiFrameDataset(
            MULTIFRAME_ROOT / "test/fringe", PHASE_ROOT / "test",
            K_max=K_MAX, frame_indices=(0, 6, 12),
        )
        _noisy_k_sweep_for_model(model_3f, ds_3f, cal, device,
                                  "3-frame PhiCalNet", rng)
    else:
        print(f"[skip 3-frame sweep] checkpoint not found: {ckpt_path_3f}")

    print("\nReference: fraction=0 rows should match baseline_1f / multiframe_3f "
          "numbers from eval_multiframe_grid.py.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["sanity", "noisy"], default=None,
                   help="Run only part 1 (sanity) or part 2 (noisy); "
                        "default runs both.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if args.only in (None, "sanity"):
        print("\n####  PART 1: learned-k sanity check  ####")
        run_sanity(device)

    if args.only in (None, "noisy"):
        print("\n\n####  PART 2: noisy-k sensitivity sweep (on 3-frame PhiCalNet)  ####")
        run_noisy_k(device)


if __name__ == "__main__":
    main()
