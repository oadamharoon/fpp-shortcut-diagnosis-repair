"""
Unified evaluation across the multi-frame ablation grid.

Evaluates each trained configuration on the test split and prints object
depth MAE / RMSE / percentiles. Classical baseline (analytical phase from
all 18 frames + oracle k) is also reported as the data-limit ceiling.

Configurations:
  baseline_1f    : existing PhiCalNet (1-frame, learned phi, oracle k)
  multiframe_3f  : 3-frame PhiCalNet (3-frame, learned phi, oracle k)
  hybrid_3f      : PhiCalHybridNet (3-frame, analytical phi, learned k)
  learnedk_1f    : PhiCalNetLearnedK (1-frame, learned phi, learned k)
  learnedk_3f    : PhiCalNetLearnedK (3-frame, learned phi, learned k)

Usage:
    python eval_multiframe_grid.py                    # runs all configs
    python eval_multiframe_grid.py --configs hybrid_3f,learnedk_3f
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
from models.phase_fno import PhaseFNO_FPP
from models.phase_ftp import PhaseFTP_FPP
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def full_unwrap(sin_np, cos_np, k_np):
    """MATLAB-style unwrap from (sin, cos, k) -> unwrapped phase field."""
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def error_summary(per_sample_errs, label):
    """Compute both pixel-level pooled and object-level (per-sample averaged)
    metrics. `per_sample_errs` is a list of 1D arrays -- one per test sample."""
    all_errs = np.concatenate(per_sample_errs)
    n = all_errs.size

    # Pixel-level (pooled across all object pixels)
    pixel_mae = float(all_errs.mean())
    pixel_rmse = float(np.sqrt((all_errs ** 2).mean()))

    # Object-level (per-sample then averaged)
    per_sample_mae = np.array([e.mean() for e in per_sample_errs])
    per_sample_rmse = np.array([np.sqrt((e ** 2).mean())
                                for e in per_sample_errs])
    object_mae = float(per_sample_mae.mean())
    object_rmse = float(per_sample_rmse.mean())

    pct = lambda p: float(np.percentile(all_errs, p))
    thrs = [(10, (all_errs > 10).sum()), (50, (all_errs > 50).sum()),
            (100, (all_errs > 100).sum()), (500, (all_errs > 500).sum())]
    return {
        "label": label, "n": n,
        "pixel_mae": pixel_mae, "pixel_rmse": pixel_rmse,
        "object_mae": object_mae, "object_rmse": object_rmse,
        "p50": pct(50), "p90": pct(90), "p99": pct(99), "p99.9": pct(99.9),
        "max": float(all_errs.max()),
        "thrs": {t: int(c) for t, c in thrs},
    }


def print_summary(s):
    print(f"\n--- {s['label']} ---")
    print(f"  object pixels:       {s['n']:,}")
    print(f"  Object  MAE / RMSE:  {s['object_mae']:9.3f}  /  {s['object_rmse']:9.3f}  (per-sample averaged; matches paper convention)")
    print(f"  Pixel   MAE / RMSE:  {s['pixel_mae']:9.3f}  /  {s['pixel_rmse']:9.3f}  (pooled over all pixels)")
    print(f"  p50 / p90 (mm):      {s['p50']:9.3f}  /  {s['p90']:9.3f}")
    print(f"  p99 / p99.9 (mm):    {s['p99']:9.3f}  /  {s['p99.9']:9.3f}")
    print(f"  max (mm):            {s['max']:9.3f}")
    for thr, c in s["thrs"].items():
        pct = 100 * c / s["n"]
        print(f"  err > {thr:>3} mm:       {c:>9,}   ({pct:.3f}%)")


# ---------------------------------------------------------------------------
# Per-config evaluation routines
# ---------------------------------------------------------------------------
def _per_sample_errs_oracle_phase_unet(model, device, dataset, cal_layer,
                                        label):
    """For PhaseUNetFPP-style models with oracle-k and separate unwrap step."""
    all_errs = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=label):
            s = dataset[i]
            fringe = s["fringe"].unsqueeze(0).to(device)
            out = model(fringe)
            sin_hat = out["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            k_gt = s["k_gt"].numpy()
            mask = s["mask_gt"].squeeze().bool().numpy()
            depth_gt = s["depth_gt"].squeeze().numpy()
            uph = full_unwrap(sin_hat, cos_hat, k_gt)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth_pred = cal_layer(uph_t).squeeze().cpu().numpy()
            all_errs.append(np.abs(depth_pred - depth_gt)[mask])
    return all_errs


def _per_sample_errs_end_to_end(model, device, dataset, label):
    """For models that directly return depth_hat (hybrid + learned-k)."""
    all_errs = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=label):
            s = dataset[i]
            fringe = s["fringe"].unsqueeze(0).to(device)
            out = model(fringe)
            depth_pred = out["depth_hat"].squeeze().cpu().numpy()
            mask = s["mask_gt"].squeeze().bool().numpy()
            depth_gt = s["depth_gt"].squeeze().numpy()
            all_errs.append(np.abs(depth_pred - depth_gt)[mask])
    return all_errs


def eval_baseline_1f(device):
    """Existing PhiCalNet: 1-frame input, learned phase, oracle k."""
    ckpt_path = Path("checkpoints_phase_A1_refined_v2/best_model.pth")
    if not ckpt_path.exists():
        print(f"[skip baseline_1f] checkpoint not found: {ckpt_path}")
        return None
    model = PhaseUNetFPP(variant="A1", in_channels=1, K_max=K_MAX,
                        dropout_rate=0.0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    errs = _per_sample_errs_oracle_phase_unet(model, device, ds, cal,
                                              "baseline_1f")
    return error_summary(errs, "baseline_1f  (1f, learned phi, oracle k)")


def eval_multiframe_3f(device):
    ckpt_path = Path("checkpoints_phase_A1_multiframe/best_model.pth")
    if not ckpt_path.exists():
        print(f"[skip multiframe_3f] checkpoint not found: {ckpt_path}")
        return None
    model = PhaseUNetFPP(variant="A1", in_channels=3, K_max=K_MAX,
                        dropout_rate=0.0)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "test/fringe", PHASE_ROOT / "test",
        K_max=K_MAX, frame_indices=(0, 6, 12),
    )
    errs = _per_sample_errs_oracle_phase_unet(model, device, ds, cal,
                                              "multiframe_3f")
    return error_summary(errs, "multiframe_3f  (3f, learned phi, oracle k)")


def eval_hybrid_3f(device):
    ckpt_path = Path("checkpoints_phicalnet_hybrid/best_model.pth")
    if not ckpt_path.exists():
        print(f"[skip hybrid_3f] checkpoint not found: {ckpt_path}")
        return None
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    model = PhiCalHybridNet(
        calibration_layer=cal, frame_indices=(0, 6, 12),
        K_max=K_MAX, gumbel_temperature=1.0,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "test/fringe", PHASE_ROOT / "test",
        K_max=K_MAX, frame_indices=(0, 6, 12),
    )
    errs = _per_sample_errs_end_to_end(model, device, ds, "hybrid_3f")
    return error_summary(errs, "hybrid_3f      (3f, analytical phi, learned k)")


def eval_learnedk(device, frames, tag, k_head_variant="default"):
    # Checkpoint path / display tag mirrors the training script's convention:
    # suffix is empty for the default head and "_wide" for the wide head.
    suffix = "" if k_head_variant == "default" else f"_{k_head_variant}"
    ckpt_path = Path(f"checkpoints_phicalnet_learnedk_{tag}{suffix}/best_model.pth")
    pretty_tag = f"{tag}{suffix}"
    if not ckpt_path.exists():
        print(f"[skip learnedk_{pretty_tag}] checkpoint not found: {ckpt_path}")
        return None
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    model = PhiCalNetLearnedK(
        calibration_layer=cal, in_channels=len(frames), K_max=K_MAX,
        k_head_variant=k_head_variant,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "test/fringe", PHASE_ROOT / "test",
        K_max=K_MAX, frame_indices=frames,
    )
    errs = _per_sample_errs_end_to_end(model, device, ds,
                                       f"learnedk_{pretty_tag}")
    variant_desc = "" if k_head_variant == "default" else f", {k_head_variant} k-head"
    return error_summary(
        errs,
        f"learnedk_{pretty_tag:<8}  ({tag}, learned phi, learned k{variant_desc})",
    )


def eval_fno(device, width, modes, n_blocks):
    """Single-shot FNO backbone (oracle k); evaluated like baseline_1f."""
    tag = f"phase_fno_w{width}_m{modes}_b{n_blocks}"
    ckpt_path = Path(f"checkpoints_{tag}/best_model.pth")
    if not ckpt_path.exists():
        print(f"[skip {tag}] checkpoint not found: {ckpt_path}")
        return None
    model = PhaseFNO_FPP(in_channels=1, width=width, modes=modes,
                         n_blocks=n_blocks, K_max=K_MAX)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    errs = _per_sample_errs_oracle_phase_unet(model, device, ds, cal, tag)
    return error_summary(errs,
        f"fno_w{width}_m{modes}_b{n_blocks}  (1f, FNO learned phi, oracle k)")


def eval_ftp(device, filter_type="parametric"):
    """Single-shot FTP-as-architecture backbone (oracle k); evaluated like
    baseline_1f. Physics-constrained: FFT/IFFT structure fixed, only DC
    removal subnet (~120k params) and bandpass filter learned."""
    tag = f"phase_ftp_{filter_type}"
    ckpt_path = Path(f"checkpoints_{tag}/best_model.pth")
    if not ckpt_path.exists():
        print(f"[skip {tag}] checkpoint not found: {ckpt_path}")
        return None
    model = PhaseFTP_FPP(in_channels=1, H=960, W=960,
                         filter_type=filter_type, K_max=K_MAX)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model = model.to(device).eval()
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    errs = _per_sample_errs_oracle_phase_unet(model, device, ds, cal, tag)
    return error_summary(errs,
        f"ftp_{filter_type:<10s}  (1f, FTP physics-informed phi, oracle k)")


CONFIGS = {
    "baseline_1f":       lambda d: eval_baseline_1f(d),
    "multiframe_3f":     lambda d: eval_multiframe_3f(d),
    "hybrid_3f":         lambda d: eval_hybrid_3f(d),
    "learnedk_1f":       lambda d: eval_learnedk(d, frames=(0,),       tag="1f"),
    "learnedk_3f":       lambda d: eval_learnedk(d, frames=(0, 6, 12), tag="3f"),
    "learnedk_3f_wide":  lambda d: eval_learnedk(d, frames=(0, 6, 12), tag="3f",
                                                 k_head_variant="wide"),
    "fno_m32":           lambda d: eval_fno(d, width=64, modes=32, n_blocks=4),
    "fno_m48":           lambda d: eval_fno(d, width=64, modes=48, n_blocks=4),
    "ftp_parametric":    lambda d: eval_ftp(d, filter_type="parametric"),
    "ftp_flexible":      lambda d: eval_ftp(d, filter_type="flexible"),
}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wanted = (args.configs.split(",") if args.configs
              else list(CONFIGS.keys()))
    results = []
    for name in wanted:
        name = name.strip()
        if name not in CONFIGS:
            print(f"[warn] unknown config '{name}'; choices: {list(CONFIGS)}")
            continue
        r = CONFIGS[name](device)
        if r is not None:
            results.append(r)
            print_summary(r)

    # Final compact table (both metrics so the paper-convention numbers are
    # visible alongside the pooled-pixel numbers).
    print("\n" + "=" * 96)
    print(f"{'Configuration':<50s} {'Obj MAE':>8s} {'Obj RMSE':>9s}   "
          f"{'Pix MAE':>8s} {'Pix RMSE':>9s}")
    print("-" * 96)
    for r in results:
        print(f"{r['label']:<50s} {r['object_mae']:>8.3f} {r['object_rmse']:>9.3f}   "
              f"{r['pixel_mae']:>8.3f} {r['pixel_rmse']:>9.3f}")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None,
                        help=f"Comma-separated subset of {list(CONFIGS)}. "
                             "Default: all.")
    args = parser.parse_args()
    main(args)
