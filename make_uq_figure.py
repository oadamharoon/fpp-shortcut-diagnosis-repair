"""
Render per-pixel UQ row PNGs for one or more test samples, in the same
paper_panels style as regen_phaseunet_viz.py.

For each chosen test sample we write two 1x3 row PNGs (one per
architecture):
    uq_idx{NN}_{name}_phicalnet_row.png  -- Fringe | |residual| (mm) | snapshot std (mm)
    uq_idx{NN}_{name}_unet_row.png       -- (same columns, UNet baseline)

The two error panels share their vmin/vmax across architectures so the
PhiCalNet and UNet rows are directly comparable; the cap is the joint
95th percentile of object pixels (matching the regen_phaseunet_viz.py
convention, so saturated pixels are the catastrophic outliers and the
bulk is visible).

Calibration on the validation split (the slow step) is cached to
``--cache_dir`` keyed on the snapshot-epoch list, so re-renders with
different ``--samples`` are fast.

Usage:
    python -u make_uq_figure.py
    python -u make_uq_figure.py --samples 0,5,12 --device cpu
    python -u make_uq_figure.py --phi_epochs 150,160,170,180,190 \
                                --unet_epochs 100,110,120,130,140
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Font sizes -- match regen_phaseunet_viz.py / vis_pred.py paper conventions.
TITLE_FS      = 16
CBAR_LABEL_FS = 16
CBAR_TICK_FS  = 14

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from models.calibration_layer import CalibrationLayer
from models.phase_unet import PhaseUNetFPP
sys.path.insert(0, str(Path(__file__).parent / "UNet"))
from UNet.unet import UNetFPP

from conformal_snapshot import full_unwrap, calibrate as phi_calibrate
from conformal_snapshot_unet import calibrate as unet_calibrate

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"
PHI_CKPT_DIR = Path("checkpoints_phase_A1_refined_v2")
UNET_CKPT_DIR = (BASE / "FPP-ML-Benchmarking/UNet"
                      / "checkpoints_hybrid_l1_a07_depth_individual_normalized")

K_MAX = 32
SCALE_FLOOR = 0.05


def load_phi_snapshots(epochs, device):
    models = []
    for ep in epochs:
        p = PHI_CKPT_DIR / f"checkpoint_epoch_{ep:04d}.pth"
        if not p.exists():
            raise FileNotFoundError(p)
        m = PhaseUNetFPP(variant="A1", in_channels=1, K_max=K_MAX,
                         dropout_rate=0.0)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        models.append(m.to(device).eval())
    print(f"Loaded {len(models)} PhiCalNet snapshots: {epochs}")
    return models


def load_unet_snapshots(epochs, device, dropout_rate=0.5):
    models = []
    for ep in epochs:
        p = UNET_CKPT_DIR / f"checkpoint_epoch_{ep:04d}.pth"
        if not p.exists():
            raise FileNotFoundError(p)
        m = UNetFPP(in_channels=1, out_channels=1, dropout_rate=dropout_rate)
        ckpt = torch.load(p, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        m.load_state_dict(sd)
        models.append(m.to(device).eval())
    print(f"Loaded {len(models)} UNet snapshots: {epochs}")
    return models


def predict_phi_sample(models, cal_layer, sample, device):
    fringe = sample["fringe"].unsqueeze(0).to(device)
    k_gt = sample["k_gt"].numpy()
    depths = []
    with torch.no_grad():
        for m in models:
            pred = m(fringe)
            sin_hat = pred["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = pred["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            uph = full_unwrap(sin_hat, cos_hat, k_gt)
            uph_t = (torch.from_numpy(uph)
                        .unsqueeze(0).unsqueeze(0).float().to(device))
            depths.append(cal_layer(uph_t).squeeze().cpu().numpy())
    return np.stack(depths, axis=0)  # (N, H, W)


def predict_unet_sample(models, sample, device):
    fringe = sample["fringe"].unsqueeze(0).to(device)
    d_min = float(sample["d_min"])
    d_max = float(sample["d_max"])
    depths = []
    with torch.no_grad():
        for m in models:
            pred_norm = m(fringe).squeeze().cpu().numpy()
            depths.append(pred_norm * (d_max - d_min) + d_min)
    return np.stack(depths, axis=0)  # (N, H, W)


def collect_calibration_data(predict_fn, dataset, device, desc):
    """Run predict_fn over the dataset, return per-sample residual,
    depth_std, mask (used for split-conformal calibration)."""
    residuals, depth_stds, masks = [], [], []
    for i in tqdm(range(len(dataset)), desc=desc):
        s = dataset[i]
        depths = predict_fn(s)
        depth_mean = depths.mean(axis=0)
        depth_std = depths.std(axis=0)
        depth_gt = s["depth_gt"].squeeze().numpy()
        mask = s["mask_gt"].squeeze().bool().numpy()
        residuals.append(np.abs(depth_mean - depth_gt))
        depth_stds.append(depth_std)
        masks.append(mask)
    return residuals, depth_stds, masks


def cache_key(arch, epochs):
    h = hashlib.sha1(",".join(str(e) for e in epochs).encode()).hexdigest()[:12]
    return f"{arch}_val_{h}.pkl"


def cached_calibration(arch, epochs, predict_fn, dataset, device, cache_dir):
    cache_path = cache_dir / cache_key(arch, epochs)
    if cache_path.exists():
        print(f"  loading cached calibration: {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  computing calibration -> {cache_path.name}")
    residuals, depth_stds, masks = collect_calibration_data(
        predict_fn, dataset, device, f"{arch}-val")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((residuals, depth_stds, masks), f)
    return residuals, depth_stds, masks


def _draw_panel(ax, data, cmap, vmin, vmax, title,
                cbar_label=None, show_cbar=True):
    """Image panel with top title and optional colorbar.

    Mirrors regen_phaseunet_viz.py._draw_panel exactly so the UQ rows
    share font sizes and layout with the existing paper_panels."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontsize=TITLE_FS)
    ax.axis('off')
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FS)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=CBAR_LABEL_FS)
    return im


def render_uncertainty_rows(sample_name, fringe, mask,
                            phi_residual, phi_std,
                            unet_residual, unet_std,
                            out_phi_path, out_unet_path):
    """Two 1x3 row PNGs per sample (PhiCalNet row, UNet row), each with
    Fringe | |residual| (mm) | snapshot std (mm). Color scales for the
    error panels are shared across architectures so the rows are
    directly comparable; both vmaxes are the joint 95th percentile of
    object pixels (regen_phaseunet_viz.py convention)."""
    nan_mask = np.where(mask, 1.0, np.nan)
    phi_res_v = phi_residual * nan_mask
    phi_std_v = phi_std * nan_mask
    unet_res_v = unet_residual * nan_mask
    unet_std_v = unet_std * nan_mask

    # Joint 95th-percentile vmaxes across both architectures so the two
    # row PNGs share their color scale.
    res_pool = np.concatenate([phi_residual[mask], unet_residual[mask]])
    std_pool = np.concatenate([phi_std[mask], unet_std[mask]])
    res_vmax = float(np.percentile(res_pool, 95)) if res_pool.size else 1.0
    std_vmax = float(np.percentile(std_pool, 95)) if std_pool.size else 1.0
    res_vmax = max(res_vmax, 1e-3)
    std_vmax = max(std_vmax, 1e-3)

    phi_obj_mae = float(phi_residual[mask].mean())
    phi_obj_rmse = float(np.sqrt((phi_residual[mask] ** 2).mean()))
    unet_obj_mae = float(unet_residual[mask].mean())
    unet_obj_rmse = float(np.sqrt((unet_residual[mask] ** 2).mean()))

    rows = [
        ("PhiCalNet", phi_res_v, phi_std_v, phi_obj_mae, phi_obj_rmse,
         out_phi_path),
        ("UNet baseline", unet_res_v, unet_std_v, unet_obj_mae,
         unet_obj_rmse, out_unet_path),
    ]
    for label, res_v, std_v, mae, rmse, path in rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        _draw_panel(axes[0], fringe, cmap='gray',
                    vmin=float(np.min(fringe)), vmax=float(np.max(fringe)),
                    title=f'Fringe Pattern ({label})',
                    show_cbar=False)
        _draw_panel(axes[1], res_v, cmap='hot',
                    vmin=0.0, vmax=res_vmax,
                    title=(f'Absolute Residual (mm)\n'
                           f'Object MAE: {mae:.2f}, RMSE: {rmse:.2f}'),
                    cbar_label=f'|residual| (mm, 95th %ile: {res_vmax:.1f})')
        _draw_panel(axes[2], std_v, cmap='hot',
                    vmin=0.0, vmax=std_vmax,
                    title='Snapshot Std = Uncertainty (mm)',
                    cbar_label=f'std (mm, 95th %ile: {std_vmax:.2f})')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {path}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    phi_epochs = [int(e) for e in args.phi_epochs.split(",")]
    unet_epochs = [int(e) for e in args.unet_epochs.split(",")]

    phi_models = load_phi_snapshots(phi_epochs, device)
    unet_models = load_unet_snapshots(unet_epochs, device)
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    val_ds = PhaseDataset(DATA_ROOT / "val/fringe",  PHASE_ROOT / "val",
                          K_max=K_MAX)
    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test",
                           K_max=K_MAX)

    print("\nCalibrating PhiCalNet on val set ...")
    phi_val_r, phi_val_d, phi_val_m = cached_calibration(
        "phi", phi_epochs,
        lambda s: predict_phi_sample(phi_models, cal_layer, s, device),
        val_ds, device, cache_dir)
    phi_q, _ = phi_calibrate(phi_val_r, phi_val_d, phi_val_m, "std",
                             alphas=[0.05], scale_floor=SCALE_FLOOR)
    phi_q95 = phi_q[0.05]
    print(f"  PhiCalNet q_(1-0.05) = {phi_q95:.4f}")

    print("\nCalibrating UNet on val set ...")
    unet_val_r, unet_val_d, unet_val_m = cached_calibration(
        "unet", unet_epochs,
        lambda s: predict_unet_sample(unet_models, s, device),
        val_ds, device, cache_dir)
    unet_q, _ = unet_calibrate(unet_val_r, unet_val_d, unet_val_m, "std",
                               alphas=[0.05], scale_floor=SCALE_FLOOR)
    unet_q95 = unet_q[0.05]
    print(f"  UNet      q_(1-0.05) = {unet_q95:.4f}")

    sample_indices = [int(i) for i in args.samples.split(",")]
    print(f"\nRendering test sample indices: {sample_indices}")

    for idx in sample_indices:
        if idx < 0 or idx >= len(test_ds):
            print(f"  skip idx={idx}: out of range (test has {len(test_ds)})")
            continue
        s = test_ds[idx]
        name = s["name"]
        print(f"\n[sample {idx}] {name}")

        fringe = s["fringe"].squeeze().numpy()
        depth_gt = s["depth_gt"].squeeze().numpy()
        mask = s["mask_gt"].squeeze().bool().numpy()

        phi_depths = predict_phi_sample(phi_models, cal_layer, s, device)
        phi_mean = phi_depths.mean(axis=0)
        phi_std = phi_depths.std(axis=0)
        phi_res = np.abs(phi_mean - depth_gt)

        unet_depths = predict_unet_sample(unet_models, s, device)
        unet_mean = unet_depths.mean(axis=0)
        unet_std = unet_depths.std(axis=0)
        unet_res = np.abs(unet_mean - depth_gt)

        print(f"  PhiCalNet: residual mean={phi_res[mask].mean():.2f} mm, "
              f"std mean={phi_std[mask].mean():.3f} mm, "
              f"max residual={phi_res[mask].max():.1f} mm")
        print(f"  UNet     : residual mean={unet_res[mask].mean():.2f} mm, "
              f"std mean={unet_std[mask].mean():.3f} mm, "
              f"max residual={unet_res[mask].max():.1f} mm")

        safe_name = name.replace("/", "_").replace(" ", "_")
        out_phi_path = out_dir / f"uq_idx{idx:02d}_{safe_name}_phicalnet_row.png"
        out_unet_path = out_dir / f"uq_idx{idx:02d}_{safe_name}_unet_row.png"
        render_uncertainty_rows(name, fringe, mask,
                                phi_res, phi_std,
                                unet_res, unet_std,
                                out_phi_path, out_unet_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phi_epochs", type=str, default="150,160,170,180,190",
                   help="PhiCalNet snapshot epochs")
    p.add_argument("--unet_epochs", type=str, default="100,110,120,130,140",
                   help="UNet snapshot epochs")
    p.add_argument("--samples", type=str, default="0,5,12,20",
                   help="Comma-separated test sample indices to render")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="uq_figures")
    p.add_argument("--cache_dir", type=str, default="uq_cache",
                   help="Directory for cached val-set calibration outputs")
    args = p.parse_args()
    main(args)
