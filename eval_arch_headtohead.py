"""
Head-to-head depth-MAE evaluation of the four phase-intermediate architectures,
all under the SAME A1 single-frame oracle-k pipeline used by the paper:
  forward -> (sin_hat, cos_hat) -> full_matlab_unwrap with k_gt -> calibration -> depth (mm).
Object/overall MAE and RMSE in mm, plus wrapped-phase MAE (rad), on the 30-sample test split.

Models (identical training recipe, so directly comparable):
  UNet, DemodPhaseNet, Vanilla FNO, Hybrid FNO.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from models.calibration_layer import CalibrationLayer
from models.phase_unet import PhaseUNetFPP
from models.phase_fno import PhaseFNO_FPP
from models.phase_fno_hybrid import PhaseFNOHybrid
from models.phase_demod import DemodPhaseNet

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"
T_P = 36.0
K_MAX = 32


def full_matlab_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + 2.0 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def build(name):
    if name == "UNet":
        m = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
        ck = "checkpoints_phase_A1_refined_v2/best_model.pth"
    elif name == "Demod":
        m = DemodPhaseNet(in_channels=1, K_max=K_MAX)
        ck = "checkpoints_phase_A1_refined_v2_demod/best_model.pth"
    elif name == "FNO":
        m = PhaseFNO_FPP(in_channels=1, width=64, modes=32, n_blocks=4, K_max=K_MAX)
        ck = "checkpoints_phase_A1_refined_v2_fno/best_model.pth"
    elif name == "HybridFNO":
        m = PhaseFNOHybrid(in_channels=1, width=64, modes=32, n_blocks=4, K_max=K_MAX)
        ck = "checkpoints_phase_A1_refined_v2_fnohybrid/best_model.pth"
    else:
        raise ValueError(name)
    ckpt = torch.load(ck, map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval(), (ckpt.get("epoch", -1) + 1), sum(p.numel() for p in m.parameters())


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import re


def geom_class(name):
    """Strip the trailing _A<angle> viewpoint suffix to get the object class."""
    return re.sub(r"_A\d+$", "", name)


def evaluate(name, cal, ds):
    model, epoch, nparam = build(name)
    model = model.to(DEVICE)
    o_mae, o_rmse, ov_mae, wph, names = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = model(s["fringe"].unsqueeze(0).to(DEVICE))
            sin_hat = out["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
            k_gt = s["k_gt"].numpy()
            mask = s["mask_gt"].squeeze().numpy() > 0
            dgt = s["depth_gt"].numpy().squeeze()
            uph = full_matlab_unwrap(sin_hat, cos_hat, k_gt)
            dpred = cal(torch.from_numpy(uph)[None, None].float().to(DEVICE)).squeeze().cpu().numpy()
            err = np.abs(dpred - dgt)
            o_mae.append(err[mask].mean()); o_rmse.append(np.sqrt((err[mask] ** 2).mean()))
            ov_mae.append(err.mean()); names.append(s["name"])
            sg = s["sin_phi_gt"].squeeze().numpy().astype(np.float64)
            cg = s["cos_phi_gt"].squeeze().numpy().astype(np.float64)
            d = np.abs(np.arctan2(sin_hat, cos_hat) - np.arctan2(sg, cg))
            wph.append(np.minimum(d, 2 * np.pi - d)[mask].mean())
    o_mae = np.array(o_mae)
    # per-geometry-class mean object MAE
    classes = {}
    for nm, m in zip(names, o_mae):
        classes.setdefault(geom_class(nm), []).append(m)
    cls_means = {c: float(np.mean(v)) for c, v in classes.items()}
    print(f"{name:11s} | {nparam/1e6:5.2f}M | ep{epoch:4d} | "
          f"obj MAE {o_mae.mean():7.2f} mm | obj RMSE {np.mean(o_rmse):7.2f} mm | "
          f"overall MAE {np.mean(ov_mae):7.2f} mm | wphase {np.mean(wph):.4f} rad", flush=True)
    return {"name": name, "obj_mae": float(o_mae.mean()), "cls_means": cls_means}


def main():
    torch.set_grad_enabled(False)
    cal = CalibrationLayer(str(CAL_DIR)).to(DEVICE)
    ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    print(f"Test split: {len(ds)} samples. A1 single-frame oracle-k pipeline.\n")
    results = []
    for name in ["UNet", "Demod", "FNO", "HybridFNO"]:
        try:
            results.append(evaluate(name, cal, ds))
        except Exception as e:
            print(f"{name:11s} | FAILED: {e}", flush=True)

    # Memorization test: spread of per-geometry-class MAE. A memorizer is great on
    # easy (smooth) classes and terrible on hard ones -> large max/min ratio and CV.
    print("\n=== Per-geometry-class object MAE (mm) + memorization spread ===")
    all_cls = sorted({c for r in results for c in r["cls_means"]})
    hdr = "class".ljust(34) + "".join(f"{r['name']:>11s}" for r in results)
    print(hdr); print("-" * len(hdr))
    for c in all_cls:
        row = c.ljust(34) + "".join(f"{r['cls_means'].get(c, float('nan')):11.1f}" for r in results)
        print(row)
    print("-" * len(hdr))
    for stat, fn in [("easiest class", lambda v: min(v)),
                     ("hardest class", lambda v: max(v)),
                     ("hard/easy ratio", lambda v: max(v) / max(min(v), 1e-9)),
                     ("CV (std/mean)", lambda v: float(np.std(v) / max(np.mean(v), 1e-9)))]:
        line = stat.ljust(34)
        for r in results:
            v = list(r["cls_means"].values())
            line += f"{fn(v):11.2f}"
        print(line)
    print("\nDONE")


if __name__ == "__main__":
    main()
