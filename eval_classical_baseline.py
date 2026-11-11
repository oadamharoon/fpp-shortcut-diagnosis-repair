"""
Classical-pipeline ceiling on the PhiCalNet test split.

Takes the precomputed (sin phi_gt, cos phi_gt, k_gt) already stored in
training_datasets/phase_targets/test/*.npz, runs them through the exact
same unwrap + triangulation pipeline used by precompute_phase_targets.py,
and reports per-sample and aggregate object MAE/RMSE. This is the ceiling
for how close any single-frame predictor could get if its phase prediction
were perfect -- a data-limit, not a modeling-limit.

Run:
    python -u eval_classical_baseline.py
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
CAL_DIR = BASE / "calibration_matrices"
PHASE_DIR = BASE / "training_datasets/phase_targets/test"

T_P = 36.0
SCALE = 1.446
PHASE_OFFSET = 2.9


def load_calibration():
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)
    M_c = A_c @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M_p = A_p @ np.hstack([R_p, T_p])
    return M_c, M_p


def unwrap_phase(wph, fringe_order):
    """grayUnwrap: median filter, shift, offset, smoothing -- MATLAB-equivalent."""
    fringe_order_f = median_filter(fringe_order.astype(np.float64), size=7)
    shift = -np.pi / T_P
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + fringe_order_f * 2 * np.pi - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    unwr = uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi
    return unwr + PHASE_OFFSET


def reconstruct_depth(uphH, M_c, M_p):
    """uphToXYZ + scale, 1-based pixel coords (matches reference pipeline)."""
    H, W = uphH.shape
    mc, mp = M_c, M_p
    up = uphH * T_P / (2 * np.pi)
    row_grid, col_grid = np.meshgrid(
        np.arange(1, H + 1, dtype=np.float64),
        np.arange(1, W + 1, dtype=np.float64),
        indexing="ij",
    )
    u_cam, v_cam = col_grid, row_grid
    A00 = mc[0, 0] - u_cam * mc[2, 0]
    A01 = mc[0, 1] - u_cam * mc[2, 1]
    A02 = mc[0, 2] - u_cam * mc[2, 2]
    A10 = mc[1, 0] - v_cam * mc[2, 0]
    A11 = mc[1, 1] - v_cam * mc[2, 1]
    A12 = mc[1, 2] - v_cam * mc[2, 2]
    A20 = mp[1, 0] - up * mp[2, 0]
    A21 = mp[1, 1] - up * mp[2, 1]
    A22 = mp[1, 2] - up * mp[2, 2]
    b0 = u_cam * mc[2, 3] - mc[0, 3]
    b1 = v_cam * mc[2, 3] - mc[1, 3]
    b2 = up * mp[2, 3] - mp[1, 3]
    det = (
        A00 * (A11 * A22 - A12 * A21)
        - A01 * (A10 * A22 - A12 * A20)
        + A02 * (A10 * A21 - A11 * A20)
    )
    det_z = (
        A00 * (A11 * b2 - b1 * A21)
        - A01 * (A10 * b2 - b1 * A20)
        + b0 * (A10 * A21 - A11 * A20)
    )
    valid = np.abs(det) > 1e-10
    z = np.zeros_like(uphH)
    z[valid] = det_z[valid] / det[valid]
    return z / SCALE


def main():
    M_c, M_p = load_calibration()

    all_errs = []
    per_sample = []
    samples = sorted(PHASE_DIR.glob("*.npz"))
    if not samples:
        raise RuntimeError(f"No samples found at {PHASE_DIR}")

    for npz_path in samples:
        d = np.load(npz_path)
        phi = np.arctan2(d["sin_phi"].astype(np.float64),
                         d["cos_phi"].astype(np.float64))
        k = d["k"]
        mask = d["mask"] > 0
        depth_gt = d["depth_mm"].astype(np.float64)

        uphH = unwrap_phase(phi, k)
        depth_pred = reconstruct_depth(uphH, M_c, M_p)

        err_obj = np.abs(depth_pred[mask] - depth_gt[mask])
        all_errs.append(err_obj)
        mae = float(err_obj.mean())
        rmse = float(np.sqrt((err_obj ** 2).mean()))
        per_sample.append((npz_path.stem, mae, rmse, int(err_obj.size)))

    all_errs = np.concatenate(all_errs)
    agg_mae = float(all_errs.mean())
    agg_rmse = float(np.sqrt((all_errs ** 2).mean()))

    print(f"Classical baseline (unwrap + calibration on oracle sin/cos/k)")
    print(f"  Samples:        {len(samples)}")
    print(f"  Total pixels:   {all_errs.size:,}")
    print(f"  Object MAE:     {agg_mae:.3e} mm")
    print(f"  Object RMSE:    {agg_rmse:.3e} mm")
    print(f"  50th %ile:      {np.percentile(all_errs, 50):.3e} mm")
    print(f"  90th %ile:      {np.percentile(all_errs, 90):.3e} mm")
    print(f"  99th %ile:      {np.percentile(all_errs, 99):.3e} mm")
    print(f"  99.9th %ile:    {np.percentile(all_errs, 99.9):.3e} mm")
    print(f"  Max error:      {all_errs.max():.3e} mm")

    print("\nPer-sample (MAE / RMSE, in micrometres for readability):")
    for name, mae, rmse, n in per_sample:
        print(f"  {name:40s}  MAE={mae*1e3:7.3f} um  RMSE={rmse*1e3:7.3f} um  (n={n})")


if __name__ == "__main__":
    main()
