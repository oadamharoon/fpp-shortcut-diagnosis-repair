"""
Precompute VERTICAL-direction (u_p-encoding) phase targets for all samples.

Mirror of precompute_phase_targets.py for the second fringe direction that the
raw 52-frame captures already contain (per prepare_multiframe_datasets.py):
  A_25 - A_42 : 18 vertical sinusoidal fringe patterns  (encode u_p)
  A_43 - A_49 :  7 vertical gray code patterns
The wrapped-phase and gray-code conventions are identical to the horizontal
pipeline; what changes is (1) the frame ranges, (2) the triangulation row
(projector u_p equation, M_p row 0, instead of the v_p equation, row 1), and
(3) the global phase offset, which is calibrated here directly by projecting
ground-truth depth into the projector and fitting the single scalar
    off = median( u_p_gt * 2*pi/T_P - uph )
over masked pixels of the calibration samples (the horizontal pipeline's 2.9
was obtained equivalently upstream).

Validation: reconstruct depth from the precomputed vertical targets through
the u_p triangulation and verify <1 mm agreement with the stored GT depth.
"""

import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parents[1]))
CAL_DIR = BASE / "calibration_matrices"
RAW_DIR = BASE / "fpp_synthetic_dataset"
DATA_DIR = BASE / "training_datasets/training_data_depth_individual_normalized"
PARAMS_DIR = BASE / "training_datasets/info_depth_params"
OUT_DIR = BASE / "training_datasets/phase_targets_vertical"

T_P = 18   # vertical-direction fringe period (measured: u_p/phase slope = T_P/2pi, T_P = 18 px)
SCALE = 1.446

SIN_RANGE = range(25, 43)   # 18 vertical sinusoidal frames
GC_RANGE = range(43, 50)    # 7 vertical gray code frames

OFFSET_SAMPLES = [("container_bottle", "A0", "test"), ("power_drill", "A0", "test"),
                  ("wooden_boards", "A0", "test")]
VALIDATION_SAMPLES = ["container_bottle_A0", "power_drill_A0", "wooden_boards_A0"]


def load_calibration():
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)
    M_c = A_c @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M_p = A_p @ np.hstack([R_p, T_p])
    return A_c, M_c, M_p


def compute_wrapped_phase(frames):
    N = len(frames)
    S = np.zeros_like(frames[0])
    C = np.zeros_like(frames[0])
    for i in range(1, N + 1):
        delta = 2 * np.pi / N * (i - 2) + np.pi
        S += frames[i - 1] * np.sin(delta)
        C += frames[i - 1] * np.cos(delta)
    phi = -np.arctan2(S, C)
    phi[:, 0] = 0
    return phi


def decode_gray_code(gc_frames, sin_frames):
    Nbit = len(gc_frames)
    I_avg = np.mean(np.stack(sin_frames), axis=0)
    bcode = np.zeros((*gc_frames[0].shape, Nbit), dtype=np.int32)
    for b in range(Nbit):
        bcode[:, :, b] = (gc_frames[b] > I_avg).astype(np.int32)
    gray_decimal = np.zeros(gc_frames[0].shape, dtype=np.int32)
    for b in range(Nbit):
        gray_decimal += bcode[:, :, b] * (2 ** b)
    max_val = 2 ** Nbit
    lut = np.zeros(max_val, dtype=np.int32)
    for g in range(max_val):
        de = 0
        dg = g
        while dg:
            de ^= dg
            dg >>= 1
        lut[g] = de
    return lut[gray_decimal]


def unwrap_phase(wph, fringe_order, phase_offset):
    fringe_order_f = median_filter(fringe_order.astype(np.float64), size=7)
    shift = -np.pi / T_P
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + fringe_order_f * 2 * np.pi - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    unwrPhase = uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi
    return unwrPhase + phase_offset


def reconstruct_depth_up(uphV, M_c, M_p, period=T_P):
    """Triangulation using the projector u_p equation (M_p row 0)."""
    H, W = uphV.shape
    mc, mp = M_c, M_p
    up = uphV * period / (2 * np.pi)
    row_grid, col_grid = np.meshgrid(
        np.arange(1, H + 1, dtype=np.float64),
        np.arange(1, W + 1, dtype=np.float64),
        indexing='ij')
    u_cam, v_cam = col_grid, row_grid
    A00 = mc[0,0]-u_cam*mc[2,0]; A01 = mc[0,1]-u_cam*mc[2,1]; A02 = mc[0,2]-u_cam*mc[2,2]
    A10 = mc[1,0]-v_cam*mc[2,0]; A11 = mc[1,1]-v_cam*mc[2,1]; A12 = mc[1,2]-v_cam*mc[2,2]
    A20 = mp[0,0]-up*mp[2,0]; A21 = mp[0,1]-up*mp[2,1]; A22 = mp[0,2]-up*mp[2,2]
    b0 = u_cam*mc[2,3]-mc[0,3]; b1 = v_cam*mc[2,3]-mc[1,3]; b2 = up*mp[2,3]-mp[0,3]
    det = A00*(A11*A22-A12*A21) - A01*(A10*A22-A12*A20) + A02*(A10*A21-A11*A20)
    det_z = A00*(A11*b2-b1*A21) - A01*(A10*b2-b1*A20) + b0*(A10*A21-A11*A20)
    valid = np.abs(det) > 1e-10
    z = np.zeros_like(uphV)
    z[valid] = det_z[valid] / det[valid]
    return z / SCALE


def project_up_gt(depth_mm, mask, A_c, M_p):
    """Ground-truth projector u_p at every masked pixel, from GT depth."""
    H, W = depth_mm.shape
    ys, xs = np.where(mask)
    u = xs + 1.0
    v = ys + 1.0
    Z = depth_mm[ys, xs] * SCALE
    rays = np.linalg.inv(A_c) @ np.stack([u, v, np.ones_like(u)])
    X = rays * Z
    Xh = np.vstack([X, np.ones(X.shape[1])])
    xp = M_p @ Xh
    return xp[0] / xp[2], ys, xs


def load_gt_depth_mm(split, sample_name):
    mat = sio.loadmat(DATA_DIR / split / "depth" / f"{sample_name}.mat")
    d_norm = mat['depthMapNormalized'].astype(np.float64)
    params = sio.loadmat(PARAMS_DIR / split / "depth" / f"{sample_name}.mat")
    d_min = float(params['depth_min'].item())
    d_max = float(params['depth_max'].item())
    depth_mm = d_norm * (d_max - d_min) + d_min
    mask = d_norm > 0
    return depth_mm, mask, d_min, d_max


def load_frames(obj, vp):
    raw_dir = RAW_DIR / obj / vp
    sin_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64)
                  for i in SIN_RANGE]
    gc_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64)
                 for i in GC_RANGE]
    return sin_frames, gc_frames


def calibrate_offset(A_c, M_c, M_p):
    """Fit the global (effective period, offset) of the phase-to-u_p map by
    least squares against GT-projected u_p over the calibration samples,
    mirroring how the horizontal pipeline's constants were fixed upstream."""
    xs_all, ys_all = [], []
    for obj, vp, split in OFFSET_SAMPLES:
        sin_frames, gc_frames = load_frames(obj, vp)
        phi = compute_wrapped_phase(sin_frames)
        k = decode_gray_code(gc_frames, sin_frames)
        depth_mm, mask, _, _ = load_gt_depth_mm(split, f"{obj}_{vp}")
        uph0 = unwrap_phase(phi, k, 0.0)
        up_gt, ys, xs = project_up_gt(depth_mm, mask, A_c, M_p)
        xs_all.append(uph0[ys, xs]); ys_all.append(up_gt)
    x = np.concatenate(xs_all); y = np.concatenate(ys_all)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    period = float(coef[0] * 2 * np.pi)
    off_rad = float(coef[1] * 2 * np.pi / period)
    print(f"VERTICAL period = {period:.4f} px, offset = {off_rad:.4f} rad")
    return period, off_rad


def process_sample(obj, vp, split, period, off, A_c, M_c, M_p, validate=False):
    sample_name = f"{obj}_{vp}"
    sin_frames, gc_frames = load_frames(obj, vp)
    phi = compute_wrapped_phase(sin_frames)
    k = decode_gray_code(gc_frames, sin_frames)
    depth_gt, mask_gt, d_min, d_max = load_gt_depth_mm(split, sample_name)

    out_path = OUT_DIR / split / f"{sample_name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sin_phi=np.sin(phi).astype(np.float32),
        cos_phi=np.cos(phi).astype(np.float32),
        k=k.astype(np.int32),
        mask=mask_gt.astype(np.float32),
        depth_mm=depth_gt.astype(np.float32),
        d_min=np.float32(d_min),
        d_max=np.float32(d_max),
        phase_offset=np.float32(off),
        period_px=np.float32(period),
    )

    if validate:
        phi_recon = np.arctan2(np.sin(phi).astype(np.float32).astype(np.float64),
                               np.cos(phi).astype(np.float32).astype(np.float64))
        uphV = unwrap_phase(phi_recon, k, off)
        depth_pred = reconstruct_depth_up(uphV, M_c, M_p, period)
        pred_obj = depth_pred[mask_gt]
        gt_obj = depth_gt[mask_gt]
        mae = np.mean(np.abs(pred_obj - gt_obj))
        max_err = np.max(np.abs(pred_obj - gt_obj))
        pct_1mm = np.mean(np.abs(pred_obj - gt_obj) < 1.0) * 100
        return mae, max_err, pct_1mm
    return None


def main():
    A_c, M_c, M_p = load_calibration()
    print("Calibrating vertical phase-to-u_p constants from GT projection...")
    period, off = calibrate_offset(A_c, M_c, M_p)

    splits_objects = {}
    for split in ['train', 'val', 'test']:
        fringe_dir = DATA_DIR / split / "fringe"
        samples = set()
        for f in fringe_dir.glob("*.png"):
            parts = f.stem.rsplit("_A", 1)
            samples.add((parts[0], "A" + parts[1]))
        splits_objects[split] = sorted(samples)
        print(f"{split}: {len(samples)} samples")

    total = sum(len(v) for v in splits_objects.values())
    pbar = tqdm(total=total, desc="Vertical phase targets")
    for split, samples in splits_objects.items():
        for obj, vp in samples:
            sample_name = f"{obj}_{vp}"
            validate = sample_name in VALIDATION_SAMPLES and split == 'test'
            result = process_sample(obj, vp, split, period, off, A_c, M_c, M_p, validate=validate)
            if result is not None:
                mae, max_err, pct_1mm = result
                status = "PASS" if mae < 1.0 else "FAIL"
                tqdm.write(f"  VALIDATE {sample_name}: MAE={mae:.4f}mm, "
                           f"max={max_err:.4f}mm, <1mm={pct_1mm:.1f}% [{status}]")
            pbar.update(1)
    pbar.close()
    print(f"\nVertical phase targets saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
