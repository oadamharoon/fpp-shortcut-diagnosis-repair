"""
Precompute phase targets for all samples in the dataset.

For each sample:
  1. Compute wrapped phase from 18 horizontal sinusoidal frames
  2. Decode gray code fringe order from 7 gray code patterns
  3. Store sin(phi), cos(phi), k, and mask as .npz

Validation: reconstruct depth from precomputed targets through the
calibration layer and verify <1mm agreement with stored depth.mat.
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
OUT_DIR = BASE / "training_datasets/phase_targets"

T_P = 36
SCALE = 1.446
PHASE_OFFSET = 2.9

VALIDATION_SAMPLES = ["container_bottle_A0", "power_drill_A0", "wooden_boards_A0"]


def load_calibration():
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)
    M_c = A_c @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M_p = A_p @ np.hstack([R_p, T_p])
    return M_c, M_p


def compute_wrapped_phase(frames):
    """ComputeNStepPhase.m: delta = 2*pi/N*(i-2)+pi, phi = -atan2(S, C)"""
    N = len(frames)
    S = np.zeros_like(frames[0])
    C = np.zeros_like(frames[0])
    for i in range(1, N + 1):
        delta = 2 * np.pi / N * (i - 2) + np.pi
        S += frames[i - 1] * np.sin(delta)
        C += frames[i - 1] * np.cos(delta)
    phi = -np.arctan2(S, C)
    phi[0, :] = 0
    return phi


def decode_gray_code(gc_frames, sin_frames):
    """grayPhaseUnwrap.m: threshold=mean(sinusoidal), LSB first, 7 bits."""
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

    fringe_order = lut[gray_decimal]
    return fringe_order


def unwrap_phase(wph, fringe_order):
    """grayUnwrap: median filter, shift, offset, smoothing."""
    fringe_order_f = median_filter(fringe_order.astype(np.float64), size=7)

    shift = -np.pi / T_P
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + fringe_order_f * 2 * np.pi - shift

    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    unwrPhase = uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi

    uphH = unwrPhase + PHASE_OFFSET
    return uphH


def reconstruct_depth(uphH, M_c, M_p):
    """uphToXYZ + scale, 1-based pixel coords."""
    H, W = uphH.shape
    mc, mp = M_c, M_p
    up = uphH * T_P / (2 * np.pi)

    row_grid, col_grid = np.meshgrid(
        np.arange(1, H + 1, dtype=np.float64),
        np.arange(1, W + 1, dtype=np.float64),
        indexing='ij'
    )
    u_cam, v_cam = col_grid, row_grid

    A00 = mc[0,0]-u_cam*mc[2,0]; A01 = mc[0,1]-u_cam*mc[2,1]; A02 = mc[0,2]-u_cam*mc[2,2]
    A10 = mc[1,0]-v_cam*mc[2,0]; A11 = mc[1,1]-v_cam*mc[2,1]; A12 = mc[1,2]-v_cam*mc[2,2]
    A20 = mp[1,0]-up*mp[2,0]; A21 = mp[1,1]-up*mp[2,1]; A22 = mp[1,2]-up*mp[2,2]
    b0 = u_cam*mc[2,3]-mc[0,3]; b1 = v_cam*mc[2,3]-mc[1,3]; b2 = up*mp[2,3]-mp[1,3]

    det = A00*(A11*A22-A12*A21) - A01*(A10*A22-A12*A20) + A02*(A10*A21-A11*A20)
    det_z = A00*(A11*b2-b1*A21) - A01*(A10*b2-b1*A20) + b0*(A10*A21-A11*A20)
    valid = np.abs(det) > 1e-10
    z = np.zeros_like(uphH)
    z[valid] = det_z[valid] / det[valid]
    return z / SCALE


def load_gt_depth_mm(split, sample_name):
    mat = sio.loadmat(DATA_DIR / split / "depth" / f"{sample_name}.mat")
    d_norm = mat['depthMapNormalized'].astype(np.float64)
    params = sio.loadmat(PARAMS_DIR / split / "depth" / f"{sample_name}.mat")
    d_min = float(params['depth_min'].item())
    d_max = float(params['depth_max'].item())
    depth_mm = d_norm * (d_max - d_min) + d_min
    mask = d_norm > 0
    return depth_mm, mask, d_min, d_max


def process_sample(obj, vp, split, M_c, M_p, validate=False):
    """Process one sample: compute phase targets, optionally validate."""
    sample_name = f"{obj}_{vp}"
    raw_dir = RAW_DIR / obj / vp

    # Load frames
    sin_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64) for i in range(18)]
    gc_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64) for i in range(18, 25)]

    # Compute wrapped phase
    phi = compute_wrapped_phase(sin_frames)

    # Decode gray code
    k = decode_gray_code(gc_frames, sin_frames)

    # Load GT depth for mask
    depth_gt, mask_gt, d_min, d_max = load_gt_depth_mm(split, sample_name)

    # Save targets
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
    )

    # Validation: reconstruct depth from precomputed targets
    if validate:
        # Recover phi from sin/cos
        phi_recon = np.arctan2(
            np.sin(phi).astype(np.float32).astype(np.float64),
            np.cos(phi).astype(np.float32).astype(np.float64)
        )
        uphH = unwrap_phase(phi_recon, k)
        depth_pred = reconstruct_depth(uphH, M_c, M_p)

        pred_obj = depth_pred[mask_gt]
        gt_obj = depth_gt[mask_gt]
        mae = np.mean(np.abs(pred_obj - gt_obj))
        max_err = np.max(np.abs(pred_obj - gt_obj))
        pct_1mm = np.mean(np.abs(pred_obj - gt_obj) < 1.0) * 100

        return mae, max_err, pct_1mm

    return None


def main():
    M_c, M_p = load_calibration()

    # Gather all samples across splits
    splits_objects = {}
    for split in ['train', 'val', 'test']:
        fringe_dir = DATA_DIR / split / "fringe"
        samples = set()
        for f in fringe_dir.glob("*.png"):
            parts = f.stem.rsplit("_A", 1)
            obj, vp = parts[0], "A" + parts[1]
            samples.add((obj, vp))
        splits_objects[split] = sorted(samples)
        print(f"{split}: {len(samples)} samples")

    # Process all samples
    total = sum(len(v) for v in splits_objects.values())
    print(f"\nProcessing {total} samples...")

    pbar = tqdm(total=total, desc="Precomputing phase targets")
    for split, samples in splits_objects.items():
        for obj, vp in samples:
            sample_name = f"{obj}_{vp}"
            validate = sample_name in VALIDATION_SAMPLES and split == 'test'
            result = process_sample(obj, vp, split, M_c, M_p, validate=validate)
            if result is not None:
                mae, max_err, pct_1mm = result
                status = "PASS" if mae < 1.0 else "FAIL"
                tqdm.write(f"  VALIDATE {sample_name}: MAE={mae:.4f}mm, "
                          f"max={max_err:.4f}mm, <1mm={pct_1mm:.1f}% [{status}]")
            pbar.update(1)
    pbar.close()

    # Final validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY (from precomputed sin/cos/k targets)")
    print("=" * 70)
    for sample_name in VALIDATION_SAMPLES:
        npz = np.load(OUT_DIR / "test" / f"{sample_name}.npz")
        phi_recon = np.arctan2(
            npz['sin_phi'].astype(np.float64),
            npz['cos_phi'].astype(np.float64)
        )
        k = npz['k']
        mask = npz['mask'] > 0
        depth_gt = npz['depth_mm'].astype(np.float64)

        uphH = unwrap_phase(phi_recon, k)
        depth_pred = reconstruct_depth(uphH, M_c, M_p)

        mae = np.mean(np.abs(depth_pred[mask] - depth_gt[mask]))
        max_err = np.max(np.abs(depth_pred[mask] - depth_gt[mask]))
        pct = np.mean(np.abs(depth_pred[mask] - depth_gt[mask]) < 1.0) * 100
        status = "PASS" if mae < 1.0 else "FAIL"
        print(f"  {sample_name:30s}  MAE={mae:.4f}mm  max={max_err:.4f}mm  <1mm={pct:.1f}%  [{status}]")

    print(f"\nPhase targets saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
