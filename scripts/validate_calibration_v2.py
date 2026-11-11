"""
Validation v2: Match MATLAB pipeline exactly.

Reproduces mainreconstrgraycode.m + fringeAnalysis2.m step by step.
"""

import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter
from pathlib import Path
from PIL import Image

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parents[1]))
CAL_DIR = BASE / "calibration_matrices"
RAW_DIR = BASE / "fpp_synthetic_dataset"
DEPTH_DIR = BASE / "training_datasets/training_data_depth_individual_normalized/test/depth"
PARAMS_DIR = BASE / "training_datasets/info_depth_params/test/depth"

T_P = 36
SCALE = 1.446
N_STEPS = 18
PHASE_OFFSET = 2.9  # manual offset from grayUnwrap

TEST_SAMPLES = ["container_bottle_A0", "power_drill_A0", "wooden_boards_A0"]


def load_calibration():
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)
    R_c = np.eye(3)
    T_c = np.zeros((3, 1))
    M_c = A_c @ np.hstack([R_c, T_c])
    M_p = A_p @ np.hstack([R_p, T_p])
    return M_c, M_p


def compute_wrapped_phase_matlab(frames):
    """
    Exact match of ComputeNStepPhase.m:
      S += I(:,:,i) * sin(2*pi/N * (i-2) + pi)   for i=1..N
      C += I(:,:,i) * cos(2*pi/N * (i-2) + pi)   for i=1..N
      ph = -atan2(S, C)
    """
    N = len(frames)
    S = np.zeros_like(frames[0])
    C = np.zeros_like(frames[0])
    for i_matlab in range(1, N + 1):  # i=1..18
        delta = 2 * np.pi / N * (i_matlab - 2) + np.pi
        S += frames[i_matlab - 1] * np.sin(delta)
        C += frames[i_matlab - 1] * np.cos(delta)
    phi = -np.arctan2(S, C)
    phi[0, :] = 0  # ph(1,:) = 0 in MATLAB
    return phi


def decode_gray_code_matlab(gc_frames, sin_frames):
    """
    Exact match of grayPhaseUnwrap.m:
      - Threshold = mean of sinusoidal frames (I_avg)
      - ALL Nbit gray code frames used
      - bi2de with LSB first (MATLAB default)
      - inversegrayCode for gray-to-binary
    """
    Nbit = len(gc_frames)
    H, W = gc_frames[0].shape

    # Threshold: mean of sinusoidal frames
    I_avg = np.mean(np.stack(sin_frames), axis=0)

    # Binarize
    bcode = np.zeros((H, W, Nbit), dtype=np.int32)
    for b in range(Nbit):
        bcode[:, :, b] = (gc_frames[b] > I_avg).astype(np.int32)

    # bi2de (LSB first): bcode[:,:,0] is bit 0 (LSB)
    gray_decimal = np.zeros((H, W), dtype=np.int32)
    for b in range(Nbit):
        gray_decimal += bcode[:, :, b] * (2 ** b)

    # Gray to binary (inversegrayCode)
    fringe_order = np.zeros_like(gray_decimal)
    max_val = 2 ** Nbit
    lut = np.zeros(max_val, dtype=np.int32)
    for g in range(max_val):
        de = 0
        de_gray = g
        while de_gray:
            de ^= de_gray
            de_gray >>= 1
        lut[g] = de

    for g in range(max_val):
        fringe_order[gray_decimal == g] = lut[g]

    return fringe_order


def gray_unwrap_matlab(wph, fringe_order, T):
    """
    Exact match of grayUnwrap in fringeAnalysis2.m:
      fringe_order = medfilt2(fringe_order, [7,7])
      shift = -pi/T
      wph_shifted = atan2(sin(wph+shift), cos(wph+shift))
      uph = wph_shifted + fringe_order * 2*pi - shift
      smooth + round correction
      uphH = unwrPhase + 2.9
    """
    fringe_order = median_filter(fringe_order.astype(np.float64), size=7)

    shift = -np.pi / T
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + fringe_order * 2 * np.pi
    uph = uph - shift

    # Median filter + round correction (3 passes)
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    unwrPhase = uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi

    uphH = unwrPhase + PHASE_OFFSET
    return uphH


def reconstruct_depth(uphH, M_c, M_p):
    """
    Exact match of uphToXYZ in fringeAnalysis2.m.
    Note the MATLAB convention: u = v1 (column), v = u1 (row).
    Uses P row 1 (v_p).
    """
    H, W = uphH.shape
    mc, mp = M_c, M_p

    # Projector coordinate
    up = uphH * T_P / (2 * np.pi)

    # Camera pixel grids
    # MATLAB: for u1=1:w, for v1=1:h → u=v1, v=u1
    # In MATLAB arrays: uphH(u1, v1) where u1=row, v1=col
    # Then: u = v1 (column index, 1-based), v = u1 (row index, 1-based)
    # In Python 0-based: u = col (0-based), v = row (0-based)
    # But MATLAB is 1-based, so u = col+1, v = row+1? No — MATLAB pixel coords
    # in the calibration are 0-based (standard camera convention).
    # Actually, MATLAB's loop: u=v1 (1-based column), v=u1 (1-based row)
    # The calibration matrices expect 0-based or 1-based? Let me check...
    # The intrinsic matrix has cx=480.78, cy=480.73 for a 960x960 image.
    # If 0-based, center pixel = (479.5, 479.5). cx=480.78 is close to 480.5
    # (0-based center). If 1-based, center = (480.5, 480.5), cx=480.78 is also
    # close. Let's use 1-based to match MATLAB.

    row_grid, col_grid = np.meshgrid(
        np.arange(1, H + 1, dtype=np.float64),
        np.arange(1, W + 1, dtype=np.float64),
        indexing='ij'
    )
    u_cam = col_grid  # u = v1 (column, 1-based)
    v_cam = row_grid   # v = u1 (row, 1-based)

    # Build A and b (using M_p row 1 for v_p)
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

    # Cramer's rule for z
    det = (A00 * (A11 * A22 - A12 * A21)
         - A01 * (A10 * A22 - A12 * A20)
         + A02 * (A10 * A21 - A11 * A20))

    det_z = (A00 * (A11 * b2 - b1 * A21)
           - A01 * (A10 * b2 - b1 * A20)
           + b0 * (A10 * A21 - A11 * A20))

    valid = np.abs(det) > 1e-10
    z = np.zeros_like(uphH)
    z[valid] = det_z[valid] / det[valid]

    # Apply scale factor (mainreconstrgraycode.m: z = z/scale)
    z = z / SCALE

    return z


def load_gt_depth_mm(sample_name):
    mat = sio.loadmat(DEPTH_DIR / f"{sample_name}.mat")
    d_norm = mat['depthMapNormalized'].astype(np.float64)
    params = sio.loadmat(PARAMS_DIR / f"{sample_name}.mat")
    d_min = float(params['depth_min'].item())
    d_max = float(params['depth_max'].item())
    depth_mm = d_norm * (d_max - d_min) + d_min
    mask = d_norm > 0
    return depth_mm, mask, d_min, d_max


def load_raw_frames(sample_name):
    parts = sample_name.rsplit("_A", 1)
    obj, vp = parts[0], "A" + parts[1]
    raw_dir = RAW_DIR / obj / vp

    sin_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64) for i in range(18)]
    gc_frames = [np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64) for i in range(18, 25)]
    return sin_frames, gc_frames


def main():
    M_c, M_p = load_calibration()

    for sample_name in TEST_SAMPLES:
        print("=" * 70)
        print(f"Sample: {sample_name}")
        print("=" * 70)

        depth_gt, mask, d_min, d_max = load_gt_depth_mm(sample_name)
        sin_frames, gc_frames = load_raw_frames(sample_name)

        # Step 1: Wrapped phase (ComputeNStepPhase)
        wph = compute_wrapped_phase_matlab(sin_frames)
        print(f"  Wrapped phase range: [{wph[mask].min():.4f}, {wph[mask].max():.4f}]")

        # Step 2: Gray code decode (grayPhaseUnwrap)
        fringe_order = decode_gray_code_matlab(gc_frames, sin_frames)
        print(f"  Fringe order range: [{fringe_order[mask].min()}, {fringe_order[mask].max()}]")

        # Step 3: Phase unwrapping (grayUnwrap)
        uphH = gray_unwrap_matlab(wph, fringe_order, T_P)
        print(f"  Unwrapped phase range: [{uphH[mask].min():.4f}, {uphH[mask].max():.4f}]")

        # Step 4: Reconstruct depth (uphToXYZ + scale)
        depth_pred = reconstruct_depth(uphH, M_c, M_p)

        # Compare
        pred_obj = depth_pred[mask]
        gt_obj = depth_gt[mask]
        mae = np.mean(np.abs(pred_obj - gt_obj))
        max_err = np.max(np.abs(pred_obj - gt_obj))
        pct_1mm = np.mean(np.abs(pred_obj - gt_obj) < 1.0) * 100

        print(f"\n  depth_gt  range (object): [{gt_obj.min():.2f}, {gt_obj.max():.2f}] mm")
        print(f"  depth_pred range (object): [{pred_obj.min():.2f}, {pred_obj.max():.2f}] mm")
        print(f"  MAE:       {mae:.4f} mm")
        print(f"  Max error: {max_err:.4f} mm")
        print(f"  % < 1mm:   {pct_1mm:.1f}%")

        status = "PASS" if mae < 1.0 else "FAIL"
        print(f"  Status:    {status}")
        print()


if __name__ == "__main__":
    main()
