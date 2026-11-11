"""
Validation script: calibration layer regression test.

Computes wrapped phase from 18 horizontal sinusoidal frames, decodes gray
codes for fringe order k, computes absolute phase, then reconstructs depth
via triangulation using calibration matrices. Compares to stored depth.mat.

Go/no-go gate: <1 mm MAE on object pixels for 3 representative test samples.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image

# =============================================================================
# Paths
# =============================================================================
import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parents[1]))
CAL_DIR = BASE / "calibration_matrices"
RAW_DIR = BASE / "fpp_synthetic_dataset"
DEPTH_DIR = BASE / "training_datasets/training_data_depth_individual_normalized/test/depth"
PARAMS_DIR = BASE / "training_datasets/info_depth_params/test/depth"

# System parameters (from spec)
T_P = 36        # projector pixels per fringe period
K_MAX = 32      # number of fringe orders (5 bits)
N_STEPS = 18    # phase-shifting steps

# Test samples: 3 representative objects
TEST_SAMPLES = ["container_bottle_A0", "power_drill_A0", "wooden_boards_A0"]

# Frame indices in the 52-image sequence:
#   0-17:  18 horizontal sinusoidal
#   18-24: 7 horizontal gray code
#   25-42: 18 vertical sinusoidal
#   43-49: 7 vertical gray code
#   50:    black
#   51:    white
HORIZ_SINUSOIDAL = list(range(0, 18))
HORIZ_GRAYCODE = list(range(18, 25))  # 7 patterns, but only 5 bits used
BLACK_FRAME = 50
WHITE_FRAME = 51


# =============================================================================
# Load calibration matrices
# =============================================================================
def load_calibration():
    """Load and build M_c (3x4) and M_p (3x4) projection matrices."""
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    R_c = np.loadtxt(CAL_DIR / "CamRotationMatrix.txt")
    T_c = np.loadtxt(CAL_DIR / "CamTranslationVector.txt").reshape(3, 1)
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)

    # Build projection matrices: M = A @ [R | T]
    RT_c = np.hstack([R_c, T_c])  # 3x4
    RT_p = np.hstack([R_p, T_p])  # 3x4
    M_c = A_c @ RT_c  # 3x4
    M_p = A_p @ RT_p  # 3x4

    print("Calibration matrices loaded:")
    print(f"  A_c:\n{A_c}")
    print(f"  A_p:\n{A_p}")
    print(f"  T_p: {T_p.flatten()}")
    print(f"  M_c shape: {M_c.shape}")
    print(f"  M_p shape: {M_p.shape}")
    return M_c, M_p


# =============================================================================
# Phase computation from 18-step phase shifting
# =============================================================================
def compute_wrapped_phase(frames):
    """
    Compute wrapped phase from N-step phase-shifted fringe images.

    phi(x,y) = -atan2(sum_n I_n sin(2*pi*n/N), sum_n I_n cos(2*pi*n/N))

    Args:
        frames: list of N grayscale images, each (H, W) float32

    Returns:
        phi: (H, W) wrapped phase in (-pi, pi]
    """
    N = len(frames)
    num = np.zeros_like(frames[0])
    den = np.zeros_like(frames[0])

    for n in range(N):
        delta_n = 2 * np.pi * (n + 1) / N  # 1-indexed: n=1..N
        num += frames[n] * np.sin(delta_n)
        den += frames[n] * np.cos(delta_n)

    phi = -np.arctan2(num, den)
    return phi


# =============================================================================
# Gray code decoding
# =============================================================================
def gray_to_binary(gray_val, n_bits):
    """Convert Gray code integer to binary integer."""
    binary = gray_val
    mask = gray_val >> 1
    while mask > 0:
        binary ^= mask
        mask >>= 1
    return binary


def decode_gray_codes(gc_frames, black_img, white_img, n_bits=5):
    """
    Decode gray code patterns to fringe order k.

    Args:
        gc_frames: list of gray code images (7 total, use first 5)
                   ordered MSB to LSB (descending stripe width)
        black_img: black reference image
        white_img: white reference image
        n_bits: number of bits to decode (5 for K_max=32)

    Returns:
        k: (H, W) integer array with fringe order in [0, 31]
    """
    threshold = (black_img.astype(np.float64) + white_img.astype(np.float64)) / 2.0

    # Per the spec: MSB-to-LSB = patterns [22, 21, 20, 19, 18] in 1-indexed
    # = [4, 3, 2, 1, 0] in 0-indexed within the gc_frames list
    # (gc_frames[0] = frame 18 = pattern with narrowest stripes = LSB candidate)
    # (gc_frames[4] = frame 22 = pattern with widest stripes = MSB candidate)
    # Spec says descending stripe width = MSB first, so order is [4, 3, 2, 1, 0]
    gray_code_val = np.zeros(gc_frames[0].shape, dtype=np.int32)
    for bit_idx in range(n_bits):
        # MSB first: gc_frames[4] is MSB (widest stripes), gc_frames[0] is LSB
        frame_idx = (n_bits - 1) - bit_idx  # 4, 3, 2, 1, 0
        bit_val = (gc_frames[frame_idx] > threshold).astype(np.int32)
        gray_code_val |= (bit_val << (n_bits - 1 - bit_idx))

    # Convert Gray code to binary
    k = np.zeros_like(gray_code_val)
    for val in range(K_MAX):
        mask = gray_code_val == val
        k[mask] = gray_to_binary(val, n_bits)

    return k


# =============================================================================
# Triangulation: phase + calibration -> depth
# =============================================================================
def reconstruct_depth(phi, k, M_c, M_p, use_row=1):
    """
    Reconstruct depth from absolute phase using triangulation.

    Args:
        phi: (H, W) wrapped phase
        k: (H, W) fringe order
        M_c: (3, 4) camera projection matrix
        M_p: (3, 4) projector projection matrix
        use_row: which row of M_p to use (0 for u_p, 1 for v_p)

    Returns:
        depth: (H, W) depth in mm (z coordinate)
    """
    H, W = phi.shape

    # Absolute phase
    Phi = phi + 2 * np.pi * k.astype(np.float64)

    # Convert to projector coordinate
    p_coord = Phi * T_P / (2 * np.pi)

    # Build pixel coordinate grids (camera)
    v_c, u_c = np.meshgrid(np.arange(H, dtype=np.float64),
                            np.arange(W, dtype=np.float64), indexing='ij')

    # Extract M_c elements
    mc = M_c
    # Extract M_p elements
    mp = M_p
    r = use_row  # which row of M_p to use

    # Build A matrix and b vector for each pixel
    # A[0]: m_c11 - u_c*m_c31, m_c12 - u_c*m_c32, m_c13 - u_c*m_c33
    # A[1]: m_c21 - v_c*m_c31, m_c22 - v_c*m_c32, m_c23 - v_c*m_c33
    # A[2]: m_p[r,0] - p*m_p[2,0], m_p[r,1] - p*m_p[2,1], m_p[r,2] - p*m_p[2,2]

    A00 = mc[0, 0] - u_c * mc[2, 0]
    A01 = mc[0, 1] - u_c * mc[2, 1]
    A02 = mc[0, 2] - u_c * mc[2, 2]

    A10 = mc[1, 0] - v_c * mc[2, 0]
    A11 = mc[1, 1] - v_c * mc[2, 1]
    A12 = mc[1, 2] - v_c * mc[2, 2]

    A20 = mp[r, 0] - p_coord * mp[2, 0]
    A21 = mp[r, 1] - p_coord * mp[2, 1]
    A22 = mp[r, 2] - p_coord * mp[2, 2]

    b0 = u_c * mc[2, 3] - mc[0, 3]
    b1 = v_c * mc[2, 3] - mc[1, 3]
    b2 = p_coord * mp[2, 3] - mp[r, 3]

    # Solve 3x3 system per pixel using Cramer's rule (vectorized)
    # det(A)
    det = (A00 * (A11 * A22 - A12 * A21)
         - A01 * (A10 * A22 - A12 * A20)
         + A02 * (A10 * A21 - A11 * A20))

    # z = det(A_z) / det(A) where A_z replaces column 2 with b
    det_z = (A00 * (A11 * b2 - b1 * A21)
           - A01 * (A10 * b2 - b1 * A20)
           + b0 * (A10 * A21 - A11 * A20))

    # Avoid division by zero
    valid = np.abs(det) > 1e-10
    depth = np.zeros_like(phi)
    depth[valid] = det_z[valid] / det[valid]

    return depth


# =============================================================================
# Load raw frames for one sample
# =============================================================================
def load_raw_frames(obj_name):
    """Load all relevant raw frames for an object-viewpoint."""
    # Parse object name and viewpoint: e.g., "container_bottle_A0"
    parts = obj_name.rsplit("_A", 1)
    obj = parts[0]
    vp = "A" + parts[1]

    raw_dir = RAW_DIR / obj / vp

    # Load 18 horizontal sinusoidal frames
    sin_frames = []
    for i in HORIZ_SINUSOIDAL:
        img = np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64)
        sin_frames.append(img)

    # Load horizontal gray code frames (indices 18-24, 7 patterns)
    gc_frames = []
    for i in HORIZ_GRAYCODE:
        img = np.array(Image.open(raw_dir / f"A_{i}.png").convert('L'), dtype=np.float64)
        gc_frames.append(img)

    # Load black and white references
    black = np.array(Image.open(raw_dir / f"A_{BLACK_FRAME}.png").convert('L'), dtype=np.float64)
    white = np.array(Image.open(raw_dir / f"A_{WHITE_FRAME}.png").convert('L'), dtype=np.float64)

    return sin_frames, gc_frames, black, white


# =============================================================================
# Load ground truth depth in mm
# =============================================================================
def load_gt_depth_mm(sample_name):
    """Load ground truth depth map and denormalize to mm."""
    mat = sio.loadmat(DEPTH_DIR / f"{sample_name}.mat")
    d_norm = mat['depthMapNormalized'].astype(np.float64)

    params = sio.loadmat(PARAMS_DIR / f"{sample_name}.mat")
    d_min = float(params['depth_min'].item())
    d_max = float(params['depth_max'].item())

    depth_mm = d_norm * (d_max - d_min) + d_min
    mask = d_norm > 0  # object pixels

    return depth_mm, mask, d_min, d_max


# =============================================================================
# Main validation
# =============================================================================
def main():
    M_c, M_p = load_calibration()
    print()

    all_passed = True

    for sample_name in TEST_SAMPLES:
        print("=" * 70)
        print(f"Sample: {sample_name}")
        print("=" * 70)

        # Load ground truth
        depth_gt_mm, mask, d_min, d_max = load_gt_depth_mm(sample_name)
        print(f"  GT depth range (object): [{depth_gt_mm[mask].min():.2f}, {depth_gt_mm[mask].max():.2f}] mm")
        print(f"  Object pixels: {mask.sum()}")

        # Load raw frames
        sin_frames, gc_frames, black, white = load_raw_frames(sample_name)
        print(f"  Loaded {len(sin_frames)} sinusoidal + {len(gc_frames)} gray code frames")

        # Compute wrapped phase
        phi = compute_wrapped_phase(sin_frames)
        print(f"  Wrapped phase range: [{phi.min():.4f}, {phi.max():.4f}] rad")

        # Decode gray codes
        k = decode_gray_codes(gc_frames, black, white, n_bits=5)
        print(f"  Fringe order k range: [{k.min()}, {k.max()}]")
        print(f"  Unique k values: {np.unique(k).tolist()}")

        # Try both M_p rows to find the correct convention
        for use_row in [0, 1]:
            depth_pred = reconstruct_depth(phi, k, M_c, M_p, use_row=use_row)

            # Compare on object pixels
            pred_obj = depth_pred[mask]
            gt_obj = depth_gt_mm[mask]

            mae = np.mean(np.abs(pred_obj - gt_obj))
            max_err = np.max(np.abs(pred_obj - gt_obj))

            print(f"\n  Row {use_row}: pred depth range (object): [{pred_obj.min():.2f}, {pred_obj.max():.2f}] mm")
            print(f"  Row {use_row}: MAE = {mae:.6f} mm, Max error = {max_err:.6f} mm")

            if mae < 1.0:
                print(f"  Row {use_row}: *** PASS *** (MAE < 1 mm)")
            else:
                print(f"  Row {use_row}: FAIL (MAE >= 1 mm)")

        print()

    # Final summary with the correct row
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    # Re-run with the row that works
    for use_row in [0, 1]:
        print(f"\n--- Using M_p row {use_row} ---")
        passed = True
        for sample_name in TEST_SAMPLES:
            depth_gt_mm, mask, d_min, d_max = load_gt_depth_mm(sample_name)
            sin_frames, gc_frames, black, white = load_raw_frames(sample_name)
            phi = compute_wrapped_phase(sin_frames)
            k = decode_gray_codes(gc_frames, black, white, n_bits=5)
            depth_pred = reconstruct_depth(phi, k, M_c, M_p, use_row=use_row)

            pred_obj = depth_pred[mask]
            gt_obj = depth_gt_mm[mask]
            mae = np.mean(np.abs(pred_obj - gt_obj))
            max_err = np.max(np.abs(pred_obj - gt_obj))
            pct_within_1mm = np.mean(np.abs(pred_obj - gt_obj) < 1.0) * 100

            status = "PASS" if mae < 1.0 else "FAIL"
            if mae >= 1.0:
                passed = False

            print(f"  {sample_name:30s}  MAE={mae:.6f} mm  Max={max_err:.6f} mm  "
                  f"<1mm: {pct_within_1mm:.1f}%  [{status}]")

        if passed:
            print(f"\n  *** ALL SAMPLES PASS with M_p row {use_row} ***")
        else:
            print(f"\n  Some samples FAIL with M_p row {use_row}")


if __name__ == "__main__":
    main()
