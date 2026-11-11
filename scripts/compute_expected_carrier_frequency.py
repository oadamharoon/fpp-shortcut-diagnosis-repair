"""
Compute expected carrier frequency for FTP architecture.

Synthesizes a flat plane at z=1800mm (midpoint of depth range), computes
what the fringe image would look like in the camera, and finds the carrier
frequency peak in the FFT.

Reports (fu0, fv0) in pixel-frequency bin units for a 960x960 FFT.
"""

import numpy as np
import yaml
from pathlib import Path

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parents[1]))
CAL_DIR = BASE / "calibration_matrices"
OUT_DIR = BASE / "FPP-ML-Benchmarking/configs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_P = 36
SCALE = 1.446
PHASE_OFFSET = 2.9
H, W = 960, 960


def main():
    # Load calibration
    A_c = np.loadtxt(CAL_DIR / "CamIntrinsicMatrix.txt")
    A_p = np.loadtxt(CAL_DIR / "ProjIntrinsicMatrix.txt")
    R_p = np.loadtxt(CAL_DIR / "ProjRotationMatrix.txt")
    T_p = np.loadtxt(CAL_DIR / "ProjTranslationVector.txt").reshape(3, 1)
    M_p = A_p @ np.hstack([R_p, T_p])

    fx, fy, cx, cy = A_c[0, 0], A_c[1, 1], A_c[0, 2], A_c[1, 2]

    # Flat plane at z = 1800mm (pre-scale: z_world = 1800 * SCALE)
    # The stored depth is z_world / SCALE, so z_stored = 1800mm corresponds
    # to z_world = 1800 * SCALE = 2602.8mm in the triangulation space
    z_world = 1800.0 * SCALE

    # For each camera pixel, back-project to 3D at z_world
    # Camera is at origin, so:
    #   x = (u - cx) * z / fx,  y = (v - cy) * z / fy
    # Using 1-based coords to match MATLAB
    row_grid, col_grid = np.meshgrid(
        np.arange(1, H + 1, dtype=np.float64),
        np.arange(1, W + 1, dtype=np.float64),
        indexing='ij'
    )

    x_3d = (col_grid - cx) * z_world / fx
    y_3d = (row_grid - cy) * z_world / fy

    # Project through M_p to get v_p
    proj_v = M_p[1, 0] * x_3d + M_p[1, 1] * y_3d + M_p[1, 2] * z_world + M_p[1, 3]
    proj_w = M_p[2, 0] * x_3d + M_p[2, 1] * y_3d + M_p[2, 2] * z_world + M_p[2, 3]
    v_p = proj_v / proj_w

    print(f"Flat plane at z_stored=1800mm (z_world={z_world:.1f}mm)")
    print(f"  v_p range: [{v_p.min():.2f}, {v_p.max():.2f}]")
    print(f"  v_p at center: {v_p[H//2, W//2]:.2f}")

    # Synthesize ideal fringe image: I = 0.5 * (1 + cos(2*pi*v_p/T_P))
    fringe_synthetic = 0.5 * (1.0 + np.cos(2.0 * np.pi * v_p / T_P))

    print(f"  Synthetic fringe range: [{fringe_synthetic.min():.4f}, {fringe_synthetic.max():.4f}]")

    # FFT and find carrier peak
    F = np.fft.fftshift(np.fft.fft2(fringe_synthetic))
    mag = np.abs(F)

    # Zero out DC region
    cy_fft, cx_fft = H // 2, W // 2
    mag[cy_fft - 10:cy_fft + 10, cx_fft - 10:cx_fft + 10] = 0

    # Find peak (only in upper half to get one sideband)
    mag_upper = mag.copy()
    mag_upper[cy_fft:, :] = 0  # zero lower half
    peak = np.unravel_index(np.argmax(mag_upper), mag_upper.shape)

    fu0 = peak[0] - cy_fft  # offset from center in row direction
    fv0 = peak[1] - cx_fft  # offset from center in col direction

    print(f"\n  FFT peak at: ({peak[0]}, {peak[1]})")
    print(f"  Carrier frequency offset from center: fu0={fu0}, fv0={fv0}")
    print(f"  Camera-space period: T_c = {H / abs(fu0) if fu0 != 0 else 'inf':.2f} pixels (row direction)")
    if fv0 != 0:
        print(f"  Camera-space period (col): {W / abs(fv0):.2f} pixels")
    print(f"  Carrier is along: {'rows (vertical)' if abs(fu0) > abs(fv0) else 'cols (horizontal)'}")

    # Also check with actual fringe image for comparison
    from PIL import Image
    real_fringe = np.array(Image.open(
        BASE / "fpp_synthetic_dataset/container_bottle/A0/A_0.png"
    ).convert('L'), dtype=np.float64) / 255.0

    F_real = np.fft.fftshift(np.fft.fft2(real_fringe))
    mag_real = np.abs(F_real)
    mag_real[cy_fft - 10:cy_fft + 10, cx_fft - 10:cx_fft + 10] = 0
    mag_real_upper = mag_real.copy()
    mag_real_upper[cy_fft:, :] = 0
    peak_real = np.unravel_index(np.argmax(mag_real_upper), mag_real_upper.shape)

    print(f"\n  Real fringe (container_bottle A_0) FFT peak: ({peak_real[0]}, {peak_real[1]})")
    print(f"  Real carrier offset: fu0={peak_real[0]-cy_fft}, fv0={peak_real[1]-cx_fft}")

    # Save config
    config = {
        "T_p": int(T_P),
        "K_max": 32,
        "scale_factor": float(SCALE),
        "phase_offset": float(PHASE_OFFSET),
        "image_resolution": [H, W],
        "projector_resolution": [1140, 912],
        "expected_carrier_frequency": {
            "fu0": int(fu0),
            "fv0": int(fv0),
            "camera_space_period": float(H / abs(fu0)) if fu0 != 0 else None,
        },
        "calibration_dir": str(CAL_DIR),
        "n_phase_steps": 18,
        "n_graycode_patterns": 7,
        "phase_formula": "delta = 2*pi/N*(i-2)+pi for i=1..N; phi=-atan2(S,C)",
        "graycode_threshold": "mean of sinusoidal frames",
        "graycode_bit_order": "LSB first (A_18=bit0, A_24=bit6)",
    }

    yaml_path = OUT_DIR / "ftp_architecture.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig saved to: {yaml_path}")

    # Save synthetic flat-plane fringe for visual inspection
    from PIL import Image as PILImage
    fringe_uint8 = (fringe_synthetic * 255).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(fringe_uint8, mode='L').save(OUT_DIR / "synthetic_flat_plane_fringe.png")
    print(f"Synthetic fringe saved to: {OUT_DIR / 'synthetic_flat_plane_fringe.png'}")


if __name__ == "__main__":
    main()
