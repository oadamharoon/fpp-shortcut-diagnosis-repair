"""
Run the paper's PhiCalNet (checkpoints_phase_A1_refined_v2) on the flat-plane
capture and produce a depth-prediction image.

This mirrors regen_phaseunet_viz.py exactly for the model -> unwrap -> calibration
path. The only difference is the input source: the flat plane has no precomputed
phase-target .npz, so we load the raw frames directly and decode the oracle
fringe order k from the gray-code frames the same way the dataset pipeline does
(precompute_phase_targets.decode_gray_code + the loader's 7x7 median filter and
clamp to [0, K_max-1]). PhiCalNet uses GT/oracle k, so this is the faithful eval.

No ground-truth depth is required or used here -- this outputs only the model's
depth prediction (plus uniformity stats, which is what the flat-plane test cares
about).

Frame convention (same as the dataset): P_0..P_17 are the 18 sinusoidal frames
(P_0 is also the single-shot network input), P_18..P_24 are the 7 gray-code
frames.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer import CalibrationLayer

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
CAL_DIR = BASE / "calibration_matrices"

T_P = 36.0
K_MAX = 32

# Colormaps. Depth uses 'jet' to match the UNet flat-plane depth figure
# (rainbow). sin/cos use a diverging red(high)/blue(low) map to match the
# existing predicted-phase figure.
DEPTH_CMAP = 'jet'
SINCOS_CMAP = 'RdBu_r'


def decode_gray_code(gc_frames, sin_frames):
    """grayPhaseUnwrap.m: threshold=mean(sinusoidal), LSB first, 7 bits.

    Verbatim from scripts/precompute_phase_targets.py so the oracle k matches
    the k baked into the training/eval phase targets.
    """
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


def full_unwrap(sin_np, cos_np, k_np):
    """Identical to regen_phaseunet_viz.full_unwrap."""
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def load_frame(path):
    return np.array(Image.open(path).convert('L'), dtype=np.float64)


def save_native(data, cmap, vmin, vmax, out_path):
    """Save at exactly data.shape pixels (1:1, no shrinking, no colorbar)."""
    plt.imsave(out_path, data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    print(f"Saved (native {data.shape[0]}x{data.shape[1]}): {out_path}")


def save_panel(data, cmap, vmin, vmax, out_path, cbar_label=None):
    """Save a clean image panel with optional colorbar, no title/axes.

    Rendered large + high-dpi with nearest-neighbour so the 960-px field is
    never downsampled (avoids the moire that smaller panels produce); each
    data pixel maps to >=1 screen pixel and stays crisp.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.axis('off')
    if cbar_label is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    plane_dir = Path(args.plane_dir)

    # --- Frames -----------------------------------------------------------
    sin_frames = [load_frame(plane_dir / f"P_{i}.png") for i in range(18)]
    gc_frames = [load_frame(plane_dir / f"P_{i}.png") for i in range(18, 25)]
    H, W = sin_frames[0].shape
    print(f"Flat-plane frames: {plane_dir}  size={H}x{W}")

    # Single-shot network input is the first frame, normalized like the loader.
    fringe = (sin_frames[0] / 255.0).astype(np.float32)
    fringe_t = torch.from_numpy(fringe).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    # --- Oracle k from gray code (matches dataset loader handling) --------
    k_raw = decode_gray_code(gc_frames, sin_frames).astype(np.float64)
    k_filtered = median_filter(k_raw, size=7)
    k_gt = np.clip(k_filtered, 0, K_MAX - 1).astype(np.int64)

    # --- Model (paper PhiCalNet) ------------------------------------------
    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    cal_layer = CalibrationLayer(str(CAL_DIR), image_size=(H, W)).to(device)

    with torch.no_grad():
        out = model(fringe_t)
    sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
    cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)

    # --- Unwrap + calibration -> depth (identical to regen path) ----------
    uph = full_unwrap(sin_hat, cos_hat, k_gt)
    uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
    depth_pred = cal_layer(uph_t).squeeze().cpu().numpy()

    # --- Stats (flat-plane uniformity is the quantity of interest) --------
    finite = np.isfinite(depth_pred)
    dp = depth_pred[finite]
    print(f"Predicted depth (mm): mean={dp.mean():.2f}  std={dp.std():.2f}  "
          f"min={dp.min():.2f}  max={dp.max():.2f}  "
          f"p5={np.percentile(dp,5):.2f}  p95={np.percentile(dp,95):.2f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{args.prefix}_depth_mm.npy", depth_pred)

    # --- Four separate clean panels (no titles; LaTeX adds subcaptions) ----
    d_vmin = args.vmin if args.vmin is not None else float(dp.min())
    d_vmax = args.vmax if args.vmax is not None else float(dp.max())
    print(f"Depth color range: [{d_vmin:.1f}, {d_vmax:.1f}] mm")

    save_panel(fringe, 'gray', float(fringe.min()), float(fringe.max()),
               save_dir / f"{args.prefix}_fringe.png")
    save_panel(sin_hat, SINCOS_CMAP, -1.0, 1.0,
               save_dir / f"{args.prefix}_sin.png", cbar_label=r'$\sin\phi$')
    save_panel(cos_hat, SINCOS_CMAP, -1.0, 1.0,
               save_dir / f"{args.prefix}_cos.png", cbar_label=r'$\cos\phi$')
    save_panel(depth_pred, DEPTH_CMAP, d_vmin, d_vmax,
               save_dir / f"{args.prefix}_depth.png", cbar_label='Depth')

    # Native-resolution versions: 1:1 pixel mapping, no colorbar, no shrinking.
    save_native(fringe, 'gray', float(fringe.min()), float(fringe.max()),
                save_dir / f"{args.prefix}_fringe_native.png")
    save_native(sin_hat, SINCOS_CMAP, -1.0, 1.0,
                save_dir / f"{args.prefix}_sin_native.png")
    save_native(cos_hat, SINCOS_CMAP, -1.0, 1.0,
                save_dir / f"{args.prefix}_cos_native.png")
    save_native(depth_pred, DEPTH_CMAP, d_vmin, d_vmax,
                save_dir / f"{args.prefix}_depth_native.png")
    print(f"Saved: {save_dir / (args.prefix + '_depth_mm.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_phase_A1_refined_v2/best_model.pth")
    parser.add_argument("--plane_dir", type=str,
                        default=str(BASE / "training_datasets/flat_in_range/P1"))
    parser.add_argument("--save_dir", type=str,
                        default="visualizations_flat_plane")
    parser.add_argument("--prefix", type=str, default="flat_plane_P1")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Depth colorbar min (mm); default = data min")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Depth colorbar max (mm); default = data max")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
