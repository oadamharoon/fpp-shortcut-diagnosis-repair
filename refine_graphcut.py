"""
Test-time fringe-order refinement of PhiCalNet (A1) outputs via
multi-label MRF / alpha-expansion graph cuts.

Setup:
  PhiCalNet predicts wrapped phase (sin, cos), uses oracle k, applies a
  median-filter "snap to nearest 2*pi multiple" trick (full_unwrap) to
  produce an unwrapped phase Phi_init. The §5.5 failure mode is
  wrap-boundary stripes where this trick locally agrees with itself but
  globally disagrees with the true unwrap by 2*pi.

We refine Phi_init by running an MRF over per-pixel integer labels
representing 2*pi shifts:

    Phi_refined(u, v) = Phi_init(u, v) + 2*pi * label(u, v)

where label in {-2, -1, 0, +1, +2}. Energy:

    E(label) = sum_p   alpha * |label(p)|                           # unary: stay close to PhiCalNet
             + sum_pq  lambda * w(p, q) * |label(p) - label(q)|     # pairwise: smoothness

w(p, q) is an optional edge-aware weight; by default w == 1.

Solved with alpha-expansion via gco-wrapper (Boykov-Kolmogorov GCO).

Eval: object MAE / RMSE before vs after refinement, computed via
the existing differentiable CalibrationLayer.

Usage:
    python -u refine_graphcut.py --alpha 1.0 --lam 5.0
    python -u refine_graphcut.py --samples 0,5,12 --alpha 0.5 --lam 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm

import gco

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from models.calibration_layer import CalibrationLayer
from models.phase_unet import PhaseUNetFPP

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"
CKPT = Path("checkpoints_phase_A1_refined_v2/best_model.pth")

T_P = 36.0
K_MAX = 32


def full_unwrap(sin_np, cos_np, k_np):
    """PhiCalNet's reference unwrapper (matches conformal_snapshot.py)."""
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def refine_phase(phi_init, mask, num_shifts, alpha, lam,
                 edge_aware=False, edge_data=None, edge_sigma=0.5,
                 n_iter=5):
    """Run alpha-expansion graph cut over per-pixel 2*pi shift labels.

    phi_init: (H, W) float64, initial unwrapped phase.
    mask: (H, W) bool, object mask. Non-object pixels keep their initial
        phase (the MRF still operates over the full grid; non-object
        pixels just get unary cost = 0 across all labels and pairwise
        coupling that effectively floats them).
    num_shifts: integer >= 1; labels are {-num_shifts, ..., +num_shifts}.
    alpha: unary coefficient (cost per |shift|).
    lam: pairwise coefficient (cost per |label diff| between neighbors).
    edge_aware: if True, scale pairwise cost by exp(-|grad(edge_data)|^2 / 2 sigma^2).
        Used to relax smoothness across true depth discontinuities.
    edge_data: (H, W) array used for edge weight (typically wrapped phase
        gradient or input fringe gradient). Required if edge_aware.
    n_iter: alpha-expansion sweep iterations.

    Returns:
        labels: (H, W) int, optimal shift labels.
        phi_refined: (H, W) float64, phi_init + 2*pi * labels.
    """
    H, W = phi_init.shape
    n_labels = 2 * num_shifts + 1
    shift_values = np.arange(-num_shifts, num_shifts + 1)  # e.g. [-2,-1,0,1,2]

    # Unary cost: cost(p, label) = alpha * |label| for object pixels;
    # 0 for background (so background floats to whatever the smoothness term prefers).
    # Shape: (H*W, n_labels).
    label_costs = alpha * np.abs(shift_values).astype(np.float32)  # (n_labels,)
    unary = np.broadcast_to(label_costs, (H * W, n_labels)).copy()
    # Background pixels: zero cost across all labels.
    unary[~mask.flatten(), :] = 0.0

    # Pairwise label cost: |label_i - label_j| (truncated linear).
    # Shape: (n_labels, n_labels).
    pair_label = np.abs(shift_values[:, None] - shift_values[None, :]).astype(np.float32)

    # Edge weights (per-edge). gco's cut_grid_graph takes (V, H) edge weights:
    #   V: (H-1, W) vertical-edge weights
    #   H: (H, W-1) horizontal-edge weights
    if edge_aware:
        if edge_data is None:
            raise ValueError("edge_aware=True requires edge_data")
        gy = np.abs(np.diff(edge_data, axis=0))  # (H-1, W)
        gx = np.abs(np.diff(edge_data, axis=1))  # (H, W-1)
        v_edge = lam * np.exp(-(gy ** 2) / (2.0 * edge_sigma ** 2))
        h_edge = lam * np.exp(-(gx ** 2) / (2.0 * edge_sigma ** 2))
    else:
        v_edge = np.full((H - 1, W), lam, dtype=np.float32)
        h_edge = np.full((H, W - 1), lam, dtype=np.float32)

    # Solve via gco's grid graph.
    labels_flat = gco.cut_grid_graph(
        unary.reshape(H, W, n_labels).astype(np.float32),
        pair_label,
        v_edge.astype(np.float32),
        h_edge.astype(np.float32),
        n_iter=n_iter,
        algorithm="expansion",
    )
    labels = labels_flat.reshape(H, W)
    # gco returns indices into the label set, convert to actual shift values.
    shift_field = shift_values[labels]
    phi_refined = phi_init + 2 * np.pi * shift_field
    return shift_field, phi_refined


def evaluate_sample(model, cal_layer, sample, device,
                    num_shifts, alpha, lam,
                    edge_aware=False, edge_sigma=0.5):
    """Run PhiCalNet on one test sample, then refine via graph cut.
    Returns dict with baseline and refined depth metrics."""
    fringe = sample["fringe"].unsqueeze(0).to(device)
    k_gt = sample["k_gt"].numpy()
    mask = sample["mask_gt"].squeeze().bool().numpy()
    depth_gt = sample["depth_gt"].squeeze().numpy()

    with torch.no_grad():
        out = model(fringe)
    sin_hat = out["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
    cos_hat = out["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)

    # Baseline unwrap (PhiCalNet's median-filter trick, plus oracle k).
    phi_init = full_unwrap(sin_hat, cos_hat, k_gt)

    # Edge data for edge-aware smoothness: wrapped-phase gradient magnitude.
    edge_data = None
    if edge_aware:
        wph = np.arctan2(sin_hat, cos_hat)
        # Use sin/cos directly to avoid 2*pi wrap discontinuities in the
        # wrapped-phase gradient; |d(sin)| + |d(cos)| is smooth across wraps.
        edge_data = np.abs(sin_hat) + np.abs(cos_hat)

    # Refine via graph cut.
    shift_field, phi_refined = refine_phase(
        phi_init, mask, num_shifts, alpha, lam,
        edge_aware=edge_aware, edge_data=edge_data, edge_sigma=edge_sigma,
    )

    # Recompute depth via the differentiable calibration layer.
    def phi_to_depth(phi_np):
        t = torch.from_numpy(phi_np).unsqueeze(0).unsqueeze(0).float().to(device)
        return cal_layer(t).squeeze().cpu().numpy()

    depth_init = phi_to_depth(phi_init)
    depth_refined = phi_to_depth(phi_refined)

    err_init = np.abs(depth_init - depth_gt)
    err_refined = np.abs(depth_refined - depth_gt)

    n_flipped = int((shift_field[mask] != 0).sum())
    return {
        "name": sample["name"],
        "n_obj_pixels": int(mask.sum()),
        "n_flipped": n_flipped,
        "frac_flipped": n_flipped / max(int(mask.sum()), 1),
        "init_mae": float(err_init[mask].mean()),
        "init_rmse": float(np.sqrt((err_init[mask] ** 2).mean())),
        "refined_mae": float(err_refined[mask].mean()),
        "refined_rmse": float(np.sqrt((err_refined[mask] ** 2).mean())),
        "init_max": float(err_init[mask].max()),
        "refined_max": float(err_refined[mask].max()),
    }


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"alpha={args.alpha}, lam={args.lam}, num_shifts={args.num_shifts}, "
          f"edge_aware={args.edge_aware}")

    model = PhaseUNetFPP(variant="A1", in_channels=1, K_max=K_MAX,
                         dropout_rate=0.0)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test",
                           K_max=K_MAX)

    if args.samples:
        idxs = [int(i) for i in args.samples.split(",")]
    else:
        idxs = list(range(len(test_ds)))

    rows = []
    t0 = time.time()
    for idx in tqdm(idxs, desc="refine"):
        s = test_ds[idx]
        r = evaluate_sample(model, cal_layer, s, device,
                            num_shifts=args.num_shifts,
                            alpha=args.alpha, lam=args.lam,
                            edge_aware=args.edge_aware,
                            edge_sigma=args.edge_sigma)
        rows.append(r)
        if args.verbose:
            print(f"  [{idx:3d}] {r['name']:>40s} "
                  f"MAE {r['init_mae']:6.2f} -> {r['refined_mae']:6.2f}  "
                  f"RMSE {r['init_rmse']:7.2f} -> {r['refined_rmse']:7.2f}  "
                  f"max {r['init_max']:7.1f} -> {r['refined_max']:7.1f}  "
                  f"flipped {100 * r['frac_flipped']:5.2f}%")
    dt = time.time() - t0

    # Aggregate (per-sample averaged, paper convention).
    init_mae = np.mean([r["init_mae"] for r in rows])
    init_rmse = np.mean([r["init_rmse"] for r in rows])
    refined_mae = np.mean([r["refined_mae"] for r in rows])
    refined_rmse = np.mean([r["refined_rmse"] for r in rows])

    # Pooled (over all object pixels).
    pool_init_mae = np.average([r["init_mae"] for r in rows],
                               weights=[r["n_obj_pixels"] for r in rows])
    pool_init_rmse = np.sqrt(np.average(
        [r["init_rmse"] ** 2 for r in rows],
        weights=[r["n_obj_pixels"] for r in rows]))
    pool_refined_mae = np.average([r["refined_mae"] for r in rows],
                                  weights=[r["n_obj_pixels"] for r in rows])
    pool_refined_rmse = np.sqrt(np.average(
        [r["refined_rmse"] ** 2 for r in rows],
        weights=[r["n_obj_pixels"] for r in rows]))

    print("\n" + "=" * 80)
    print(f"Graph-cut refinement on {len(rows)} test samples ({dt:.1f}s total)")
    print(f"  alpha={args.alpha}  lam={args.lam}  num_shifts={args.num_shifts}  "
          f"edge_aware={args.edge_aware}")
    print("-" * 80)
    print(f"{'metric':<26s} {'before':>12s} {'after':>12s} {'delta':>12s}")
    print(f"{'Object MAE  (per-sample)':<26s} {init_mae:>12.3f} "
          f"{refined_mae:>12.3f} {refined_mae - init_mae:>+12.3f}")
    print(f"{'Object RMSE (per-sample)':<26s} {init_rmse:>12.3f} "
          f"{refined_rmse:>12.3f} {refined_rmse - init_rmse:>+12.3f}")
    print(f"{'Pixel  MAE  (pooled)':<26s} {pool_init_mae:>12.3f} "
          f"{pool_refined_mae:>12.3f} {pool_refined_mae - pool_init_mae:>+12.3f}")
    print(f"{'Pixel  RMSE (pooled)':<26s} {pool_init_rmse:>12.3f} "
          f"{pool_refined_rmse:>12.3f} {pool_refined_rmse - pool_init_rmse:>+12.3f}")
    n_flipped = sum(r["n_flipped"] for r in rows)
    n_total = sum(r["n_obj_pixels"] for r in rows)
    print(f"\nPixels flipped: {n_flipped:,} / {n_total:,} "
          f"({100 * n_flipped / max(n_total, 1):.3f}%)")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Unary cost per unit |shift|")
    p.add_argument("--lam", type=float, default=5.0,
                   help="Pairwise cost coefficient")
    p.add_argument("--num_shifts", type=int, default=2,
                   help="Allowed shifts {-num_shifts ... +num_shifts}")
    p.add_argument("--edge_aware", action="store_true",
                   help="Edge-aware pairwise weight via wrapped-phase magnitude gradient")
    p.add_argument("--edge_sigma", type=float, default=0.5,
                   help="Gaussian width for edge weight")
    p.add_argument("--samples", type=str, default=None,
                   help="Comma-separated test sample indices (default: all)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)
