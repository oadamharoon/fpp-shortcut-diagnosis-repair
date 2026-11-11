"""
Test-time refinement of PhiCalNet (A1) unwrapped phase via classical
least-squares phase unwrapping (Pritt & Shipman 1994 family), with
PhiCalNet's prediction used as the initialization and anchor.

Energy:

    E(Phi) = sum_{(p,q) in edges}  w_pq * ((Phi(q) - Phi(p)) - pv_pq)^2
           + lam * sum_p           (Phi(p) - Phi_init(p))^2

where:
    pv_pq = wrap_to_pi(phi_wrapped(q) - phi_wrapped(p))   [target diff]
    Phi_init(p) = PhiCalNet's full_unwrap output at p     [anchor]
    w_pq = edge weight (uniform by default; edge-aware is optional)

Setting the gradient to zero yields a sparse, symmetric, positive-definite
linear system in Phi:

    (L + lam*I) Phi = b + lam * Phi_init

where L is the (weighted) graph Laplacian on the 4-connected grid and
b is the divergence of pv. Solved with conjugate gradient via a matrix-
free LinearOperator (no need to materialize the H*W square sparse matrix).

Optional integer-snap step: round Phi_refined to the nearest 2*pi shift
of Phi_init at each pixel.

Usage:
    python -u refine_lsunwrap.py --lam 0.01 --device cpu
    python -u refine_lsunwrap.py --samples 0,5,12,20 --lam 0.001 --snap
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm

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


def wrap_to_pi(x):
    """Map x to its principal value in (-pi, pi]."""
    return np.angle(np.exp(1j * x))


def least_squares_unwrap(phi_wrapped, phi_init, mask, lam=0.01,
                         edge_weights_v=None, edge_weights_h=None,
                         tol=1e-6, max_iter=500):
    """Solve (L + lam*I) Phi = b + lam * Phi_init via CG.

    Args:
        phi_wrapped: (H, W) wrapped phase from PhiCalNet, in (-pi, pi].
        phi_init: (H, W) initial unwrapped phase from PhiCalNet (full_unwrap).
        mask: (H, W) bool object mask. Edges where either endpoint is
            outside the mask carry zero data weight (background phase is
            unreliable), so background pixels are pinned to phi_init by
            the regularizer.
        lam: regularization strength (stay close to phi_init).
        edge_weights_v: (H-1, W) optional vertical-edge weights, default ones.
        edge_weights_h: (H, W-1) optional horizontal-edge weights, default ones.
        tol, max_iter: CG convergence parameters.

    Returns:
        phi_refined: (H, W) refined unwrapped phase.
        info: int CG return flag (0 = success).
    """
    H, W = phi_wrapped.shape
    N = H * W

    # Principal-value differences along each edge direction.
    # pv_v[i, j] is the target diff between pixel (i+1, j) and (i, j).
    pv_v = wrap_to_pi(phi_wrapped[1:, :] - phi_wrapped[:-1, :])  # (H-1, W)
    pv_h = wrap_to_pi(phi_wrapped[:, 1:] - phi_wrapped[:, :-1])  # (H, W-1)

    if edge_weights_v is None:
        edge_weights_v = np.ones((H - 1, W), dtype=np.float64)
    if edge_weights_h is None:
        edge_weights_h = np.ones((H, W - 1), dtype=np.float64)

    # Zero out edge weights wherever either endpoint is outside the mask.
    # Background phase from PhiCalNet is unreliable, so we don't trust those
    # data terms; the lam regularizer keeps background pinned to phi_init.
    mask_v_edge = mask[:-1, :] & mask[1:, :]   # (H-1, W)
    mask_h_edge = mask[:, :-1] & mask[:, 1:]   # (H, W-1)
    edge_weights_v = edge_weights_v * mask_v_edge.astype(np.float64)
    edge_weights_h = edge_weights_h * mask_h_edge.astype(np.float64)

    # Per-pixel degree in the weighted Laplacian.
    deg = np.zeros((H, W), dtype=np.float64)
    deg[:-1, :] += edge_weights_v
    deg[1:, :]  += edge_weights_v
    deg[:, :-1] += edge_weights_h
    deg[:, 1:]  += edge_weights_h

    # Matrix-free apply of (L + lam*I).
    def matvec(x):
        x = x.reshape(H, W)
        out = (deg + lam) * x
        # Vertical edges (between (i, j) and (i+1, j)).
        out[:-1, :] -= edge_weights_v * x[1:, :]
        out[1:, :]  -= edge_weights_v * x[:-1, :]
        # Horizontal edges (between (i, j) and (i, j+1)).
        out[:, :-1] -= edge_weights_h * x[:, 1:]
        out[:, 1:]  -= edge_weights_h * x[:, :-1]
        return out.flatten()

    A_op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)

    # b(p) = lam * Phi_init(p)
    #      + sum over edges (p, q): -w_pq * pv_pq    (with sign per orientation)
    # Vertical edge: pv_v[i, j] is for "pixel (i+1, j) minus pixel (i, j)".
    # Contribution to gradient ∂E/∂Phi(p):
    #   For p = (i,   j): + w_pq * pv_v[i, j]
    #   For p = (i+1, j): - w_pq * pv_v[i, j]
    # Sign flips for horizontal.
    b = lam * phi_init.copy()
    b[:-1, :] -= edge_weights_v * pv_v
    b[1:, :]  += edge_weights_v * pv_v
    b[:, :-1] -= edge_weights_h * pv_h
    b[:, 1:]  += edge_weights_h * pv_h

    x0 = phi_init.flatten()
    sol, info = cg(A_op, b.flatten(), x0=x0, atol=tol, rtol=tol, maxiter=max_iter)
    return sol.reshape(H, W), info


def snap_to_integer_k(phi_refined, phi_init):
    """Round phi_refined to the nearest 2*pi shift of phi_init at each pixel.
    Returns (phi_snapped, k_shift) where k_shift is the integer offset."""
    delta = phi_refined - phi_init
    k_shift = np.round(delta / (2 * np.pi)).astype(np.int32)
    return phi_init + 2 * np.pi * k_shift, k_shift


def evaluate_sample(model, cal_layer, sample, device, lam, snap):
    fringe = sample["fringe"].unsqueeze(0).to(device)
    k_gt = sample["k_gt"].numpy()
    mask = sample["mask_gt"].squeeze().bool().numpy()
    depth_gt = sample["depth_gt"].squeeze().numpy()

    with torch.no_grad():
        out = model(fringe)
    sin_hat = out["sin_phi_hat"].squeeze().cpu().numpy().astype(np.float64)
    cos_hat = out["cos_phi_hat"].squeeze().cpu().numpy().astype(np.float64)

    # PhiCalNet baseline: full_unwrap with oracle k.
    phi_init = full_unwrap(sin_hat, cos_hat, k_gt)
    phi_wrapped = np.arctan2(sin_hat, cos_hat).astype(np.float64)

    # LS unwrap (mask-aware: edges crossing mask boundary carry no data weight).
    phi_refined, info = least_squares_unwrap(phi_wrapped, phi_init,
                                             mask=mask, lam=lam)

    if snap:
        phi_final, k_shift = snap_to_integer_k(phi_refined, phi_init)
    else:
        phi_final = phi_refined
        k_shift = None

    # Recompute depths.
    def to_depth(phi_np):
        t = torch.from_numpy(phi_np).unsqueeze(0).unsqueeze(0).float().to(device)
        return cal_layer(t).squeeze().cpu().numpy()

    depth_init = to_depth(phi_init)
    depth_refined = to_depth(phi_final)

    err_init = np.abs(depth_init - depth_gt)
    err_refined = np.abs(depth_refined - depth_gt)

    if k_shift is not None:
        n_flipped = int((k_shift[mask] != 0).sum())
    else:
        # Continuous solution: count pixels whose phase changed by > pi.
        n_flipped = int((np.abs(phi_final - phi_init)[mask] > np.pi).sum())

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
        "cg_info": int(info),
    }


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"lam={args.lam}, snap={args.snap}")

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
    for idx in tqdm(idxs, desc="ls-unwrap"):
        s = test_ds[idx]
        r = evaluate_sample(model, cal_layer, s, device,
                            lam=args.lam, snap=args.snap)
        rows.append(r)
        if args.verbose:
            print(f"  [{idx:3d}] {r['name']:>40s} "
                  f"MAE {r['init_mae']:6.2f} -> {r['refined_mae']:6.2f}  "
                  f"RMSE {r['init_rmse']:7.2f} -> {r['refined_rmse']:7.2f}  "
                  f"max {r['init_max']:7.1f} -> {r['refined_max']:7.1f}  "
                  f"flipped {100 * r['frac_flipped']:5.2f}%  cg={r['cg_info']}")
    dt = time.time() - t0

    init_mae = np.mean([r["init_mae"] for r in rows])
    init_rmse = np.mean([r["init_rmse"] for r in rows])
    refined_mae = np.mean([r["refined_mae"] for r in rows])
    refined_rmse = np.mean([r["refined_rmse"] for r in rows])

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
    print(f"Least-squares unwrap refinement on {len(rows)} test samples "
          f"({dt:.1f}s total)")
    print(f"  lam={args.lam}  snap={args.snap}")
    print("-" * 80)
    print(f"{'metric':<30s} {'before':>12s} {'after':>12s} {'delta':>12s}")
    print(f"{'Object MAE  (per-sample)':<30s} {init_mae:>12.3f} "
          f"{refined_mae:>12.3f} {refined_mae - init_mae:>+12.3f}")
    print(f"{'Object RMSE (per-sample)':<30s} {init_rmse:>12.3f} "
          f"{refined_rmse:>12.3f} {refined_rmse - init_rmse:>+12.3f}")
    print(f"{'Pixel  MAE  (pooled)':<30s} {pool_init_mae:>12.3f} "
          f"{pool_refined_mae:>12.3f} {pool_refined_mae - pool_init_mae:>+12.3f}")
    print(f"{'Pixel  RMSE (pooled)':<30s} {pool_init_rmse:>12.3f} "
          f"{pool_refined_rmse:>12.3f} {pool_refined_rmse - pool_init_rmse:>+12.3f}")
    n_flipped = sum(r["n_flipped"] for r in rows)
    n_total = sum(r["n_obj_pixels"] for r in rows)
    print(f"\nPixels flipped (>pi shift): {n_flipped:,} / {n_total:,} "
          f"({100 * n_flipped / max(n_total, 1):.3f}%)")
    print("=" * 80)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=0.01,
                   help="Regularization strength (stay close to PhiCalNet)")
    p.add_argument("--snap", action="store_true",
                   help="Round refined phase to nearest 2*pi shift of phi_init")
    p.add_argument("--samples", type=str, default=None,
                   help="Comma-separated test sample indices (default: all)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    main(args)
