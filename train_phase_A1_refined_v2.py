"""
Refined A1 v2 — targets wrap-boundary outliers specifically.

Builds on refined v1 (phase-gradient loss, lower min_lr, weight-scheduled
depth) with two additional targeted enhancements:

  IDEA 3: Circular phase loss on phi directly (not sin/cos).
    Computes phase error as arctan2(sin(phi_hat - phi_gt), cos(phi_hat - phi_gt)),
    which is the proper geodesic distance on the unit circle. This measures
    the quantity we actually care about (angular phase error) rather than
    the sin/cos representation, which decouples at wrap boundaries where
    small sin/cos errors flip atan2 by nearly 2*pi.

  IDEA 1: Boundary-weighted phase loss.
    Upweights pixels where |phi_gt| is near pi (wrap boundaries). These
    are exactly the pixels where tiny phase errors cascade through the
    unwrap round-correction into full-period depth errors.
    Weight = 1 + alpha * (|phi_gt| / pi)^2, alpha = 3 (so boundary pixels
    get 4x weight vs mid-fringe pixels).

Keeps the remaining refined-v1 enhancements:
  - Phase-gradient loss (on sin/cos, which are smooth everywhere)
  - Fixed w_depth = 0.1 (no ramp — isolates loss-formulation effects)
  - min_lr = 1e-7

Writes to checkpoints_phase_A1_refined_v2/ so refined-v1 is untouched.
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer import CalibrationLayer
from data.phase_dataset import PhaseDataset
from train_phase_variant import depth_loss_hybrid, compute_depth_from_phase
from train_phase_A1_refined import phase_gradient_loss, compute_depth_weight

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32


def circular_phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask,
                         boundary_alpha=3.0):
    """
    Wrap-aware circular loss on phi, with boundary weighting.

    Circular distance: for two phases a, b in (-pi, pi], the proper
    geodesic distance is arctan2(sin(a-b), cos(a-b)) which handles the
    wrap topology correctly. When a is near +pi and b is near -pi, raw
    |a-b| is ~2*pi but circular distance is ~0 (they're close on the
    circle). Conversely, sin/cos L1 over-penalizes these cases.

    Boundary weighting: pixels where |phi_gt| is near pi are exactly the
    ones that cascade to full-period depth errors. We upweight these.

    Args:
        sin_hat, cos_hat: (B, 1, H, W) normalized (sin^2 + cos^2 = 1)
        sin_gt, cos_gt: (B, 1, H, W) GT
        mask: (B, 1, H, W) object mask
        boundary_alpha: weight multiplier for boundary pixels (0 = uniform)

    Returns:
        Scalar loss (weighted mean of per-pixel circular distance).
    """
    # phi_hat = atan2(sin_hat, cos_hat); same for gt
    # We compute the difference phi_hat - phi_gt directly via trig identities
    # to avoid atan2 singularities and keep the full wrap geometry:
    #   sin(phi_hat - phi_gt) = sin_hat*cos_gt - cos_hat*sin_gt
    #   cos(phi_hat - phi_gt) = cos_hat*cos_gt + sin_hat*sin_gt
    sin_diff = sin_hat * cos_gt - cos_hat * sin_gt
    cos_diff = cos_hat * cos_gt + sin_hat * sin_gt

    # Circular distance magnitude in [0, pi]
    # atan2(|sin_diff|, cos_diff) gives the absolute angular distance
    circular_dist = torch.atan2(torch.abs(sin_diff), cos_diff)

    # Boundary weighting based on |phi_gt|
    # phi_gt = atan2(sin_gt, cos_gt), |phi_gt| / pi in [0, 1]
    phi_gt = torch.atan2(sin_gt, cos_gt)
    phi_gt_normalized = torch.abs(phi_gt) / torch.pi  # [0, 1]
    weight = 1.0 + boundary_alpha * phi_gt_normalized ** 2

    # Masked weighted mean
    w_mask = weight * mask
    num = (circular_dist * w_mask).sum()
    den = w_mask.sum().clamp(min=1)
    return num / den


def train_epoch(model, loader, optimizer, cal_layer, device,
                w_phase, w_grad, w_depth, boundary_alpha, iteration=0):
    model.train()
    running = {'total': 0.0, 'phase': 0.0, 'grad': 0.0, 'depth': 0.0}
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        fringe = batch['fringe'].to(device)
        sin_gt = batch['sin_phi_gt'].to(device)
        cos_gt = batch['cos_phi_gt'].to(device)
        k_gt = batch['k_gt'].to(device)
        mask = batch['mask_gt'].to(device)
        depth_gt = batch['depth_gt'].to(device)
        d_min = torch.tensor([b for b in batch['d_min']], device=device, dtype=torch.float32)
        d_max = torch.tensor([b for b in batch['d_max']], device=device, dtype=torch.float32)

        optimizer.zero_grad()
        out = model(fringe)
        sin_hat = out['sin_phi_hat']
        cos_hat = out['cos_phi_hat']

        k_tensor = k_gt.unsqueeze(1)
        depth_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

        # Primary loss: wrap-aware circular distance with boundary weighting
        L_phase = circular_phase_loss(
            sin_hat, cos_hat, sin_gt, cos_gt, mask, boundary_alpha
        )
        # Secondary: gradient loss on sin/cos (smoothness prior)
        L_grad = phase_gradient_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
        # Tertiary: depth consistency
        L_depth = depth_loss_hybrid(depth_mm, depth_gt, mask, d_min, d_max)

        loss = w_phase * L_phase + w_grad * L_grad + w_depth * L_depth

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running['total'] += loss.item()
        running['phase'] += L_phase.item()
        running['grad'] += L_grad.item()
        running['depth'] += L_depth.item()
        iteration += 1
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "ph": f"{L_phase.item():.4f}",
            "g": f"{L_grad.item():.4f}",
            "d": f"{L_depth.item():.4f}",
        })

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}, iteration


def validate(model, loader, cal_layer, device,
             w_phase, w_grad, w_depth, boundary_alpha):
    model.eval()
    running = {'total': 0.0, 'phase': 0.0, 'grad': 0.0, 'depth': 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            fringe = batch['fringe'].to(device)
            sin_gt = batch['sin_phi_gt'].to(device)
            cos_gt = batch['cos_phi_gt'].to(device)
            k_gt = batch['k_gt'].to(device)
            mask = batch['mask_gt'].to(device)
            depth_gt = batch['depth_gt'].to(device)
            d_min = torch.tensor([b for b in batch['d_min']], device=device, dtype=torch.float32)
            d_max = torch.tensor([b for b in batch['d_max']], device=device, dtype=torch.float32)

            out = model(fringe)
            sin_hat = out['sin_phi_hat']
            cos_hat = out['cos_phi_hat']
            k_tensor = k_gt.unsqueeze(1)
            depth_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

            L_phase = circular_phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask, boundary_alpha)
            L_grad = phase_gradient_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
            L_depth = depth_loss_hybrid(depth_mm, depth_gt, mask, d_min, d_max)
            total = w_phase * L_phase + w_grad * L_grad + w_depth * L_depth

            running['total'] += total.item()
            running['phase'] += L_phase.item()
            running['grad'] += L_grad.item()
            running['depth'] += L_depth.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tag = "phase_A1_refined_v2"
    if args.seed != 42:
        tag += f"_s{args.seed}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("Phase-Intermediate A1 REFINED v2 — wrap-boundary targeting")
    print("=" * 70)
    print("Enhancements vs refined v1:")
    print("  IDEA 3: Circular phase loss on phi (not sin/cos)")
    print(f"  IDEA 1: Boundary weighting alpha = {args.boundary_alpha}")
    print("Kept from refined v1:")
    print("  - Phase gradient loss on sin/cos")
    print("  - min_lr = 1e-7")
    print("  - Fixed w_depth = 0.1 (no ramp — isolates loss-formulation effects)")
    print("=" * 70)

    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train", K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val", K_max=K_MAX)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_total', 'train_phase', 'train_grad', 'train_depth',
                'val_total', 'val_phase', 'val_grad', 'val_depth',
                'time_seconds', 'learning_rate', 'iteration', 'w_depth',
            ])

    start_epoch = 0
    best_val_phase = float('inf')
    iteration = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_phase = ckpt['loss']
        iteration = ckpt.get('iteration', 0)

    w_phase = 1.0
    w_grad = 0.5
    boundary_alpha = args.boundary_alpha

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        w_depth = 0.1  # Fixed — no ramp (ablation to isolate loss-formulation effects)

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha, iteration
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha
        )

        scheduler.step(val_stats['phase'])
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_stats['total']), float(train_stats['phase']),
                float(train_stats['grad']), float(train_stats['depth']),
                float(val_stats['total']), float(val_stats['phase']),
                float(val_stats['grad']), float(val_stats['depth']),
                int(round(dt)), float(lr), int(iteration), float(w_depth),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_stats['total']:.4f} (ph={train_stats['phase']:.4f} "
              f"g={train_stats['grad']:.4f} d={train_stats['depth']:.4f}) | "
              f"Val: ph={val_stats['phase']:.4f} | w_d={w_depth:.3f} | "
              f"LR: {lr:.2e} | It: {iteration}", end="")

        if val_stats['phase'] < best_val_phase:
            best_val_phase = val_stats['phase']
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_stats['phase'],
                'iteration': iteration,
                'variant': 'A1',
            }, ckpt_dir / "best_model.pth")
            print(" <- NEW BEST!")
        else:
            print()

        if (epoch + 1) % 10 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_stats['phase'],
                'iteration': iteration,
                'variant': 'A1',
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-7 * 1.5:
            print(f"\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best val_phase: {best_val_phase:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary_alpha", type=float, default=3.0,
                        help="Boundary weight multiplier (weight = 1 + alpha*(|phi|/pi)^2)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="seed 42 reproduces the original run and keeps its tag; "
                             "other seeds get a _s<seed> suffix")
    args = parser.parse_args()
    main(args)
