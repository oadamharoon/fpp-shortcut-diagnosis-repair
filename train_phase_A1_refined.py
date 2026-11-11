"""
Refined A1 training — extends the base A1 with three improvements aimed at
pushing object MAE below 1 mm:

  1. Phase-gradient loss: L1 on spatial derivatives of sin/cos predictions.
     Encourages smooth phase (errors are scattered, so smoothness priors
     should pull them toward correct values).

  2. Longer training: min_lr lowered from 1e-6 to 1e-7. The base A1 run
     showed continued slow val_phase improvement at 1e-6 before early stop.

  3. Fixed w_depth = 0.1 (no ramp). Previously used a ramp from 0.05 to
     0.5 which appeared to confound the phase-gradient-loss ablation.
     Fixed weight is now used to cleanly isolate the effect of (1) and (2).

Writes to checkpoints_phase_A1_refined/ so the base A1 run is untouched.
NOTE: previous ramp-based training results are overwritten by resubmission.
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
from train_phase_variant import (
    phase_loss, depth_loss_hybrid, compute_depth_from_phase,
)

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32


def phase_gradient_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask):
    """
    L1 on spatial gradients of sin/cos. Both sides use the same
    differencing so discontinuities in GT (at wrap boundaries) match those
    in prediction.

    sin and cos are continuous under phase wrapping (cos(2pi-eps) ~ cos(eps)),
    so their spatial gradients are well-defined even at wrap boundaries.
    """
    # Horizontal gradients
    dsin_x_hat = sin_hat[:, :, :, 1:] - sin_hat[:, :, :, :-1]
    dcos_x_hat = cos_hat[:, :, :, 1:] - cos_hat[:, :, :, :-1]
    dsin_x_gt = sin_gt[:, :, :, 1:] - sin_gt[:, :, :, :-1]
    dcos_x_gt = cos_gt[:, :, :, 1:] - cos_gt[:, :, :, :-1]
    # Mask: object pixels, and both sides of the difference must be object
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    # Vertical gradients
    dsin_y_hat = sin_hat[:, :, 1:, :] - sin_hat[:, :, :-1, :]
    dcos_y_hat = cos_hat[:, :, 1:, :] - cos_hat[:, :, :-1, :]
    dsin_y_gt = sin_gt[:, :, 1:, :] - sin_gt[:, :, :-1, :]
    dcos_y_gt = cos_gt[:, :, 1:, :] - cos_gt[:, :, :-1, :]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    nx = mask_x.sum().clamp(min=1)
    ny = mask_y.sum().clamp(min=1)

    L_x = ((torch.abs(dsin_x_hat - dsin_x_gt) + torch.abs(dcos_x_hat - dcos_x_gt)) * mask_x).sum() / (2 * nx)
    L_y = ((torch.abs(dsin_y_hat - dsin_y_gt) + torch.abs(dcos_y_hat - dcos_y_gt)) * mask_y).sum() / (2 * ny)

    return (L_x + L_y) / 2.0


def compute_depth_weight(epoch, ramp_start=0, ramp_end=100, w_start=0.05, w_end=0.5):
    """Linear ramp from w_start at epoch ramp_start to w_end at ramp_end."""
    if epoch <= ramp_start:
        return w_start
    if epoch >= ramp_end:
        return w_end
    t = (epoch - ramp_start) / (ramp_end - ramp_start)
    return w_start + t * (w_end - w_start)


def train_epoch(model, loader, optimizer, cal_layer, device,
                w_phase, w_grad, w_depth, iteration=0):
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

        # A1: GT k
        k_tensor = k_gt.unsqueeze(1)
        depth_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

        L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
        L_grad = phase_gradient_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
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


def validate(model, loader, cal_layer, device, w_phase, w_grad, w_depth):
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

            L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
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
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tag = "phase_A1_refined"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("Phase-Intermediate A1 REFINED Training (fixed w_depth ablation)")
    print("=" * 70)
    print("Enhancements vs base A1:")
    print("  (1) Phase-gradient loss (L1 on spatial grad of sin/cos)")
    print("  (2) Lower min_lr: 1e-7 (vs 1e-6 base)")
    print("  (3) Fixed w_depth = 0.1 (no ramp — isolates phase-gradient effect)")
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
    # Enhancement #2: lower min_lr to allow more LR tiers of training
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

    # Fixed weights
    w_phase = 1.0
    w_grad = 0.5   # Enhancement #1: gradient loss weight

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        # Enhancement #3: ramping depth weight
        w_depth = 0.1  # Fixed — no ramp (ablation to isolate phase-gradient loss)

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, iteration
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth
        )

        # Scheduler on val_phase (primary metric, as in base A1)
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

        # Lower min_lr tolerance
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
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
