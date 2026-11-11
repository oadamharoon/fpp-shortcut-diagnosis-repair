"""
A1 training with residual depth correction head.

Two changes from base A1:
  (1) Removes the weight-ramped depth loss. Keeps a constant low w_depth=0.05
      for consistency supervision only. Refined v1's log showed w_depth
      ramping up to 0.5 caused phase learning to plateau.
  (2) Adds a residual correction head that predicts per-pixel depth
      correction in mm, bounded to +/-50mm. The network learns phase as the
      primary signal; residual only activates where phase-based
      reconstruction has systematic errors.

Final depth supervision uses depth_from_phase + residual_mm.

Writes to checkpoints_phase_A1_residual/ so base A1 is untouched.
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet_residual import PhaseUNetResidual
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


def train_epoch(model, loader, optimizer, cal_layer, device,
                w_phase, w_depth, w_residual_reg, iteration=0):
    model.train()
    running = {'total': 0.0, 'phase': 0.0, 'depth': 0.0, 'residual_mag': 0.0}
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
        residual_mm = out['residual_mm']

        k_tensor = k_gt.unsqueeze(1)
        depth_phase_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

        # Final depth: phase-based + correction
        depth_final_mm = depth_phase_mm + residual_mm

        L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
        L_depth = depth_loss_hybrid(depth_final_mm, depth_gt, mask, d_min, d_max)

        # L2 regularization on residual: keep it small by default, only activate where needed
        # Only penalize residual magnitude on object pixels
        n = mask.sum().clamp(min=1)
        L_residual_reg = ((residual_mm ** 2) * mask).sum() / n

        loss = w_phase * L_phase + w_depth * L_depth + w_residual_reg * L_residual_reg

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running['total'] += loss.item()
        running['phase'] += L_phase.item()
        running['depth'] += L_depth.item()
        running['residual_mag'] += (torch.abs(residual_mm) * mask).sum().item() / n.item()
        iteration += 1
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "ph": f"{L_phase.item():.4f}",
            "d": f"{L_depth.item():.4f}",
            "|r|": f"{running['residual_mag']/(iteration - 0.5):.2f}mm",
        })

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}, iteration


def validate(model, loader, cal_layer, device, w_phase, w_depth, w_residual_reg):
    model.eval()
    running = {'total': 0.0, 'phase': 0.0, 'depth': 0.0, 'residual_mag': 0.0}
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
            residual_mm = out['residual_mm']
            k_tensor = k_gt.unsqueeze(1)
            depth_phase_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)
            depth_final_mm = depth_phase_mm + residual_mm

            L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
            L_depth = depth_loss_hybrid(depth_final_mm, depth_gt, mask, d_min, d_max)
            n = mask.sum().clamp(min=1)
            L_residual_reg = ((residual_mm ** 2) * mask).sum() / n
            total = w_phase * L_phase + w_depth * L_depth + w_residual_reg * L_residual_reg
            running['total'] += total.item()
            running['phase'] += L_phase.item()
            running['depth'] += L_depth.item()
            running['residual_mag'] += (torch.abs(residual_mm) * mask).sum().item() / n.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tag = "phase_A1_residual"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("Phase A1 with Residual Depth Correction Head")
    print("=" * 70)
    print("Changes from base A1:")
    print("  - Removed weight-ramped depth loss (kept constant w_depth=0.1)")
    print("  - Added residual correction head (1 ch, tanh-bounded to +/-50mm)")
    print("  - Final depth = depth_from_phase + residual_mm")
    print("  - L2 regularization on residual (w=0.001) to keep it small by default")
    print("=" * 70)

    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train", K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val", K_max=K_MAX)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    model = PhaseUNetResidual(in_channels=1, dropout_rate=0.0, residual_scale=50.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_total', 'train_phase', 'train_depth', 'train_residual_mag',
                'val_total', 'val_phase', 'val_depth', 'val_residual_mag',
                'time_seconds', 'learning_rate', 'iteration',
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

    # Fixed weights throughout training
    w_phase = 1.0
    w_depth = 0.1   # Constant — no ramping
    w_residual_reg = 0.001  # Small L2 reg to keep residual off when not needed

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_depth, w_residual_reg, iteration
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_depth, w_residual_reg
        )

        scheduler.step(val_stats['phase'])
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_stats['total']), float(train_stats['phase']),
                float(train_stats['depth']), float(train_stats['residual_mag']),
                float(val_stats['total']), float(val_stats['phase']),
                float(val_stats['depth']), float(val_stats['residual_mag']),
                int(round(dt)), float(lr), int(iteration),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: ph={train_stats['phase']:.4f} d={train_stats['depth']:.4f} "
              f"|r|={train_stats['residual_mag']:.2f}mm | "
              f"Val: ph={val_stats['phase']:.4f} d={val_stats['depth']:.4f} "
              f"|r|={val_stats['residual_mag']:.2f}mm | "
              f"LR: {lr:.2e}", end="")

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
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
