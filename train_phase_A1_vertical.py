"""
Single-frame u_p (vertical-fringe) phase backbone.

Mirror of train_phase_A1_refined_v2.py for the second fringe direction the
raw captures already contain. Identical recipe (losses, optimizer, schedule,
seed); what changes:
  (1) Input: one vertical fringe frame per scene (frame F25, the direction's
      first frame), via a symlinked single-frame dataset dir.
  (2) Targets: phase_targets_vertical (u_p-encoding wrapped phase + gray-code
      fringe order from the vertical gray codes).
  (3) Depth loss runs through CalibrationLayerVertical (u_p triangulation,
      T_p = 18.0291 px, offset 2.5461 rad), and the unwrap shift in
      compute_depth_from_phase uses the vertical period.
  (4) K_MAX = 64 (vertical fringe order reaches ~51 with the 18 px period).

Checkpoints -> checkpoints_phase_A1_vertical/
"""

import sys
import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer_vertical import CalibrationLayerVertical, T_P_VERTICAL
from data.phase_dataset import PhaseDataset
from train_phase_variant import depth_loss_hybrid, compute_depth_from_phase
from train_phase_A1_refined import phase_gradient_loss
from train_phase_A1_refined_v2 import circular_phase_loss

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_vertical_frame0"
PHASE_ROOT = BASE / "training_datasets/phase_targets_vertical"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 64


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
        depth_mm, _, _ = compute_depth_from_phase(
            sin_hat, cos_hat, k_tensor, cal_layer, T_p=T_P_VERTICAL)

        L_phase = circular_phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask, boundary_alpha)
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
        pbar.set_postfix({"L": f"{loss.item():.4f}", "ph": f"{L_phase.item():.4f}"})

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
            depth_mm, _, _ = compute_depth_from_phase(
                sin_hat, cos_hat, k_tensor, cal_layer, T_p=T_P_VERTICAL)

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
    if device != "cuda":
        sys.exit("FATAL: CUDA unavailable on this node; refusing to train on CPU.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tag = "phase_A1_vertical" + (f"_s{args.seed}" if args.seed != 42 else "")
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("Single-frame u_p (vertical-fringe) backbone, refined-v2 recipe")
    print(f"Data: {DATA_ROOT}")
    print(f"Targets: {PHASE_ROOT}")
    print(f"T_p = {T_P_VERTICAL}, K_MAX = {K_MAX}")
    print("=" * 70)

    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train", K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val", K_max=K_MAX)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    cal_layer = CalibrationLayerVertical(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_total', 'train_phase', 'train_grad', 'train_depth',
                'val_total', 'val_phase', 'val_grad', 'val_depth',
                'time_seconds', 'learning_rate', 'iteration', 'w_depth'])

    best_val_phase = float('inf')
    iteration = 0
    w_phase, w_grad, w_depth = 1.0, 0.5, 0.1
    boundary_alpha = 3.0

    print("\nStarting training...\n")
    for epoch in range(0, 1000):
        t0 = time.time()
        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha, iteration)
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha)

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
                int(round(dt)), float(lr), int(iteration), float(w_depth)])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_stats['total']:.4f} (ph={train_stats['phase']:.4f}) | "
              f"Val: ph={val_stats['phase']:.4f} | LR: {lr:.2e}", end="")

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
                'direction': 'vertical',
                'T_p': T_P_VERTICAL,
                'K_max': K_MAX,
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
                'direction': 'vertical',
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-7 * 1.5:
            print("\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print(f"Best val_phase: {best_val_phase:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
