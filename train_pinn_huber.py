"""
PINN UNet variant: Huber loss on unwrapped absolute phase.

Same as train_pinn_unet.py but replaces the L1 physics loss with a Huber
loss. The Huber loss caps penalties at a threshold delta, preventing
large phase errors (near object boundaries, fine details) from
dominating the gradient signal while keeping the bijective property
of unwrapped phase.

L_huber(x) = 0.5 * x^2          if |x| < delta
           = delta * (|x| - 0.5*delta)  otherwise

Checkpoints saved to checkpoints_pinn_huber_lambda{X}_delta{Y}/ so they
don't collide with the main PINN runs.
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "UNet"))

from UNet.unet import UNetFPP, HybridL1Loss
from models.depth_to_phase_layer import DepthToPhaseLayer
from train_pinn_unet import PINNDataset

# =============================================================================
# Paths
# =============================================================================
import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
PARAMS_DIR = BASE / "training_datasets/info_depth_params"
CAL_DIR = BASE / "calibration_matrices"


# =============================================================================
# Huber physics loss
# =============================================================================
class PhysicsHuberLoss(nn.Module):
    """
    Huber loss on unwrapped phase consistency.
    Same physics principle as L1 version but caps large errors at delta.
    """
    def __init__(self, depth_to_phase_layer, delta=3.14159):
        super().__init__()
        self.d2p = depth_to_phase_layer
        self.delta = delta

    def forward(self, depth_pred_norm, abs_phase_gt, mask, d_min, d_max):
        # Denormalize to mm
        d_range = (d_max - d_min).view(-1, 1, 1, 1)
        d_min_v = d_min.view(-1, 1, 1, 1)
        depth_mm = depth_pred_norm * d_range + d_min_v

        # Depth → absolute phase
        abs_phase_pred = self.d2p(depth_mm)

        # Masked Huber loss
        diff = abs_phase_pred - abs_phase_gt
        abs_diff = torch.abs(diff)

        # Huber: 0.5*x^2 if |x|<delta, else delta*(|x|-0.5*delta)
        quadratic = 0.5 * diff * diff
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        huber = torch.where(abs_diff < self.delta, quadratic, linear)

        n = mask.sum().clamp(min=1)
        return (huber * mask).sum() / n


# =============================================================================
# Training loop (same structure as train_pinn_unet.py)
# =============================================================================
def train_epoch(model, loader, optimizer, depth_loss_fn, physics_loss_fn,
                lambda_physics, device, iteration=0):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        fringe = batch['fringe'].to(device)
        depth_gt = batch['depth_norm'].to(device)
        abs_phase = batch['abs_phase_gt'].to(device)
        mask = batch['mask'].to(device)
        d_min = torch.tensor([b for b in batch['d_min']], device=device, dtype=torch.float32)
        d_max = torch.tensor([b for b in batch['d_max']], device=device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(fringe)

        L_depth = depth_loss_fn(pred, depth_gt)
        L_physics = physics_loss_fn(pred, abs_phase, mask, d_min, d_max)

        loss = L_depth + lambda_physics * L_physics

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss.item()
        iteration += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "depth": f"{L_depth.item():.4f}",
                          "phys": f"{L_physics.item():.4f}"})

    return running_loss / max(len(loader), 1), iteration


def validate(model, loader, depth_loss_fn, physics_loss_fn,
             lambda_physics, device):
    model.eval()
    running_combined = 0.0
    running_depth = 0.0
    running_physics = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            fringe = batch['fringe'].to(device)
            depth_gt = batch['depth_norm'].to(device)
            abs_phase = batch['abs_phase_gt'].to(device)
            mask = batch['mask'].to(device)
            d_min = torch.tensor([b for b in batch['d_min']], device=device, dtype=torch.float32)
            d_max = torch.tensor([b for b in batch['d_max']], device=device, dtype=torch.float32)

            pred = model(fringe)
            L_depth = depth_loss_fn(pred, depth_gt)
            L_physics = physics_loss_fn(pred, abs_phase, mask, d_min, d_max)
            loss = L_depth + lambda_physics * L_physics
            running_combined += loss.item()
            running_depth += L_depth.item()
            running_physics += L_physics.item()

    n = max(len(loader), 1)
    return running_combined / n, running_depth / n, running_physics / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    lambda_physics = args.lambda_physics
    huber_delta = args.huber_delta
    tag = f"pinn_huber_lambda{lambda_physics}_delta{huber_delta}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("PINN UNet Training (Huber) - Physics-Constrained FPP")
    print("=" * 70)
    print(f"Lambda physics: {lambda_physics}")
    print(f"Huber delta: {huber_delta}")
    print(f"Loss: HybridL1(alpha=0.7) + {lambda_physics} * Huber(delta={huber_delta}) on abs phase")
    print(f"Device: {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print("=" * 70)

    # Dataset (reuses PINNDataset from train_pinn_unet)
    train_ds = PINNDataset(
        DATA_ROOT / "train/fringe", DATA_ROOT / "train/depth",
        PHASE_ROOT / "train", PARAMS_DIR / "train/depth"
    )
    val_ds = PINNDataset(
        DATA_ROOT / "val/fringe", DATA_ROOT / "val/depth",
        PHASE_ROOT / "val", PARAMS_DIR / "val/depth"
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    # Model
    model = UNetFPP(in_channels=1, out_channels=1, dropout_rate=0.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss functions
    depth_loss_fn = HybridL1Loss(alpha=0.7)
    d2p = DepthToPhaseLayer(str(CAL_DIR)).to(device)
    physics_loss_fn = PhysicsHuberLoss(d2p, delta=huber_delta)

    # Optimizer (same as baseline)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6
    )

    # Logging
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss',
                                    'time_seconds', 'learning_rate', 'iteration'])

    start_epoch = 0
    best_val_loss = float('inf')
    iteration = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['loss']
        iteration = ckpt.get('iteration', 0)

    # Warmup
    warmup_epochs = 100
    print(f"Physics loss warmup: 0 -> {lambda_physics} over {warmup_epochs} epochs\n")

    print("Starting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        if epoch < warmup_epochs:
            current_lambda = lambda_physics * (epoch / warmup_epochs)
        else:
            current_lambda = lambda_physics

        train_loss, iteration = train_epoch(
            model, train_loader, optimizer, depth_loss_fn, physics_loss_fn,
            current_lambda, device, iteration
        )
        val_combined, val_depth, val_physics = validate(
            model, val_loader, depth_loss_fn, physics_loss_fn,
            current_lambda, device
        )

        scheduler.step(val_depth)
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, float(train_loss),
                                    float(val_depth), int(round(dt)),
                                    float(lr), int(iteration)])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_loss:.4f} | Val_d: {val_depth:.4f} | "
              f"Val_p: {val_physics:.4f} | lambda: {current_lambda:.4f} | "
              f"LR: {lr:.2e} | Iter: {iteration}", end="")

        if val_depth < best_val_loss:
            best_val_loss = val_depth
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_depth,
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
                'loss': val_depth,
                'iteration': iteration,
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-6 * 1.5:
            print(f"\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best val_depth: {best_val_loss:.6f}")
    print(f"Best model saved: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_physics", type=float, default=0.01)
    parser.add_argument("--huber_delta", type=float, default=3.14159,
                        help="Huber threshold (radians). Pi ~ one fringe period half-width.")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
