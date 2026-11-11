"""
PINN UNet training for FPP depth prediction.

Same UNet architecture as baseline, but with an additional physics
consistency loss: predicted depth is back-projected through the
calibration equations to compute the expected fringe phase, which
is compared against the GT wrapped phase from 18-step phase shifting.

L_total = L_depth + lambda_physics * L_physics

L_depth: HybridL1 on normalized depth (same as baseline)
L_physics: L1 on sin/cos of (depth-derived phase vs GT phase), masked
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "UNet"))

from UNet.unet import UNetFPP, HybridL1Loss
from models.depth_to_phase_layer import DepthToPhaseLayer

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
# Dataset: extends baseline with phase targets and depth params
# =============================================================================
class PINNDataset(Dataset):
    """Loads fringe + depth (normalized) + phase targets + d_min/d_max."""

    def __init__(self, fringe_dir, depth_dir, phase_dir, params_dir,
                 depth_key="depthMapNormalized"):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.phase_dir = Path(phase_dir)
        self.params_dir = Path(params_dir)
        self.depth_key = depth_key

        self.samples = []
        for fringe_path in sorted(self.fringe_dir.glob("*.png")):
            stem = fringe_path.stem
            depth_path = self.depth_dir / f"{stem}.mat"
            phase_path = self.phase_dir / f"{stem}.npz"
            params_path = self.params_dir / f"{stem}.mat"

            if depth_path.exists() and phase_path.exists() and params_path.exists():
                self.samples.append((fringe_path, depth_path, phase_path, params_path))

        if not self.samples:
            raise ValueError(f"No matching samples in {fringe_dir}")

        print(f"PINNDataset: {len(self.samples)} samples from {fringe_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fringe_path, depth_path, phase_path, params_path = self.samples[idx]

        # Fringe image
        fringe = np.array(Image.open(fringe_path).convert('L'), dtype=np.float32) / 255.0
        fringe = torch.from_numpy(fringe).unsqueeze(0)

        # Normalized depth
        mat = sio.loadmat(depth_path)
        depth_norm = mat[self.depth_key].astype(np.float32)
        depth_norm = torch.from_numpy(depth_norm).unsqueeze(0)

        # Phase targets (for physics loss)
        npz = np.load(phase_path)
        sin_phi = torch.from_numpy(npz['sin_phi'])  # (H, W)
        cos_phi = torch.from_numpy(npz['cos_phi'])  # (H, W)
        k = torch.from_numpy(npz['k'].astype(np.float32))  # (H, W)
        mask = torch.from_numpy(npz['mask']).unsqueeze(0)  # (1, H, W)
        # Absolute (unwrapped) phase for PINN loss — bijective in depth
        phi_wrapped = torch.atan2(sin_phi, cos_phi)  # (H, W)
        abs_phase_gt = (phi_wrapped + 2 * np.pi * k).unsqueeze(0)  # (1, H, W)

        # Depth params (for denormalization in physics loss)
        params = sio.loadmat(params_path)
        d_min = float(params['depth_min'].item())
        d_max = float(params['depth_max'].item())

        return {
            'fringe': fringe,
            'depth_norm': depth_norm,
            'abs_phase_gt': abs_phase_gt,
            'mask': mask,
            'd_min': d_min,
            'd_max': d_max,
            'name': fringe_path.stem,
        }


# =============================================================================
# Physics consistency loss
# =============================================================================
class PhysicsConsistencyLoss(nn.Module):
    """
    Phase consistency between depth-derived absolute phase and GT phase.
    Uses unwrapped (absolute) phase to avoid multi-modality at fringe wraps.
    L = L1(abs_phase_pred, abs_phase_gt) over object pixels.
    """
    def __init__(self, depth_to_phase_layer):
        super().__init__()
        self.d2p = depth_to_phase_layer

    def forward(self, depth_pred_norm, abs_phase_gt, mask, d_min, d_max):
        """
        Args:
            depth_pred_norm: (B, 1, H, W) predicted depth in [0, 1]
            abs_phase_gt: (B, 1, H, W) GT absolute phase (radians)
            mask: (B, 1, H, W) object mask
            d_min, d_max: (B,) per-sample depth range
        """
        # Denormalize to mm
        d_range = (d_max - d_min).view(-1, 1, 1, 1)
        d_min_v = d_min.view(-1, 1, 1, 1)
        depth_mm = depth_pred_norm * d_range + d_min_v

        # Depth → absolute phase (differentiable, bijective)
        abs_phase_pred = self.d2p(depth_mm)

        # Masked L1 on absolute phase
        n = mask.sum().clamp(min=1)
        loss = (torch.abs(abs_phase_pred - abs_phase_gt) * mask).sum() / n

        return loss


# =============================================================================
# Training
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
    """
    Returns (combined_loss, depth_only_loss, physics_only_loss).
    Use depth_only_loss for scheduler/checkpoint decisions to avoid
    confounding from lambda ramping.
    """
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
    tag = f"pinn_unet_lambda{lambda_physics}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("PINN UNet Training - Physics-Constrained FPP")
    print("=" * 70)
    print(f"Lambda physics: {lambda_physics}")
    print(f"Loss: HybridL1(alpha=0.7) + {lambda_physics} * PhaseConsistency")
    print(f"Device: {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print("=" * 70)

    # Dataset
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

    # Model (same UNet as baseline)
    model = UNetFPP(in_channels=1, out_channels=1, dropout_rate=0.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss functions
    depth_loss_fn = HybridL1Loss(alpha=0.7)
    d2p = DepthToPhaseLayer(str(CAL_DIR)).to(device)
    physics_loss_fn = PhysicsConsistencyLoss(d2p)

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

    # Warmup: ramp lambda from 0 to target over first N epochs
    warmup_epochs = 100
    print(f"Physics loss warmup: 0 -> {lambda_physics} over {warmup_epochs} epochs\n")

    print("Starting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        # Ramp lambda linearly during warmup
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

        # Use depth-only loss for scheduler and best model (lambda-invariant)
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
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_physics", type=float, default=0.1,
                        help="Weight for physics consistency loss")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
