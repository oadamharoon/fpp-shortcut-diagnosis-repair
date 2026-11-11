"""
Training script for FTP-as-Architecture variants.

Variants:
  D: parametric Gaussian filter (--filter parametric)
  E: flexible learned filter (--filter flexible)
  F: fine-tune from D with higher depth weight (--variant F)

Losses:
  L_phase = L1(sin(phi_hat), sin(phi_gt)) + L1(cos(phi_hat), cos(phi_gt))  [masked]
  L_k = CrossEntropy(k_logits, k_gt)  [object pixels]
  L_depth = HybridL1(depth_hat_norm, depth_gt_norm, alpha=0.7)

  D/E total: L = L_phase + 0.1*L_k + 0.1*L_depth
  F total:   L = L_phase + 0.1*L_k + 0.5*L_depth
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
from models.calibration_layer import CalibrationLayer
from models.ftp_architecture import FTPUNet
from data.ftp_dataset import FTPDataset


# =============================================================================
# Paths
# =============================================================================
import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"


# =============================================================================
# Losses
# =============================================================================

class PhaseLoss(nn.Module):
    """L1 loss on sin/cos of wrapped phase, masked to object pixels."""
    def forward(self, phi_hat, sin_gt, cos_gt, mask):
        sin_hat = torch.sin(phi_hat)
        cos_hat = torch.cos(phi_hat)
        n = mask.sum().clamp(min=1)
        loss_sin = (torch.abs(sin_hat - sin_gt) * mask).sum() / n
        loss_cos = (torch.abs(cos_hat - cos_gt) * mask).sum() / n
        return loss_sin + loss_cos


class FringeOrderLoss(nn.Module):
    """Cross-entropy loss on fringe order, averaged over object pixels."""
    def forward(self, k_logits, k_gt, mask):
        # k_logits: (B, K, H, W), k_gt: (B, H, W), mask: (B, 1, H, W)
        mask_2d = mask.squeeze(1)  # (B, H, W)
        B, K, H, W = k_logits.shape

        # Flatten for cross_entropy
        logits_flat = k_logits.permute(0, 2, 3, 1).reshape(-1, K)  # (B*H*W, K)
        target_flat = k_gt.reshape(-1)  # (B*H*W,)
        mask_flat = mask_2d.reshape(-1)  # (B*H*W,)

        ce = F.cross_entropy(logits_flat, target_flat, reduction='none')  # (B*H*W,)
        n = mask_flat.sum().clamp(min=1)
        return (ce * mask_flat).sum() / n


class DepthLoss(nn.Module):
    """HybridL1 on per-sample normalized depth, alpha=0.7."""
    def __init__(self, alpha=0.7, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, depth_hat, depth_gt, mask, d_min, d_max):
        """
        Args:
            depth_hat: (B, 1, H, W) predicted depth in mm
            depth_gt: (B, 1, H, W) GT depth in mm
            mask: (B, 1, H, W) object mask
            d_min: (B,) per-sample min depth
            d_max: (B,) per-sample max depth
        """
        # Normalize to [0, 1] per sample
        d_range = (d_max - d_min).view(-1, 1, 1, 1).clamp(min=self.eps)
        d_min_v = d_min.view(-1, 1, 1, 1)

        # Clamp predicted depth before normalization to prevent overflow
        depth_hat = depth_hat.clamp(d_min_v - d_range * 5, d_min_v + d_range * 6)

        pred_norm = (depth_hat - d_min_v) / d_range
        gt_norm = (depth_gt - d_min_v) / d_range

        # Masked L1
        diff_masked = torch.abs(pred_norm - gt_norm) * mask
        n_valid = mask.sum().clamp(min=1)
        masked_l1 = diff_masked.sum() / n_valid

        # Global L1
        global_l1 = torch.abs(pred_norm - gt_norm).mean()

        return self.alpha * masked_l1 + (1 - self.alpha) * global_l1


# =============================================================================
# Training loop
# =============================================================================

def train_epoch(model, loader, optimizer, phase_loss_fn, k_loss_fn,
                depth_loss_fn, device, w_phase, w_k, w_depth, iteration=0):
    model.train()
    running_loss = 0.0
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

        L_phase = phase_loss_fn(out['phi_hat'], sin_gt, cos_gt, mask)
        L_k = k_loss_fn(out['k_logits'], k_gt, mask)
        L_depth = depth_loss_fn(out['depth_hat'], depth_gt, mask, d_min, d_max)

        loss = w_phase * L_phase + w_k * L_k + w_depth * L_depth

        # Skip NaN batches to prevent poisoning weights
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            pbar.set_postfix({"loss": "NaN/Inf SKIPPED"})
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss.item()
        iteration += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "ph": f"{L_phase.item():.4f}",
                          "k": f"{L_k.item():.4f}",
                          "d": f"{L_depth.item():.4f}"})

    return running_loss / len(loader), iteration


def validate(model, loader, phase_loss_fn, k_loss_fn, depth_loss_fn,
             device, w_phase, w_k, w_depth):
    model.eval()
    running_loss = 0.0
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

            L_phase = phase_loss_fn(out['phi_hat'], sin_gt, cos_gt, mask)
            L_k = k_loss_fn(out['k_logits'], k_gt, mask)
            L_depth = depth_loss_fn(out['depth_hat'], depth_gt, mask, d_min, d_max)

            loss = w_phase * L_phase + w_k * L_k + w_depth * L_depth
            running_loss += loss.item()

    return running_loss / len(loader)


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Loss weights
    if args.variant == 'F':
        w_phase, w_k, w_depth = 1.0, 0.1, 0.5
    else:
        w_phase, w_k, w_depth = 1.0, 0.1, 0.1

    tag = f"ftp_{args.variant}_{args.filter}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print(f"FTP Architecture Training - Variant {args.variant}")
    print("=" * 70)
    print(f"Filter: {args.filter}")
    print(f"Weights: phase={w_phase}, k={w_k}, depth={w_depth}")
    print(f"Device: {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print("=" * 70)

    # Dataset
    train_ds = FTPDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train")
    val_ds = FTPDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    # Model
    cal_layer = CalibrationLayer(str(CAL_DIR))
    model = FTPUNet(
        calibration_layer=cal_layer,
        filter_type=args.filter,
        fu0=-21, fv0=0,
        K_max=32,
        gumbel_temperature=1.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load pretrained for variant F
    if args.variant == 'F' and args.resume is None:
        pretrained = Path(f"checkpoints_ftp_D_{args.filter}/best_model.pth")
        if pretrained.exists():
            print(f"Loading pretrained Variant D weights: {pretrained}")
            ckpt = torch.load(pretrained, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            print(f"WARNING: No pretrained D model found at {pretrained}")

    # Optimizer: separate param group for filter (no weight decay)
    filter_params = list(model.bp_filter.parameters())
    filter_param_ids = {id(p) for p in filter_params}
    other_params = [p for p in model.parameters() if id(p) not in filter_param_ids and p.requires_grad]

    optimizer = torch.optim.RMSprop([
        {'params': other_params, 'weight_decay': 1e-5},
        {'params': filter_params, 'weight_decay': 0.0},
    ], lr=1e-4, alpha=0.99)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=10, min_lr=1e-6)

    # Loss functions
    phase_loss_fn = PhaseLoss()
    k_loss_fn = FringeOrderLoss()
    depth_loss_fn = DepthLoss(alpha=0.7)

    # CSV logging
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'time_seconds',
                            'learning_rate', 'iteration'])

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    iteration = 0

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['loss']
        iteration = ckpt.get('iteration', 0)

    # Training loop
    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        epoch_start = time.time()

        train_loss, iteration = train_epoch(
            model, train_loader, optimizer,
            phase_loss_fn, k_loss_fn, depth_loss_fn,
            device, w_phase, w_k, w_depth, iteration
        )
        val_loss = validate(
            model, val_loader,
            phase_loss_fn, k_loss_fn, depth_loss_fn,
            device, w_phase, w_k, w_depth
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Log
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, float(train_loss), float(val_loss),
                            int(round(epoch_time)), float(current_lr), int(iteration)])

        print(f"Epoch [{epoch+1:3d}/1000] ({epoch_time:5.1f}s) | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | Iter: {iteration}", end="")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'iteration': iteration,
            }, ckpt_dir / "best_model.pth")
            print(" <- NEW BEST!")
        else:
            print()

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'iteration': iteration,
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        # Print filter state for parametric
        if args.filter == 'parametric' and (epoch + 1) % 10 == 0:
            f = model.bp_filter
            print(f"  Filter: fu0={f.fu0.item():.2f}, fv0={f.fv0.item():.2f}, "
                  f"sigma_u={f.log_sigma_u.exp().item():.2f}, "
                  f"sigma_v={f.log_sigma_v.exp().item():.2f}")

        # Early stopping
        if current_lr <= 1e-6 * 1.5:
            print(f"\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FTP Architecture")
    parser.add_argument("--variant", type=str, default="D",
                        choices=["D", "E", "F"],
                        help="Variant D=parametric, E=flexible, F=finetune")
    parser.add_argument("--filter", type=str, default="parametric",
                        choices=["parametric", "flexible"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
