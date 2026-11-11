"""
Training script for Phase-Intermediate UNet (variants A1 and A2).

Key design: the network outputs phase, not depth. Depth is computed
analytically via the fixed calibration layer. This forces the network
to emit phase-shaped predictions rather than depth-shaped ones, testing
whether the shape-prior shortcut identified in the interpretability
analysis was driving the ~14.5 mm floor.

Variant A1:
    Phase output head only. Uses GT fringe order (k) at training time.
    Tests: "can the network learn wrapped phase from one fringe?"
    Isolates phase-learning from fringe-order ambiguity.

Variant A2:
    Phase head + order head. Full single-shot pipeline.
    Gumbel-softmax during training (temp 1.0 → 0.1 over 50 epochs),
    hard argmax at inference.

Loss: L_phase (primary) + 0.1*L_depth (consistency) + 0.1*L_k (A2 only)
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


# =============================================================================
# Paths
# =============================================================================
import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32


# =============================================================================
# Loss functions
# =============================================================================

def phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask):
    """L1 loss on sin/cos components, averaged over object pixels."""
    n = mask.sum().clamp(min=1)
    diff_sin = (torch.abs(sin_hat - sin_gt) * mask).sum() / n
    diff_cos = (torch.abs(cos_hat - cos_gt) * mask).sum() / n
    return (diff_sin + diff_cos) / 2.0


def depth_loss_hybrid(depth_hat_mm, depth_gt_mm, mask, d_min, d_max, alpha=0.7, eps=1e-8):
    """HybridL1 on per-sample normalized depth."""
    d_range = (d_max - d_min).view(-1, 1, 1, 1).clamp(min=eps)
    d_min_v = d_min.view(-1, 1, 1, 1)
    pred_norm = (depth_hat_mm - d_min_v) / d_range
    gt_norm = (depth_gt_mm - d_min_v) / d_range

    n = mask.sum().clamp(min=1)
    masked_l1 = (torch.abs(pred_norm - gt_norm) * mask).sum() / n
    global_l1 = torch.abs(pred_norm - gt_norm).mean()
    return alpha * masked_l1 + (1 - alpha) * global_l1


def fringe_order_loss(order_logits, k_gt, mask):
    """Cross-entropy on fringe order, averaged over object pixels only."""
    B, K, H, W = order_logits.shape
    mask_2d = mask.squeeze(1)  # (B, H, W)
    logits_flat = order_logits.permute(0, 2, 3, 1).reshape(-1, K)
    target_flat = k_gt.reshape(-1)
    mask_flat = mask_2d.reshape(-1)

    ce = F.cross_entropy(logits_flat, target_flat, reduction='none')
    n = mask_flat.sum().clamp(min=1)
    return (ce * mask_flat).sum() / n


# =============================================================================
# Gumbel-softmax for fringe order
# =============================================================================

def soft_argmax_gumbel(logits, temperature, training=True):
    """
    Straight-through Gumbel-softmax for fringe order prediction.

    Training: sample with Gumbel noise, soft probabilities for gradient.
    Inference: hard argmax.

    Args:
        logits: (B, K, H, W)
        temperature: Gumbel temperature (anneal 1.0 → 0.1)
        training: if False, use hard argmax

    Returns:
        k_soft: (B, 1, H, W) float, fringe order value
    """
    B, K, H, W = logits.shape
    if training:
        # Gumbel-softmax, hard=True gives one-hot with straight-through gradient
        probs = F.gumbel_softmax(
            logits.permute(0, 2, 3, 1).reshape(-1, K),
            tau=temperature, hard=True
        )  # (B*H*W, K)
        indices = torch.arange(K, device=logits.device, dtype=torch.float32)
        k_soft = (probs * indices).sum(dim=-1)  # (B*H*W,)
        return k_soft.reshape(B, 1, H, W)
    else:
        # Hard argmax at inference
        return logits.argmax(dim=1, keepdim=True).float()


# =============================================================================
# Training and validation
# =============================================================================

def compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer, T_p=36.0):
    """
    Build absolute phase Phi and run calibration.

    Applies the MATLAB -pi/T shift+unshift to handle fringe wrap boundaries
    (without this, pixels near wrap boundaries get placed in the wrong fringe
    period and produce ~T_p depth errors).

    Phi = atan2(sin(wph - pi/T), cos(wph - pi/T)) + 2*pi*k + pi/T
    """
    shift = -torch.pi / T_p
    # Compute wrapped phase then apply shift-unshift via atan2
    # (matches MATLAB grayUnwrap: wph_shifted = atan2(sin(wph+shift), cos(wph+shift)))
    cos_s = torch.cos(torch.tensor(shift, dtype=sin_hat.dtype, device=sin_hat.device))
    sin_s = torch.sin(torch.tensor(shift, dtype=sin_hat.dtype, device=sin_hat.device))
    # sin(wph + shift) = sin(wph)cos(shift) + cos(wph)sin(shift)
    # cos(wph + shift) = cos(wph)cos(shift) - sin(wph)sin(shift)
    shifted_sin = sin_hat * cos_s + cos_hat * sin_s
    shifted_cos = cos_hat * cos_s - sin_hat * sin_s
    wph_shifted = torch.atan2(shifted_sin, shifted_cos)
    Phi = wph_shifted + 2.0 * torch.pi * k_tensor.float() - shift  # (B, 1, H, W)
    depth_mm = cal_layer(Phi)  # (B, 1, H, W) in mm
    phi_wrap = torch.atan2(sin_hat, cos_hat)  # for diagnostics
    return depth_mm, phi_wrap, Phi


def train_epoch(model, loader, optimizer, cal_layer, variant, temperature, device, iteration=0):
    model.train()
    running_total = 0.0
    running_phase = 0.0
    running_depth = 0.0
    running_k = 0.0
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

        # Fringe order: GT for A1, predicted (gumbel-softmax) for A2
        if variant == 'A1':
            k_tensor = k_gt.unsqueeze(1)  # (B, 1, H, W)
            L_k = torch.tensor(0.0, device=device)
        else:  # A2
            k_tensor = soft_argmax_gumbel(
                out['order_logits'], temperature, training=True
            )
            L_k = fringe_order_loss(out['order_logits'], k_gt, mask)

        # Build depth from phase via calibration
        depth_hat_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

        # Primary loss
        L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)

        # Secondary loss: depth consistency
        L_depth = depth_loss_hybrid(depth_hat_mm, depth_gt, mask, d_min, d_max)

        # Total
        if variant == 'A1':
            loss = L_phase + 0.1 * L_depth
        else:
            loss = L_phase + 0.1 * L_k + 0.1 * L_depth

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_total += loss.item()
        running_phase += L_phase.item()
        running_depth += L_depth.item()
        if variant == 'A2':
            running_k += L_k.item()
        iteration += 1

        postfix = {"L": f"{loss.item():.4f}",
                   "ph": f"{L_phase.item():.4f}",
                   "d": f"{L_depth.item():.4f}"}
        if variant == 'A2':
            postfix["k"] = f"{L_k.item():.4f}"
            postfix["T"] = f"{temperature:.2f}"
        pbar.set_postfix(postfix)

    n = max(len(loader), 1)
    return (running_total / n, running_phase / n, running_depth / n,
            running_k / n if variant == 'A2' else 0.0, iteration)


def validate(model, loader, cal_layer, variant, temperature, device):
    """Returns (combined, phase, depth, k) mean losses."""
    model.eval()
    running_total = 0.0
    running_phase = 0.0
    running_depth = 0.0
    running_k = 0.0
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

            if variant == 'A1':
                k_tensor = k_gt.unsqueeze(1)
                L_k = torch.tensor(0.0, device=device)
            else:
                # During validation, use hard argmax to match inference
                k_tensor = soft_argmax_gumbel(
                    out['order_logits'], temperature, training=False
                )
                L_k = fringe_order_loss(out['order_logits'], k_gt, mask)

            depth_hat_mm, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k_tensor, cal_layer)

            L_phase = phase_loss(sin_hat, cos_hat, sin_gt, cos_gt, mask)
            L_depth = depth_loss_hybrid(depth_hat_mm, depth_gt, mask, d_min, d_max)

            if variant == 'A1':
                total = L_phase + 0.1 * L_depth
            else:
                total = L_phase + 0.1 * L_k + 0.1 * L_depth

            running_total += total.item()
            running_phase += L_phase.item()
            running_depth += L_depth.item()
            if variant == 'A2':
                running_k += L_k.item()

    n = max(len(loader), 1)
    return (running_total / n, running_phase / n, running_depth / n,
            running_k / n if variant == 'A2' else 0.0)


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tag = f"phase_{args.variant}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print(f"Phase-Intermediate UNet Training — Variant {args.variant}")
    print("=" * 70)
    print(f"Variant:     {args.variant} ({'oracle k' if args.variant == 'A1' else 'learned k'})")
    print(f"Loss:        L_phase + 0.1*L_depth" + (" + 0.1*L_k" if args.variant == 'A2' else ""))
    print(f"Device:      {device}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs:        {log_dir}")
    print("=" * 70)

    # Dataset
    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train", K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val", K_max=K_MAX)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    # Model
    model = PhaseUNetFPP(
        variant=args.variant, in_channels=1, K_max=K_MAX, dropout_rate=0.0
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Calibration layer (fixed, no gradients)
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    # Optimizer + scheduler matching PINN config
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Logging
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'train_phase', 'train_depth', 'train_k',
                'val_loss', 'val_phase', 'val_depth', 'val_k',
                'time_seconds', 'learning_rate', 'iteration', 'temperature',
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

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        # Gumbel-softmax temperature anneal (A2 only); unused for A1
        if args.variant == 'A2':
            temperature = max(1.0 - (epoch / 50.0) * 0.9, 0.1)
        else:
            temperature = 1.0  # unused for A1

        train_total, train_phase, train_depth, train_k, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer,
            args.variant, temperature, device, iteration
        )
        val_total, val_phase, val_depth, val_k = validate(
            model, val_loader, cal_layer,
            args.variant, temperature, device
        )

        # Scheduler monitors L_phase (primary metric) only, not combined
        scheduler.step(val_phase)
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_total), float(train_phase), float(train_depth), float(train_k),
                float(val_total), float(val_phase), float(val_depth), float(val_k),
                int(round(dt)), float(lr), int(iteration), float(temperature),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_total:.4f} (ph={train_phase:.4f} d={train_depth:.4f}" +
              (f" k={train_k:.4f}" if args.variant == 'A2' else "") + ") | "
              f"Val: {val_total:.4f} (ph={val_phase:.4f})" + " | "
              f"T={temperature:.2f} | LR: {lr:.2e} | It: {iteration}", end="")

        if val_phase < best_val_phase:
            best_val_phase = val_phase
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_phase,
                'iteration': iteration,
                'variant': args.variant,
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
                'loss': val_phase,
                'iteration': iteration,
                'variant': args.variant,
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-6 * 1.5:
            print(f"\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best val_phase: {best_val_phase:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=['A1', 'A2'],
                        help="A1=oracle k, A2=learned k")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
