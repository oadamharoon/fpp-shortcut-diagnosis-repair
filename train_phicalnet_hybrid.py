"""
Training script for Variant 2: PhiCalHybridNet (analytical phase + learned k).

Differences from train_phase_A1_multiframe.py:
  * Model is PhiCalHybridNet (wrapped phase computed analytically from the
    3 phase-shifted input frames; only the k-head is trainable).
  * No phase loss is needed -- phi_hat has no learnable parameters -- so
    training supervises k (cross-entropy) and depth (HybridL1) only.
  * The k-cross-entropy and depth losses are reused from train_ftp_variant.py
    to keep the loss behaviour identical to the existing learned-k pipeline.
  * Checkpoints / logs go to checkpoints_phicalnet_hybrid/ so neither the
    single-frame PhiCalNet nor the multi-frame learned-phase runs are
    touched.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset_multiframe import PhaseMultiFrameDataset
from models.calibration_layer import CalibrationLayer
from models.phicalnet_hybrid import PhiCalHybridNet
from train_ftp_variant import FringeOrderLoss, DepthLoss

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
MULTIFRAME_ROOT = (
    BASE / "training_datasets"
         / "training_data_depth_individual_normalized_multiframe_horizontal_only"
)
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32
FRAME_INDICES = (0, 6, 12)


def train_epoch(model, loader, optimizer, k_loss_fn, depth_loss_fn,
                device, w_k, w_depth, iteration=0):
    model.train()
    running = {"total": 0.0, "k": 0.0, "depth": 0.0}
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        fringe = batch["fringe"].to(device)
        k_gt = batch["k_gt"].to(device)
        mask = batch["mask_gt"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        d_min = torch.tensor([b for b in batch["d_min"]], device=device,
                             dtype=torch.float32)
        d_max = torch.tensor([b for b in batch["d_max"]], device=device,
                             dtype=torch.float32)

        optimizer.zero_grad()
        out = model(fringe)

        L_k = k_loss_fn(out["k_logits"], k_gt, mask)
        L_depth = depth_loss_fn(out["depth_hat"], depth_gt, mask, d_min, d_max)
        loss = w_k * L_k + w_depth * L_depth

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running["total"] += loss.item()
        running["k"] += L_k.item()
        running["depth"] += L_depth.item()
        iteration += 1
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "k": f"{L_k.item():.4f}",
            "d": f"{L_depth.item():.4f}",
        })

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}, iteration


def validate(model, loader, k_loss_fn, depth_loss_fn,
             device, w_k, w_depth):
    model.eval()
    running = {"total": 0.0, "k": 0.0, "depth": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            fringe = batch["fringe"].to(device)
            k_gt = batch["k_gt"].to(device)
            mask = batch["mask_gt"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            d_min = torch.tensor([b for b in batch["d_min"]], device=device,
                                 dtype=torch.float32)
            d_max = torch.tensor([b for b in batch["d_max"]], device=device,
                                 dtype=torch.float32)

            out = model(fringe)
            L_k = k_loss_fn(out["k_logits"], k_gt, mask)
            L_depth = depth_loss_fn(out["depth_hat"], depth_gt, mask,
                                    d_min, d_max)
            total = w_k * L_k + w_depth * L_depth

            running["total"] += total.item()
            running["k"] += L_k.item()
            running["depth"] += L_depth.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tag = "phicalnet_hybrid"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("PhiCalHybridNet training (analytical phase + learned k)")
    print(f"Frames: F{FRAME_INDICES[0]}, F{FRAME_INDICES[1]}, F{FRAME_INDICES[2]}")
    print(f"Fringe source: {MULTIFRAME_ROOT}")
    print(f"Loss: w_k * CE(k) + w_depth * HybridL1(depth) "
          f"(no phase loss -- phi is analytical)")
    print("=" * 70)

    train_ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "train/fringe", PHASE_ROOT / "train",
        K_max=K_MAX, frame_indices=FRAME_INDICES,
    )
    val_ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "val/fringe", PHASE_ROOT / "val",
        K_max=K_MAX, frame_indices=FRAME_INDICES,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)
    model = PhiCalHybridNet(
        calibration_layer=cal_layer, frame_indices=FRAME_INDICES,
        K_max=K_MAX, gumbel_temperature=args.gumbel_temperature,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}  trainable: {trainable:,} "
          f"(k-head only)")

    k_loss_fn = FringeOrderLoss()
    depth_loss_fn = DepthLoss(alpha=0.7)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4,
                                    weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=10, min_lr=1e-7)

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_total", "train_k", "train_depth",
                "val_total", "val_k", "val_depth",
                "time_seconds", "learning_rate", "iteration", "w_depth",
            ])

    start_epoch = 0
    best_val = float("inf")
    iteration = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["loss"]
        iteration = ckpt.get("iteration", 0)

    w_k = args.w_k
    w_depth = args.w_depth

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, k_loss_fn, depth_loss_fn,
            device, w_k, w_depth, iteration,
        )
        val_stats = validate(
            model, val_loader, k_loss_fn, depth_loss_fn,
            device, w_k, w_depth,
        )

        scheduler.step(val_stats["total"])
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_stats["total"]), float(train_stats["k"]),
                float(train_stats["depth"]),
                float(val_stats["total"]), float(val_stats["k"]),
                float(val_stats["depth"]),
                int(round(dt)), float(lr), int(iteration), float(w_depth),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_stats['total']:.4f} (k={train_stats['k']:.4f} "
              f"d={train_stats['depth']:.4f}) | "
              f"Val: {val_stats['total']:.4f} (k={val_stats['k']:.4f} "
              f"d={val_stats['depth']:.4f}) | "
              f"LR: {lr:.2e} | It: {iteration}", end="")

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_stats["total"],
                "iteration": iteration,
                "frame_indices": FRAME_INDICES,
                "K_max": K_MAX,
            }, ckpt_dir / "best_model.pth")
            print(" <- NEW BEST!")
        else:
            print()

        if (epoch + 1) % 10 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_stats["total"],
                "iteration": iteration,
                "frame_indices": FRAME_INDICES,
                "K_max": K_MAX,
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-7 * 1.5:
            print("\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best val_total: {best_val:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_k", type=float, default=1.0,
                        help="Weight on fringe-order cross-entropy loss")
    parser.add_argument("--w_depth", type=float, default=0.1,
                        help="Weight on depth HybridL1 loss")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0,
                        help="Temperature for the differentiable argmax on k")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
