"""
Training script for Variant 1: PhiCalNetLearnedK (learned phase + learned k).

PhaseUNet backbone predicts (sin phi, cos phi); a FringeOrderHead attached
to the backbone predicts k. Both heads are trained jointly.

Works at any frame count N>=1. Pass frame indices via --frames:

    python train_phicalnet_learnedk.py --frames 0              # 1-frame
    python train_phicalnet_learnedk.py --frames 0,6,12         # 3-frame
    python train_phicalnet_learnedk.py --frames 0,3,6,9,12,15  # 6-frame

Checkpoints and logs are tagged with the frame count:
    checkpoints_phicalnet_learnedk_{N}f/    logs_phicalnet_learnedk_{N}f/
so multiple configurations can run in parallel without clobbering.

Loss: w_phase * circular_phase_loss(phi, phi_gt)
    + w_grad  * phase_gradient_loss(sin, cos)
    + w_k     * CE(k_logits, k_gt)
    + w_depth * HybridL1(depth_hat, depth_gt)
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
from models.phicalnet_learnedk import PhiCalNetLearnedK
from train_ftp_variant import FringeOrderLoss, DepthLoss
from train_phase_A1_refined_v2 import circular_phase_loss
from train_phase_A1_refined import phase_gradient_loss

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
MULTIFRAME_ROOT = (
    BASE / "training_datasets"
         / "training_data_depth_individual_normalized_multiframe_horizontal_only"
)
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32


def parse_frames(s: str):
    indices = tuple(int(x) for x in s.split(",") if x.strip())
    if not indices:
        raise ValueError("--frames must be a non-empty comma-separated list")
    return indices


def train_epoch(model, loader, optimizer, k_loss_fn, depth_loss_fn,
                device, w_phase, w_grad, w_k, w_depth,
                boundary_alpha, iteration=0):
    model.train()
    running = {"total": 0.0, "phase": 0.0, "grad": 0.0, "k": 0.0, "depth": 0.0}
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        fringe = batch["fringe"].to(device)
        sin_gt = batch["sin_phi_gt"].to(device)
        cos_gt = batch["cos_phi_gt"].to(device)
        k_gt = batch["k_gt"].to(device)
        mask = batch["mask_gt"].to(device)
        depth_gt = batch["depth_gt"].to(device)
        d_min = torch.tensor([b for b in batch["d_min"]], device=device,
                             dtype=torch.float32)
        d_max = torch.tensor([b for b in batch["d_max"]], device=device,
                             dtype=torch.float32)

        optimizer.zero_grad()
        out = model(fringe)

        L_phase = circular_phase_loss(
            out["sin_phi_hat"], out["cos_phi_hat"], sin_gt, cos_gt, mask,
            boundary_alpha
        )
        L_grad = phase_gradient_loss(
            out["sin_phi_hat"], out["cos_phi_hat"], sin_gt, cos_gt, mask
        )
        L_k = k_loss_fn(out["k_logits"], k_gt, mask)
        L_depth = depth_loss_fn(out["depth_hat"], depth_gt, mask, d_min, d_max)
        loss = (w_phase * L_phase + w_grad * L_grad
                + w_k * L_k + w_depth * L_depth)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            iteration += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running["total"] += loss.item()
        running["phase"] += L_phase.item()
        running["grad"] += L_grad.item()
        running["k"] += L_k.item()
        running["depth"] += L_depth.item()
        iteration += 1
        pbar.set_postfix({
            "L": f"{loss.item():.4f}",
            "ph": f"{L_phase.item():.4f}",
            "k": f"{L_k.item():.4f}",
            "d": f"{L_depth.item():.4f}",
        })

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}, iteration


def validate(model, loader, k_loss_fn, depth_loss_fn,
             device, w_phase, w_grad, w_k, w_depth, boundary_alpha):
    model.eval()
    running = {"total": 0.0, "phase": 0.0, "grad": 0.0, "k": 0.0, "depth": 0.0}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            fringe = batch["fringe"].to(device)
            sin_gt = batch["sin_phi_gt"].to(device)
            cos_gt = batch["cos_phi_gt"].to(device)
            k_gt = batch["k_gt"].to(device)
            mask = batch["mask_gt"].to(device)
            depth_gt = batch["depth_gt"].to(device)
            d_min = torch.tensor([b for b in batch["d_min"]], device=device,
                                 dtype=torch.float32)
            d_max = torch.tensor([b for b in batch["d_max"]], device=device,
                                 dtype=torch.float32)

            out = model(fringe)
            L_phase = circular_phase_loss(
                out["sin_phi_hat"], out["cos_phi_hat"], sin_gt, cos_gt, mask,
                boundary_alpha
            )
            L_grad = phase_gradient_loss(
                out["sin_phi_hat"], out["cos_phi_hat"], sin_gt, cos_gt, mask
            )
            L_k = k_loss_fn(out["k_logits"], k_gt, mask)
            L_depth = depth_loss_fn(out["depth_hat"], depth_gt, mask,
                                    d_min, d_max)
            total = (w_phase * L_phase + w_grad * L_grad
                     + w_k * L_k + w_depth * L_depth)

            running["total"] += total.item()
            running["phase"] += L_phase.item()
            running["grad"] += L_grad.item()
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

    frame_indices = parse_frames(args.frames)
    N = len(frame_indices)

    # Include k-head variant in the tag so wide-head runs don't collide
    # with the default-head runs.
    suffix = "" if args.k_head_variant == "default" else f"_{args.k_head_variant}"
    tag = f"phicalnet_learnedk_{N}f{suffix}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print(f"PhiCalNetLearnedK training (learned phase + learned k)")
    print(f"Frame indices: {frame_indices}  (N = {N})")
    print(f"k-head variant: {args.k_head_variant}")
    print(f"Fringe source: {MULTIFRAME_ROOT}")
    print(f"Loss: w_phase={args.w_phase}*circular + w_grad={args.w_grad}*grad "
          f"+ w_k={args.w_k}*CE + w_depth={args.w_depth}*HybridL1")
    print("=" * 70)

    train_ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "train/fringe", PHASE_ROOT / "train",
        K_max=K_MAX, frame_indices=frame_indices,
    )
    val_ds = PhaseMultiFrameDataset(
        MULTIFRAME_ROOT / "val/fringe", PHASE_ROOT / "val",
        K_max=K_MAX, frame_indices=frame_indices,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)
    model = PhiCalNetLearnedK(
        calibration_layer=cal_layer, in_channels=N, K_max=K_MAX,
        gumbel_temperature=args.gumbel_temperature,
        k_head_variant=args.k_head_variant,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    khead_params = sum(p.numel() for p in model.k_head.parameters())
    print(f"Model parameters: {total_params:,} "
          f"(in_channels={N}, k-head={khead_params:,})")

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
                "epoch", "train_total", "train_phase", "train_grad",
                "train_k", "train_depth",
                "val_total", "val_phase", "val_grad", "val_k", "val_depth",
                "time_seconds", "learning_rate", "iteration",
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

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, k_loss_fn, depth_loss_fn,
            device, args.w_phase, args.w_grad, args.w_k, args.w_depth,
            args.boundary_alpha, iteration,
        )
        val_stats = validate(
            model, val_loader, k_loss_fn, depth_loss_fn,
            device, args.w_phase, args.w_grad, args.w_k, args.w_depth,
            args.boundary_alpha,
        )

        scheduler.step(val_stats["total"])
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_stats["total"]), float(train_stats["phase"]),
                float(train_stats["grad"]), float(train_stats["k"]),
                float(train_stats["depth"]),
                float(val_stats["total"]), float(val_stats["phase"]),
                float(val_stats["grad"]), float(val_stats["k"]),
                float(val_stats["depth"]),
                int(round(dt)), float(lr), int(iteration),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_stats['total']:.4f} "
              f"(ph={train_stats['phase']:.4f} k={train_stats['k']:.4f} "
              f"d={train_stats['depth']:.4f}) | "
              f"Val: {val_stats['total']:.4f} "
              f"(ph={val_stats['phase']:.4f} k={val_stats['k']:.4f} "
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
                "frame_indices": frame_indices,
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
                "frame_indices": frame_indices,
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
    parser.add_argument("--frames", type=str, required=True,
                        help="Comma-separated frame indices (e.g. '0' or '0,6,12')")
    parser.add_argument("--w_phase", type=float, default=1.0)
    parser.add_argument("--w_grad", type=float, default=0.5)
    parser.add_argument("--w_k", type=float, default=1.0)
    parser.add_argument("--w_depth", type=float, default=0.1)
    parser.add_argument("--boundary_alpha", type=float, default=3.0,
                        help="Wrap-boundary weight multiplier in phase loss")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0)
    parser.add_argument("--k_head_variant", type=str, default="default",
                        choices=["default", "wide"],
                        help="'default' = 128ch/3-middle (~0.38M params); "
                             "'wide' = 256ch/5-middle (~3.3M params)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
