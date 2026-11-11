"""
Train PhaseFNO_FPP (Fourier Neural Operator backbone) on the single-shot
PhiCalNet pipeline.

Identical to train_phase_A1_refined_v2.py except:
  * Backbone is PhaseFNO_FPP instead of PhaseUNetFPP.
  * Checkpoint and log directories are tagged 'phase_fno' so the existing
    UNet baseline run is untouched.

Loss formulation, optimizer (RMSprop), scheduler (ReduceLROnPlateau,
min_lr 1e-7), boundary-weighted circular phase loss, gradient loss, and
HybridL1 depth loss are all held fixed. The delta vs the UNet baseline
is therefore attributable to the architectural inductive bias.
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

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from models.calibration_layer import CalibrationLayer
from models.phase_fno import PhaseFNO_FPP
from train_phase_variant import depth_loss_hybrid, compute_depth_from_phase
from train_phase_A1_refined import phase_gradient_loss
from train_phase_A1_refined_v2 import (
    circular_phase_loss, train_epoch, validate,
)

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tag = f"phase_fno_w{args.width}_m{args.modes}_b{args.n_blocks}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print("PhaseFNO single-shot training (Fourier Neural Operator backbone)")
    print(f"FNO config: width={args.width}, modes={args.modes}, "
          f"n_blocks={args.n_blocks}")
    print(f"Loss formulation follows train_phase_A1_refined_v2.py exactly.")
    print("=" * 70)

    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train",
                            K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val",
                          K_max=K_MAX)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    model = PhaseFNO_FPP(
        in_channels=1,
        width=args.width,
        modes=args.modes,
        n_blocks=args.n_blocks,
        K_max=K_MAX,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4,
                                    weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_total", "train_phase", "train_grad",
                "train_depth",
                "val_total", "val_phase", "val_grad", "val_depth",
                "time_seconds", "learning_rate", "iteration", "w_depth",
            ])

    start_epoch = 0
    best_val_phase = float("inf")
    iteration = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_phase = ckpt["loss"]
        iteration = ckpt.get("iteration", 0)

    w_phase = 1.0
    w_grad = 0.5
    boundary_alpha = args.boundary_alpha

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()
        w_depth = 0.1  # fixed (matches refined_v2)

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha, iteration,
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha,
        )

        scheduler.step(val_stats["phase"])
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                float(train_stats["total"]), float(train_stats["phase"]),
                float(train_stats["grad"]),  float(train_stats["depth"]),
                float(val_stats["total"]),   float(val_stats["phase"]),
                float(val_stats["grad"]),    float(val_stats["depth"]),
                int(round(dt)), float(lr), int(iteration), float(w_depth),
            ])

        print(f"Epoch [{epoch+1:3d}/1000] ({dt:5.1f}s) | "
              f"Train: {train_stats['total']:.4f} (ph={train_stats['phase']:.4f} "
              f"g={train_stats['grad']:.4f} d={train_stats['depth']:.4f}) | "
              f"Val: ph={val_stats['phase']:.4f} | w_d={w_depth:.3f} | "
              f"LR: {lr:.2e} | It: {iteration}", end="")

        if val_stats["phase"] < best_val_phase:
            best_val_phase = val_stats["phase"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_stats["phase"],
                "iteration": iteration,
                "fno_config": {
                    "width": args.width, "modes": args.modes,
                    "n_blocks": args.n_blocks,
                },
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
                "loss": val_stats["phase"],
                "iteration": iteration,
                "fno_config": {
                    "width": args.width, "modes": args.modes,
                    "n_blocks": args.n_blocks,
                },
            }, ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pth")

        if lr <= 1e-7 * 1.5:
            print("\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best val_phase: {best_val_phase:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64,
                        help="FNO channel width")
    parser.add_argument("--modes", type=int, default=32,
                        help="Number of Fourier modes per axis")
    parser.add_argument("--n_blocks", type=int, default=4,
                        help="Number of FNO blocks")
    parser.add_argument("--boundary_alpha", type=float, default=3.0,
                        help="Wrap-boundary weight in circular phase loss")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
