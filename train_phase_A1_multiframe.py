"""
Multi-frame PhiCalNet (A1 refined v2 loss, 3-channel input).

Identical to train_phase_A1_refined_v2.py except:
  (1) Input is the 3-frame stack F0, F6, F12 from
      `training_data_depth_individual_normalized_multiframe_horizontal_only`,
      loaded via PhaseMultiFrameDataset. These three frames are exactly 120
      degrees apart and span the classical 3-step phase-shift set, so the
      network has enough observational information to recover phase
      analytically.
  (2) Model is instantiated with in_channels=3.
  (3) Checkpoints go to checkpoints_phase_A1_multiframe/ so the single-frame
      run is untouched.

Everything else -- loss formulation, optimizer, scheduler, logging -- is
held constant, so the delta between the two runs is attributable to input
frame count.
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
from models.phase_unet import PhaseUNetFPP
from models.phase_fno_hybrid import PhaseFNOHybrid
from models.calibration_layer import CalibrationLayer
from data.phase_dataset_multiframe import PhaseMultiFrameDataset
from data.phase_dataset_multiframe_sumdiff import PhaseMultiFrameSumDiffDataset
from train_phase_A1_refined_v2 import train_epoch, validate

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
MULTIFRAME_ROOT = (
    BASE / "training_datasets"
         / "training_data_depth_individual_normalized_multiframe_horizontal_only"
)
DEFAULT_MULTIFRAME_ROOT = MULTIFRAME_ROOT
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32
FRAME_INDICES = (0, 6, 12)  # 120-deg phase shifts under the 18-step generator


def main(args):
    global FRAME_INDICES, MULTIFRAME_ROOT
    FRAME_INDICES = tuple(int(x) for x in args.frames.split(","))
    if args.data_root:
        MULTIFRAME_ROOT = Path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if FRAME_INDICES == (0, 6, 12):
        tag = "phase_A1_multiframe"   # original 3-frame run keeps its dirs
    else:
        tag = "phase_A1_multiframe_f" + "-".join(str(i) for i in FRAME_INDICES)
    if args.encode == "sumdiff":
        tag += "_sumdiff"
    if args.data_root:
        tag += "_" + Path(args.data_root).name.replace("multiframe_horizontal_", "")
    if args.backbone != "unet":
        tag += f"_{args.backbone}"
    if args.seed != 42:
        tag += f"_s{args.seed}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print(f"PhiCalNet multi-frame training ({len(FRAME_INDICES)} phase-shifted channels)")
    print(f"Frames: " + ", ".join(f"F{i}" for i in FRAME_INDICES))
    print(f"Fringe source: {MULTIFRAME_ROOT}")
    print("Loss formulation follows train_phase_A1_refined_v2.py exactly.")
    print("=" * 70)

    DS = PhaseMultiFrameSumDiffDataset if args.encode == "sumdiff" else PhaseMultiFrameDataset
    train_ds = DS(
        MULTIFRAME_ROOT / "train/fringe", PHASE_ROOT / "train",
        K_max=K_MAX, frame_indices=FRAME_INDICES,
    )
    val_ds = DS(
        MULTIFRAME_ROOT / "val/fringe", PHASE_ROOT / "val",
        K_max=K_MAX, frame_indices=FRAME_INDICES,
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=1, pin_memory=True)

    if args.backbone == "fnohybrid":
        model = PhaseFNOHybrid(in_channels=len(FRAME_INDICES), width=64, modes=32,
                               n_blocks=4, K_max=K_MAX).to(device)
    else:
        model = PhaseUNetFPP(variant='A1', in_channels=len(FRAME_INDICES),
                             K_max=K_MAX, dropout_rate=0.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (in_channels={len(FRAME_INDICES)})")

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
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

    w_phase = 1.0
    w_grad = 0.5
    boundary_alpha = args.boundary_alpha

    print("\nStarting training...\n")
    for epoch in range(start_epoch, 1000):
        t0 = time.time()

        w_depth = 0.1

        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha, iteration
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha
        )

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
                'in_channels': len(FRAME_INDICES),
                'frame_indices': FRAME_INDICES,
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
                'in_channels': len(FRAME_INDICES),
                'frame_indices': FRAME_INDICES,
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
    parser.add_argument("--frames", type=str, default="0,6,12",
                        help="Comma-separated frame indices, e.g. 0,6 for the 120-deg 2-frame set")
    parser.add_argument("--encode", type=str, default="raw", choices=["raw", "sumdiff"],
                        help="Input encoding: raw frames or (diff, sum)/2 channels (2-frame only)")
    parser.add_argument("--backbone", type=str, default="unet", choices=["unet", "fnohybrid"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Alternative multiframe root (e.g. gamma-distorted frames)")
    parser.add_argument("--boundary_alpha", type=float, default=3.0,
                        help="Boundary weight multiplier (weight = 1 + alpha*(|phi|/pi)^2)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
