"""
PhiCalNet loss-term ablation trainer.

Retrains PhiCalNet (the paper's a1_refined_v2 model) with exactly ONE loss
weight set to zero, holding everything else fixed. It imports the unmodified
train_epoch / validate / loss functions from train_phase_A1_refined_v2 so the
loss math, optimizer, LR schedule, data pipeline, oracle-k handling, seeds, and
stopping rule are identical to the reference run; the ONLY difference is which
weight is zeroed.

IMPORTANT (paper vs code): the reference PhiCalNet uses a THREE-term loss
    L = w_circ * L_circ + w_grad * L_grad + w_depth * L_depth
        (w_circ, w_grad, w_depth) = (1.0, 0.5, 0.1)
where L_circ = circular_phase_loss is the PRIMARY phase-fidelity term. There is
no separate L_sin_cos term in the actual code, so only three ablations exist:

  --drop circular : w_circ  = 0  (remove the primary direct phase supervision;
                                  phase is then shaped only by L_grad and by
                                  depth-through-calibration. Doubles as the
                                  "is depth-through-calibration enough?" test.)
  --drop grad     : w_grad   = 0  (remove the smoothness term; safety check)
  --drop depth    : w_depth  = 0  (remove depth supervision through the
                                  calibration layer; the architecturally
                                  critical gradient-bridge ablation)

Model selection (best checkpoint by lowest validation circular loss) is held
identical to the reference run, so "everything else" truly matches.

Usage:
    python train_phicalnet_ablation.py --drop circular
    python train_phicalnet_ablation.py --drop grad
    python train_phicalnet_ablation.py --drop depth
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

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer import CalibrationLayer
from data.phase_dataset import PhaseDataset
# Reuse the reference run's training/eval loops verbatim — guarantees identical
# loss computation, gradient clipping, NaN handling, and logging.
from train_phase_A1_refined_v2 import train_epoch, validate

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32

# Reference (full-loss) weights, identical to train_phase_A1_refined_v2.py.
BASE_W_CIRC = 1.0   # circular_phase_loss  (paper w_c; the primary phase term)
BASE_W_GRAD = 0.5   # phase_gradient_loss  (paper w_g)
BASE_W_DEPTH = 0.1  # depth_loss_hybrid    (paper w_d)

DROP_TO_WEIGHTS = {
    "circular": (0.0, BASE_W_GRAD, BASE_W_DEPTH),
    "grad":     (BASE_W_CIRC, 0.0, BASE_W_DEPTH),
    "depth":    (BASE_W_CIRC, BASE_W_GRAD, 0.0),
}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Identical seeding to the reference run.
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    w_phase, w_grad, w_depth = DROP_TO_WEIGHTS[args.drop]
    boundary_alpha = args.boundary_alpha
    tag = f"phicalnet_ablation_{args.drop}"
    ckpt_dir = Path(f"checkpoints_{tag}")
    log_dir = Path(f"logs_{tag}")

    print("=" * 70)
    print(f"PhiCalNet loss ablation — DROP '{args.drop}'")
    print("=" * 70)
    print(f"Weights (circular, grad, depth) = ({w_phase}, {w_grad}, {w_depth})")
    print(f"Reference weights               = ({BASE_W_CIRC}, {BASE_W_GRAD}, {BASE_W_DEPTH})")
    if args.drop == "depth":
        print("*" * 70)
        print("* w_depth = 0 : architecturally critical ablation.")
        print("* This removes the gradient bridge through the calibration layer.")
        print("* Watch val_phase / object error below: if depth supervision is")
        print("* doing the work the design claims, phase quality should degrade.")
        print("*" * 70)
    print("=" * 70)

    train_ds = PhaseDataset(DATA_ROOT / "train/fringe", PHASE_ROOT / "train", K_max=K_MAX)
    val_ds = PhaseDataset(DATA_ROOT / "val/fringe", PHASE_ROOT / "val", K_max=K_MAX)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    model = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'train_total', 'train_phase', 'train_grad', 'train_depth',
            'val_total', 'val_phase', 'val_grad', 'val_depth',
            'time_seconds', 'learning_rate', 'iteration',
            'w_circ', 'w_grad', 'w_depth',
        ])

    best_val_phase = float('inf')
    iteration = 0
    print("\nStarting training...\n")
    for epoch in range(0, 1000):
        t0 = time.time()
        train_stats, iteration = train_epoch(
            model, train_loader, optimizer, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha, iteration
        )
        val_stats = validate(
            model, val_loader, cal_layer, device,
            w_phase, w_grad, w_depth, boundary_alpha
        )

        # Hold the selection metric identical to the reference run: lowest
        # validation circular (phase) loss, regardless of which term is dropped.
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
                int(round(dt)), float(lr), int(iteration),
                float(w_phase), float(w_grad), float(w_depth),
            ])

        print(f"Epoch [{epoch+1:3d}] ({dt:5.1f}s) | "
              f"Train tot={train_stats['total']:.4f} "
              f"(ph={train_stats['phase']:.4f} g={train_stats['grad']:.4f} d={train_stats['depth']:.4f}) | "
              f"Val ph={val_stats['phase']:.4f} d={val_stats['depth']:.4f} | "
              f"LR={lr:.2e}", end="")

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
                'drop': args.drop,
                'weights': {'circ': w_phase, 'grad': w_grad, 'depth': w_depth},
            }, ckpt_dir / "best_model.pth")
            print(" <- NEW BEST!")
        else:
            print()

        if lr <= 1e-7 * 1.5:
            print("\nLearning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print(f"Ablation '{args.drop}' training complete. Best val_phase: {best_val_phase:.6f}")
    print(f"Best model: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", required=True, choices=["circular", "grad", "depth"],
                        help="Which loss term to zero out")
    parser.add_argument("--boundary_alpha", type=float, default=3.0,
                        help="Boundary weight multiplier (must match reference: 3.0)")
    args = parser.parse_args()
    main(args)
