"""
Evaluate FTP architecture on test set.

Computes standard metrics (Object/Background/Overall MAE and RMSE in mm)
matching the format used by vis_pred.py for other architectures.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.calibration_layer import CalibrationLayer
from models.ftp_architecture import FTPUNet
from data.ftp_dataset import FTPDataset

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"
PARAMS_DIR = BASE / "training_datasets/info_depth_params"


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    cal = CalibrationLayer(str(CAL_DIR))
    model = FTPUNet(cal, filter_type='parametric', fu0=-21, fv0=0, K_max=32)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {ckpt['epoch'] + 1}")

    # Print filter state
    f = model.bp_filter
    print(f"Filter: fu0={f.fu0.item():.2f}, fv0={f.fv0.item():.2f}, "
          f"sigma_u={f.log_sigma_u.clamp(-2,8).exp().item():.2f}, "
          f"sigma_v={f.log_sigma_v.clamp(-2,8).exp().item():.2f}")

    # Load test set
    test_ds = FTPDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
            sample = test_ds[idx]
            fringe = sample['fringe'].unsqueeze(0).to(device)
            depth_gt_mm = sample['depth_gt'].numpy().squeeze()  # (H, W) in mm
            mask = sample['mask_gt'].numpy().squeeze() > 0      # (H, W) bool
            name = sample['name']
            d_min = sample['d_min']
            d_max = sample['d_max']

            # Forward pass
            out = model(fringe)
            depth_pred_mm = out['depth_hat'].cpu().numpy().squeeze()  # (H, W)

            # Compute errors in mm
            error = np.abs(depth_pred_mm - depth_gt_mm)

            # Object metrics
            if mask.sum() > 0:
                obj_errors = error[mask]
                obj_mae = float(obj_errors.mean())
                obj_rmse = float(np.sqrt(np.mean(obj_errors ** 2)))
            else:
                obj_mae = obj_rmse = 0.0

            # Background metrics
            bg_mask = ~mask
            if bg_mask.sum() > 0:
                # For background: GT depth is d_min (from normalization)
                # but conceptually background should be 0 or near-0 depth
                # Match vis_pred.py convention: use denormalized GT
                bg_errors = error[bg_mask]
                bg_mae = float(bg_errors.mean())
                bg_rmse = float(np.sqrt(np.mean(bg_errors ** 2)))
            else:
                bg_mae = bg_rmse = 0.0

            # Overall metrics
            overall_mae = float(error.mean())
            overall_rmse = float(np.sqrt(np.mean(error ** 2)))

            all_metrics.append({
                'name': name,
                'object_mae': obj_mae,
                'object_rmse': obj_rmse,
                'bg_mae': bg_mae,
                'bg_rmse': bg_rmse,
                'overall_mae': overall_mae,
                'overall_rmse': overall_rmse,
            })

    # Aggregate
    obj_maes = [m['object_mae'] for m in all_metrics]
    obj_rmses = [m['object_rmse'] for m in all_metrics]
    bg_maes = [m['bg_mae'] for m in all_metrics]
    bg_rmses = [m['bg_rmse'] for m in all_metrics]
    overall_maes = [m['overall_mae'] for m in all_metrics]
    overall_rmses = [m['overall_rmse'] for m in all_metrics]

    print("\n" + "=" * 70)
    print("OVERALL ERROR STATISTICS (all pixels, in mm)")
    print("=" * 70)
    print(f"MAE:   avg={np.mean(overall_maes):.2f}, std={np.std(overall_maes):.2f}, "
          f"median={np.median(overall_maes):.2f}")
    print(f"       min={np.min(overall_maes):.2f}, max={np.max(overall_maes):.2f}")
    print(f"RMSE:  avg={np.mean(overall_rmses):.2f}, std={np.std(overall_rmses):.2f}")

    print("\n" + "=" * 70)
    print("OBJECT-ONLY STATISTICS (pixels where GT > 0, in mm)")
    print("=" * 70)
    print(f"MAE:   avg={np.mean(obj_maes):.2f}, std={np.std(obj_maes):.2f}, "
          f"median={np.median(obj_maes):.2f}")
    print(f"       min={np.min(obj_maes):.2f}, max={np.max(obj_maes):.2f}")
    print(f"RMSE:  avg={np.mean(obj_rmses):.2f}, std={np.std(obj_rmses):.2f}")

    print("\n" + "=" * 70)
    print("BACKGROUND STATISTICS (predicted depth where GT = 0, in mm)")
    print("=" * 70)
    print(f"MAE:   avg={np.mean(bg_maes):.2f}, std={np.std(bg_maes):.2f}, "
          f"median={np.median(bg_maes):.2f}")
    print(f"       min={np.min(bg_maes):.2f}, max={np.max(bg_maes):.2f}")
    print(f"RMSE:  avg={np.mean(bg_rmses):.2f}, std={np.std(bg_rmses):.2f}")

    # Save per-sample metrics CSV
    csv_path = save_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to: {csv_path}")

    # Save summary
    summary_path = save_dir / "summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FTP Architecture (Variant D) Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall MAE:  {np.mean(overall_maes):.2f} mm\n")
        f.write(f"Overall RMSE: {np.mean(overall_rmses):.2f} mm\n")
        f.write(f"Object MAE:   {np.mean(obj_maes):.2f} mm\n")
        f.write(f"Object RMSE:  {np.mean(obj_rmses):.2f} mm\n")
        f.write(f"BG MAE:       {np.mean(bg_maes):.2f} mm\n")
        f.write(f"BG RMSE:      {np.mean(bg_rmses):.2f} mm\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="visualizations_ftp_D")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
