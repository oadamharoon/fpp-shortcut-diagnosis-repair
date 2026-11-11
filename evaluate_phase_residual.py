"""
Evaluation for PhaseUNetResidual model.
Same as evaluate_phase_variants.py A1 but uses the residual model and adds residual_mm.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.phase_unet_residual import PhaseUNetResidual
from models.calibration_layer import CalibrationLayer
from data.phase_dataset import PhaseDataset

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32
T_P = 36.0


def full_unwrap(sin_np, cos_np, k_np):
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    wph_s = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_s + 2 * np.pi * k_np.astype(np.float64) - shift
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    return uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading residual model from {args.checkpoint}")
    model = PhaseUNetResidual(in_channels=1, dropout_rate=0.0, residual_scale=50.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)
    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
            sample = test_ds[idx]
            fringe = sample['fringe'].unsqueeze(0).to(device)
            k_gt_np = sample['k_gt'].numpy()
            mask_np = sample['mask_gt'].squeeze().numpy() > 0
            depth_gt_mm = sample['depth_gt'].numpy().squeeze()
            name = sample['name']

            out = model(fringe)
            sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            residual_mm = out['residual_mm'].squeeze().cpu().numpy()

            # Compute depth from phase via full MATLAB unwrap
            uph = full_unwrap(sin_hat, cos_hat, k_gt_np)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth_phase_mm = cal_layer(uph_t).squeeze().cpu().numpy()

            # Add residual correction
            depth_final_mm = depth_phase_mm + residual_mm

            # Metrics (both phase-only and phase+residual)
            err_phase = np.abs(depth_phase_mm - depth_gt_mm)
            err_final = np.abs(depth_final_mm - depth_gt_mm)

            obj_mae_phase = float(err_phase[mask_np].mean())
            obj_mae_final = float(err_final[mask_np].mean())
            obj_rmse_final = float(np.sqrt((err_final[mask_np] ** 2).mean()))
            bg_mae_final = float(err_final[~mask_np].mean())
            bg_rmse_final = float(np.sqrt((err_final[~mask_np] ** 2).mean()))
            overall_mae = float(err_final.mean())
            overall_rmse = float(np.sqrt((err_final ** 2).mean()))

            residual_abs_mean = float(np.abs(residual_mm[mask_np]).mean())
            residual_max = float(np.abs(residual_mm[mask_np]).max())

            all_metrics.append({
                'name': name,
                'object_mae_phase_only': obj_mae_phase,
                'object_mae_final': obj_mae_final,
                'object_rmse_final': obj_rmse_final,
                'overall_mae': overall_mae,
                'overall_rmse': overall_rmse,
                'bg_mae': bg_mae_final,
                'bg_rmse': bg_rmse_final,
                'residual_abs_mean_mm': residual_abs_mean,
                'residual_abs_max_mm': residual_max,
            })

    def agg(key):
        return float(np.mean([m[key] for m in all_metrics]))

    print("\n" + "=" * 70)
    print("Phase A1 with Residual Correction — Evaluation Results")
    print("=" * 70)
    print(f"Object MAE (phase only):      {agg('object_mae_phase_only'):.4f} mm")
    print(f"Object MAE (phase+residual):  {agg('object_mae_final'):.4f} mm")
    print(f"Object RMSE (phase+residual): {agg('object_rmse_final'):.4f} mm")
    print(f"Overall MAE:                  {agg('overall_mae'):.4f} mm")
    print(f"Overall RMSE:                 {agg('overall_rmse'):.4f} mm")
    print(f"BG MAE:                       {agg('bg_mae'):.4f} mm")
    print(f"BG RMSE:                      {agg('bg_rmse'):.4f} mm")
    print()
    print(f"Mean |residual|:              {agg('residual_abs_mean_mm'):.4f} mm")
    print(f"Max |residual| (avg):         {agg('residual_abs_max_mm'):.4f} mm")
    print()
    print(f"Improvement from residual:    "
          f"{agg('object_mae_phase_only') - agg('object_mae_final'):.4f} mm")

    csv_path = save_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nPer-sample metrics saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
