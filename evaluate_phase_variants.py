"""
Evaluation for Phase-Intermediate UNet (A1 and A2).

Reports the standard benchmark metrics (Object/BG/Overall MAE and RMSE
in mm) matching the other architectures, plus phase-specific diagnostics:
    - wrapped-phase MAE (radians)
    - sin/cos MAE per component
    - For A2: fringe-order accuracy, confusion matrix,
      object MAE decomposed into "within correct k" vs "k misclassified"

Uses the full MATLAB unwrap pipeline (shift + triple median filter + round
correction) for depth reconstruction at evaluation time to match the
reference depth.mat reproduction to <1e-3 mm when inputs are correct.
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
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer import CalibrationLayer
from data.phase_dataset import PhaseDataset

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"

K_MAX = 32
T_P = 36.0


def full_matlab_unwrap(sin_np, cos_np, k_np):
    """
    Full MATLAB unwrap: shift-unshift, triple median filter, round correction.
    Non-differentiable (used at eval only).
    """
    shift = -np.pi / T_P
    wph = np.arctan2(sin_np, cos_np).astype(np.float64)
    # Shift + re-wrap
    wph_shifted = np.arctan2(np.sin(wph + shift), np.cos(wph + shift))
    uph = wph_shifted + 2.0 * np.pi * k_np.astype(np.float64) - shift
    # Triple median filter (matches MATLAB's repeated smoothing)
    suph = median_filter(uph, size=7)
    suph = median_filter(suph, size=7)
    suph = median_filter(suph, size=7)
    # Round correction removes off-by-2pi outliers
    unwrPhase = uph - np.round((uph - suph) / (2 * np.pi)) * 2 * np.pi
    return unwrPhase


def evaluate_variant(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading {args.variant} model from {args.checkpoint}")
    model = PhaseUNetFPP(variant=args.variant, in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', -1) + 1
    print(f"Model loaded from epoch {epoch}")

    # Dataset
    test_ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)

    cal_layer = CalibrationLayer(str(CAL_DIR)).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # For A2 confusion matrix
    if args.variant == 'A2':
        k_confusion = np.zeros((K_MAX, K_MAX), dtype=np.int64)
        within_k_errors_all = []
        misclassified_k_errors_all = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
            sample = test_ds[idx]
            fringe = sample['fringe'].unsqueeze(0).to(device)
            sin_gt_t = sample['sin_phi_gt'].unsqueeze(0).to(device)
            cos_gt_t = sample['cos_phi_gt'].unsqueeze(0).to(device)
            k_gt_np = sample['k_gt'].numpy()
            mask_np = sample['mask_gt'].squeeze().numpy() > 0
            depth_gt_mm = sample['depth_gt'].numpy().squeeze()
            name = sample['name']

            # Forward
            out = model(fringe)
            sin_hat = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            cos_hat = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)

            # Select k: GT for A1, predicted for A2
            if args.variant == 'A1':
                k_use = k_gt_np
            else:
                k_hat = out['order_logits'].argmax(dim=1).squeeze().cpu().numpy()
                k_use = k_hat

            # Run full MATLAB unwrap + calibration for clean evaluation
            uph = full_matlab_unwrap(sin_hat, cos_hat, k_use)
            uph_t = torch.from_numpy(uph).unsqueeze(0).unsqueeze(0).float().to(device)
            depth_pred_mm = cal_layer(uph_t).squeeze().cpu().numpy()

            # Standard metrics in mm
            err = np.abs(depth_pred_mm - depth_gt_mm)
            obj_mae = float(err[mask_np].mean()) if mask_np.sum() > 0 else 0.0
            obj_rmse = float(np.sqrt((err[mask_np] ** 2).mean())) if mask_np.sum() > 0 else 0.0
            bg_mae = float(err[~mask_np].mean()) if (~mask_np).sum() > 0 else 0.0
            bg_rmse = float(np.sqrt((err[~mask_np] ** 2).mean())) if (~mask_np).sum() > 0 else 0.0
            overall_mae = float(err.mean())
            overall_rmse = float(np.sqrt((err ** 2).mean()))

            # Phase-specific metrics
            sin_gt = sample['sin_phi_gt'].squeeze().numpy().astype(np.float64)
            cos_gt = sample['cos_phi_gt'].squeeze().numpy().astype(np.float64)
            phi_gt = np.arctan2(sin_gt, cos_gt)
            phi_hat = np.arctan2(sin_hat, cos_hat)
            # Wrapped-phase MAE uses min(|dphi|, 2pi-|dphi|) to handle wrapping
            dphi = np.abs(phi_hat - phi_gt)
            dphi_wrapped = np.minimum(dphi, 2 * np.pi - dphi)
            wrapped_phase_mae = float(dphi_wrapped[mask_np].mean()) if mask_np.sum() > 0 else 0.0
            sin_mae = float(np.abs(sin_hat - sin_gt)[mask_np].mean()) if mask_np.sum() > 0 else 0.0
            cos_mae = float(np.abs(cos_hat - cos_gt)[mask_np].mean()) if mask_np.sum() > 0 else 0.0

            metric = {
                'name': name,
                'object_mae': obj_mae,
                'object_rmse': obj_rmse,
                'overall_mae': overall_mae,
                'overall_rmse': overall_rmse,
                'bg_mae': bg_mae,
                'bg_rmse': bg_rmse,
                'wrapped_phase_mae_rad': wrapped_phase_mae,
                'sin_mae': sin_mae,
                'cos_mae': cos_mae,
            }

            if args.variant == 'A2':
                # Fringe order accuracy (object pixels only)
                k_correct = (k_use == k_gt_np) & mask_np
                k_total_obj = mask_np.sum()
                k_acc = float(k_correct.sum()) / max(k_total_obj, 1)

                # Confusion matrix on object pixels
                for gt_k_val, pred_k_val in zip(k_gt_np[mask_np].flatten(),
                                                k_use[mask_np].flatten()):
                    if 0 <= gt_k_val < K_MAX and 0 <= pred_k_val < K_MAX:
                        k_confusion[gt_k_val, pred_k_val] += 1

                # Decompose object error by k correctness
                correct_mask = (k_use == k_gt_np) & mask_np
                wrong_mask = (k_use != k_gt_np) & mask_np
                within_k_mae = float(err[correct_mask].mean()) if correct_mask.sum() > 0 else 0.0
                miscls_k_mae = float(err[wrong_mask].mean()) if wrong_mask.sum() > 0 else 0.0

                metric['k_accuracy'] = k_acc
                metric['within_correct_k_mae'] = within_k_mae
                metric['k_misclassified_mae'] = miscls_k_mae
                metric['within_correct_k_pct'] = float(correct_mask.sum()) / max(mask_np.sum(), 1) * 100

                within_k_errors_all.append(err[correct_mask])
                misclassified_k_errors_all.append(err[wrong_mask])

            all_metrics.append(metric)

    # Aggregate
    def agg(key):
        vals = [m[key] for m in all_metrics]
        return np.mean(vals)

    print("\n" + "=" * 70)
    print(f"Phase-Intermediate {args.variant} Evaluation Results")
    print("=" * 70)
    print(f"Overall MAE:            {agg('overall_mae'):.4f} mm")
    print(f"Overall RMSE:           {agg('overall_rmse'):.4f} mm")
    print(f"Object MAE:             {agg('object_mae'):.4f} mm")
    print(f"Object RMSE:            {agg('object_rmse'):.4f} mm")
    print(f"Background MAE:         {agg('bg_mae'):.4f} mm")
    print(f"Background RMSE:        {agg('bg_rmse'):.4f} mm")
    print(f"Wrapped phase MAE:      {agg('wrapped_phase_mae_rad'):.6f} rad")
    print(f"sin(phi) MAE:           {agg('sin_mae'):.6f}")
    print(f"cos(phi) MAE:           {agg('cos_mae'):.6f}")

    if args.variant == 'A2':
        print()
        k_acc_mean = agg('k_accuracy') * 100
        print(f"Fringe-order accuracy:  {k_acc_mean:.2f} %")

        # Global decomposition (concatenate errors across samples)
        within_all = np.concatenate(within_k_errors_all) if within_k_errors_all else np.array([])
        miscls_all = np.concatenate(misclassified_k_errors_all) if misclassified_k_errors_all else np.array([])
        print(f"Within correct-k object MAE:     {within_all.mean():.4f} mm ({len(within_all)} px)")
        print(f"K-misclassified object MAE:      {miscls_all.mean() if len(miscls_all)>0 else 0:.4f} mm ({len(miscls_all)} px)")

        # Save confusion matrix
        np.save(save_dir / "k_confusion_matrix.npy", k_confusion)
        print(f"K confusion matrix saved to: {save_dir / 'k_confusion_matrix.npy'}")

    # Per-sample CSV
    csv_path = save_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nPer-sample metrics saved to: {csv_path}")

    # Summary
    summary_path = save_dir / "summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Phase-Intermediate {args.variant} Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Object MAE:   {agg('object_mae'):.4f} mm\n")
        f.write(f"Object RMSE:  {agg('object_rmse'):.4f} mm\n")
        f.write(f"Overall MAE:  {agg('overall_mae'):.4f} mm\n")
        f.write(f"Overall RMSE: {agg('overall_rmse'):.4f} mm\n")
        f.write(f"BG MAE:       {agg('bg_mae'):.4f} mm\n")
        f.write(f"BG RMSE:      {agg('bg_rmse'):.4f} mm\n")
        f.write(f"Wrapped phase MAE (rad): {agg('wrapped_phase_mae_rad'):.6f}\n")
        if args.variant == 'A2':
            f.write(f"k accuracy:              {agg('k_accuracy')*100:.2f}%\n")
            f.write(f"Within-correct-k MAE:    {within_all.mean():.4f} mm\n")
            f.write(f"K-misclassified MAE:     {miscls_all.mean() if len(miscls_all)>0 else 0:.4f} mm\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=['A1', 'A2'])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate_variant(args)
