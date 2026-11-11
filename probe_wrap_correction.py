"""
Wrap-tail CORRECTION probe (training-free).

The paper's UQ result: snapshot disagreement localizes the 0.103% wrap-boundary
pixels carrying all of PhiCalNet's RMSE; conformal REJECTION of the top 5%
cuts snapshot-mean RMSE 20.6 -> 7.4 mm. This probe asks: can those pixels be
CORRECTED instead of dropped? A wrap failure is a +/-2*pi phase error, so the
fix is an integer-period snap toward a trusted local reference.

Estimators compared on the test split (object MAE / RMSE / tail>100mm, no
pixels dropped anywhere):
  A  single best model (paper baseline, 4.46 / 17.82)
  B  snapshot-mean depth  (paper UQ baseline, RMSE ~20.6)
  C  circular-mean: average snapshot (sin,cos) coherently, then unwrap once
  D  C + wide 2*pi snap: ref = median_filter(Phi, 15), snap everywhere
  E  UQ-guided snap: flag top 5% by snapshot std, snap ONLY flagged pixels
     toward a reference built from trusted pixels
  F  E applied to the single best model (flag via |Phi - ref|, no ensemble)

Snapshots: epochs 100-140 (identical to conformal_snapshot.py / the paper).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from data.phase_dataset import PhaseDataset
from models.calibration_layer import CalibrationLayer
from conformal_snapshot import full_unwrap, load_snapshots

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = BASE / "training_datasets/training_data_depth_individual_normalized"
PHASE_ROOT = BASE / "training_datasets/phase_targets"
CAL_DIR = BASE / "calibration_matrices"
K_MAX = 32
TWO_PI = 2.0 * np.pi


def snap(Phi, ref):
    return Phi - np.round((Phi - ref) / TWO_PI) * TWO_PI


def stats(depth, depth_gt, mask, acc):
    err = np.abs(depth - depth_gt)[mask]
    acc['mae'].append(err.mean())
    acc['rmse'].append(np.sqrt((err ** 2).mean()))
    acc['tail'] += int((err > 100).sum())
    acc['n'] += int(mask.sum())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_snapshots([100, 110, 120, 130, 140], device)
    cal = CalibrationLayer(str(CAL_DIR)).to(device)
    ds = PhaseDataset(DATA_ROOT / "test/fringe", PHASE_ROOT / "test", K_max=K_MAX)

    keys = ['A_single', 'B_snapmean', 'C_circmean', 'D_C+snap', 'E_uq_snap', 'F_single+snap']
    accs = {k: {'mae': [], 'rmse': [], 'tail': 0, 'n': 0} for k in keys}

    def to_depth(Phi):
        t = torch.from_numpy(Phi)[None, None].float().to(device)
        return cal(t).squeeze().cpu().numpy()

    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            fringe = s['fringe'].unsqueeze(0).to(device)
            k_gt = s['k_gt'].numpy()
            mask = s['mask_gt'].squeeze().numpy() > 0
            depth_gt = s['depth_gt'].numpy().squeeze()

            sins, coss, Phis, depths = [], [], [], []
            for m in models:
                out = m(fringe)
                sh = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
                ch = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
                sins.append(sh); coss.append(ch)
                Phi_n = full_unwrap(sh, ch, k_gt)
                Phis.append(Phi_n)
                depths.append(to_depth(Phi_n))
            depths = np.stack(depths)
            Phis = np.stack(Phis)

            # A: single best model = snapshot at epoch 120 is NOT best_model;
            #    use the middle snapshot as proxy is wrong -- load separately below.
            # (handled after loop via separate pass; here ensemble estimators)

            # B: paper's snapshot-mean depth
            stats(depths.mean(axis=0), depth_gt, mask, accs['B_snapmean'])

            # C: coherent circular mean, then one unwrap
            sm, cm = np.mean(sins, axis=0), np.mean(coss, axis=0)
            nrm = np.sqrt(sm ** 2 + cm ** 2 + 1e-12)
            Phi_c = full_unwrap(sm / nrm, cm / nrm, k_gt)
            stats(to_depth(Phi_c), depth_gt, mask, accs['C_circmean'])

            # D: C + wide 2pi snap everywhere
            ref = median_filter(Phi_c, size=15)
            stats(to_depth(snap(Phi_c, ref)), depth_gt, mask, accs['D_C+snap'])

            # E: UQ-guided snap on the snapshot-mean phase
            Phi_mean = Phis.mean(axis=0)
            std = depths.std(axis=0)
            thr = np.quantile(std[mask], 0.95)
            flag = std >= thr
            trusted = Phi_mean.copy()
            trusted[flag] = median_filter(Phi_mean, size=31)[flag]
            ref_e = median_filter(trusted, size=15)
            Phi_e = Phi_mean.copy()
            Phi_e[flag] = snap(Phi_mean, ref_e)[flag]
            stats(to_depth(Phi_e), depth_gt, mask, accs['E_uq_snap'])

    # A and F: separate pass with the best model
    from models.phase_unet import PhaseUNetFPP
    best = PhaseUNetFPP(variant='A1', in_channels=1, K_max=K_MAX, dropout_rate=0.0)
    ck = torch.load("checkpoints_phase_A1_refined_v2/best_model.pth", map_location=device)
    best.load_state_dict(ck['model_state_dict'])
    best = best.to(device).eval()
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = best(s['fringe'].unsqueeze(0).to(device))
            sh = out['sin_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            ch = out['cos_phi_hat'].squeeze().cpu().numpy().astype(np.float64)
            k_gt = s['k_gt'].numpy()
            mask = s['mask_gt'].squeeze().numpy() > 0
            depth_gt = s['depth_gt'].numpy().squeeze()
            Phi = full_unwrap(sh, ch, k_gt)
            stats(to_depth(Phi), depth_gt, mask, accs['A_single'])
            ref = median_filter(Phi, size=15)
            flag = np.abs(Phi - ref) > np.pi   # single-model flag: gross deviation
            Phi_f = Phi.copy()
            Phi_f[flag] = snap(Phi, ref)[flag]
            stats(to_depth(Phi_f), depth_gt, mask, accs['F_single+snap'])

    print(f"\n{'estimator':16s} {'obj MAE':>8s} {'obj RMSE':>9s} {'tail>100mm':>11s}")
    print("-" * 50)
    for k in keys:
        a = accs[k]
        print(f"{k:16s} {np.mean(a['mae']):8.2f} {np.mean(a['rmse']):9.2f} "
              f"{a['tail']:6d} ({100*a['tail']/max(a['n'],1):.3f}%)")

    _persist(keys, accs)


def _persist(keys, accs):
    """Persist the table so the paper's numbers have an artifact (audit trail)."""
    import json
    out = {}
    for k in keys:
        a = accs[k]
        out[k] = {"mae_mm": float(np.mean(a['mae'])),
                  "rmse_mm": float(np.mean(a['rmse'])),
                  "tail_frac": float(a['tail'] / max(a['n'], 1))}
    json.dump(out, open(Path(__file__).resolve().parent / "wrap_correction.json", "w"), indent=1)
    print("Wrote wrap_correction.json")


if __name__ == "__main__":
    main()
