"""Multi-frame training on the vertical (u_p) carrier.

Multi-channel counterpart to train_phase_A1_vertical.py: stacks several
phase-shifted frames of the vertical fringe direction as input channels,
mirroring the multi-frame configurations of train_phase_A1_multiframe.py on
the second carrier.

Vertical specifics: frames are stored F25..F42, so a frame offset j maps to
file F(25+j); K_MAX=64 (fringe order reaches ~51 at the 18 px period); the
depth loss triangulates through CalibrationLayerVertical with T_P_VERTICAL.

-> checkpoints_phase_A1_vertical_mf_f<offsets>/
"""
import argparse, csv, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import median_filter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.phase_unet import PhaseUNetFPP
from models.calibration_layer_vertical import CalibrationLayerVertical, T_P_VERTICAL
from train_phase_variant import depth_loss_hybrid, compute_depth_from_phase
from train_phase_A1_refined import phase_gradient_loss
from train_phase_A1_refined_v2 import circular_phase_loss

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
MF_ROOT = BASE/"training_datasets/training_data_depth_individual_normalized_multiframe_vertical_only"
PHASE_ROOT = BASE/"training_datasets/phase_targets_vertical"
CAL_DIR = BASE/"calibration_matrices"
K_MAX, BASE_FRAME = 64, 25


class VerticalMultiFrameDataset(Dataset):
    """Multi-channel vertical input; offsets index phase-shifted frames, files are F(25+j)."""

    def __init__(self, fringe_dir, phase_dir, K_max=64, frame_offsets=(0, 6, 12)):
        self.fringe_dir = Path(fringe_dir); self.phase_dir = Path(phase_dir)
        self.K_max = K_max; self.offsets = tuple(frame_offsets)
        self.samples = []
        for p in sorted(self.fringe_dir.glob(f"*_F{BASE_FRAME}.png")):
            stem = p.stem.rsplit("_F", 1)[0]
            npz = self.phase_dir/f"{stem}.npz"
            paths = [self.fringe_dir/f"{stem}_F{BASE_FRAME+j}.png" for j in self.offsets]
            if npz.exists() and all(q.exists() for q in paths):
                self.samples.append((stem, npz, paths))
        if not self.samples:
            raise ValueError(f"No samples in {fringe_dir} for offsets {self.offsets}")
        print(f"VerticalMultiFrameDataset: {len(self.samples)} scenes, "
              f"offsets {self.offsets} -> files F{[BASE_FRAME+j for j in self.offsets]}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        stem, npz_path, paths = self.samples[idx]
        chans = [np.array(Image.open(p).convert('L'), dtype=np.float32)/255.0 for p in paths]
        z = np.load(npz_path)
        k = np.clip(median_filter(z['k'].astype(np.float64), size=7), 0, self.K_max-1).astype(np.int64)
        return {'fringe': torch.from_numpy(np.stack(chans, 0)),
                'sin_phi_gt': torch.from_numpy(z['sin_phi']).unsqueeze(0),
                'cos_phi_gt': torch.from_numpy(z['cos_phi']).unsqueeze(0),
                'k_gt': torch.from_numpy(k),
                'mask_gt': torch.from_numpy(z['mask']).unsqueeze(0),
                'depth_gt': torch.from_numpy(z['depth_mm']).unsqueeze(0),
                'd_min': float(z['d_min']), 'd_max': float(z['d_max']), 'name': stem}


def losses(sin_hat, cos_hat, batch, cal, device, wp, wg, wd, alpha):
    sg = batch['sin_phi_gt'].to(device); cg = batch['cos_phi_gt'].to(device)
    mask = batch['mask_gt'].to(device); dgt = batch['depth_gt'].to(device)
    k = batch['k_gt'].to(device)
    dmin = torch.tensor([b for b in batch['d_min']], device=device, dtype=torch.float32)
    dmax = torch.tensor([b for b in batch['d_max']], device=device, dtype=torch.float32)
    depth, _, _ = compute_depth_from_phase(sin_hat, cos_hat, k.unsqueeze(1), cal, T_p=T_P_VERTICAL)
    Lp = circular_phase_loss(sin_hat, cos_hat, sg, cg, mask, alpha)
    return (wp*Lp + wg*phase_gradient_loss(sin_hat, cos_hat, sg, cg, mask)
            + wd*depth_loss_hybrid(depth, dgt, mask, dmin, dmax)), Lp


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    offs = tuple(int(x) for x in args.frames.split(","))
    tag = "phase_A1_vertical_mf_f" + "-".join(map(str, offs)) + (f"_s{args.seed}" if args.seed != 42 else "")
    ckpt_dir, log_dir = Path(f"checkpoints_{tag}"), Path(f"logs_{tag}")
    if (ckpt_dir/"best_model.pth").exists():
        raise SystemExit(f"REFUSING to overwrite existing {ckpt_dir}/best_model.pth")
    print("="*74); print(f"Vertical multi-frame: offsets {offs}, T_p={T_P_VERTICAL}, K_MAX={K_MAX}")
    print("="*74, flush=True)

    tr = VerticalMultiFrameDataset(MF_ROOT/"train/fringe", PHASE_ROOT/"train", K_MAX, offs)
    va = VerticalMultiFrameDataset(MF_ROOT/"val/fringe", PHASE_ROOT/"val", K_MAX, offs)
    trl = DataLoader(tr, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    val = DataLoader(va, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    model = PhaseUNetFPP(variant='A1', in_channels=len(offs), K_max=K_MAX, dropout_rate=0.0).to(dev)
    cal = CalibrationLayerVertical(str(CAL_DIR)).to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    log_dir.mkdir(parents=True, exist_ok=True)
    csvp = log_dir/"training_log.csv"
    with open(csvp, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_phase', 'val_phase', 'lr', 'seconds'])

    best = float('inf'); wp, wg, wd, alpha = 1.0, 0.5, 0.1, 3.0
    for ep in range(1000):
        t0 = time.time(); model.train(); run = 0.0; nb = 0
        for b in tqdm(trl, desc="Training", leave=False):
            opt.zero_grad()
            o = model(b['fringe'].to(dev))
            L, Lp = losses(o['sin_phi_hat'], o['cos_phi_hat'], b, cal, dev, wp, wg, wd, alpha)
            if torch.isnan(L) or torch.isinf(L): opt.zero_grad(); continue
            L.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            run += Lp.item(); nb += 1
        model.eval(); vs = 0.0; nv = 0
        with torch.no_grad():
            for b in tqdm(val, desc="Validation", leave=False):
                o = model(b['fringe'].to(dev))
                _, Lp = losses(o['sin_phi_hat'], o['cos_phi_hat'], b, cal, dev, wp, wg, wd, alpha)
                vs += Lp.item(); nv += 1
        vp = vs/max(nv, 1); sch.step(vp); lr = opt.param_groups[0]['lr']; dt = time.time()-t0
        with open(csvp, 'a', newline='') as f:
            csv.writer(f).writerow([ep+1, run/max(nb,1), vp, float(lr), int(dt)])
        mk = ""
        if vp < best:
            best = vp; ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'loss': vp,
                        'in_channels': len(offs), 'frame_offsets': offs,
                        'frame_files': [BASE_FRAME+j for j in offs],
                        'T_p': T_P_VERTICAL, 'K_max': K_MAX}, ckpt_dir/"best_model.pth")
            mk = " <- NEW BEST!"
        print(f"[{tag}] Epoch {ep+1:3d} ({dt:5.1f}s) | train {run/max(nb,1):.4f} | "
              f"val ph={vp:.4f} | LR {lr:.2e}{mk}", flush=True)
        if lr <= 1.01e-7: print("LR floor reached, stopping."); break
    print(f"Best val_phase: {best:.6f}\nBest model: {ckpt_dir/'best_model.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=str, default="0,6,12", help="phase-shift frame offsets (files are F25+offset)")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
