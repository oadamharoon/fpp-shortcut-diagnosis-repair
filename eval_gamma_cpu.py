import sys, os, torch
sys.path.insert(0, '.')
from pathlib import Path
from data.phase_dataset_multiframe import PhaseMultiFrameDataset
from models.calibration_layer import CalibrationLayer
from models.phase_unet import PhaseUNetFPP
from eval_multiframe_grid import _per_sample_errs_oracle_phase_unet, error_summary, PHASE_ROOT, CAL_DIR, K_MAX

import os as _os
BASE = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent))
GROOT = BASE / "training_datasets/multiframe_horizontal_gamma2.2"
device = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
torch.set_grad_enabled(False)
cal = CalibrationLayer(str(CAL_DIR)).to(device)
for frames, tag in [((0,9), "gamma_f0-9"), ((0,6), "gamma_f0-6")]:
    ck = f"checkpoints_phase_A1_multiframe_f{frames[0]}-{frames[1]}_gamma2.2/best_model.pth"
    m = PhaseUNetFPP(variant="A1", in_channels=2, K_max=K_MAX, dropout_rate=0.0)
    m.load_state_dict(torch.load(ck, map_location=device)["model_state_dict"])
    m = m.to(device).eval()
    ds = PhaseMultiFrameDataset(GROOT/"test/fringe", PHASE_ROOT/"test", K_max=K_MAX, frame_indices=frames)
    r = error_summary(_per_sample_errs_oracle_phase_unet(m, device, ds, cal, tag), tag)
    print(f"RESULT {tag}: obj MAE {r['object_mae']:.2f} | obj RMSE {r['object_rmse']:.2f} | "
          f">100mm {dict(r['thrs'])[100]} ({100*dict(r['thrs'])[100]/r['n']:.3f}%)", flush=True)
print("DONE")
