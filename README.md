# Shape-Prior Shortcuts in Single-Shot Fringe Projection Profilometry

Code for the two-part study of single-shot deep learning in long-range fringe projection
profilometry (FPP):

- **Part I — Diagnosing shape-prior shortcuts.** A depth-regressing network trained on single
  fringe images reconstructs by detecting object boundaries and filling with learned shape
  templates rather than by demodulating phase.
- **Part II — Repairing shape-prior shortcuts (PhiCalNet).** Routing reconstruction through
  predicted wrapped phase and a fixed differentiable calibration layer removes the shape-prior
  solution from the hypothesis space by construction, cutting single-shot object MAE from
  14.54 mm to 4.46 mm on the same data, optimizer and schedule.

Preprints are on arXiv. See [Citing](#citing).

## Contents

| Path | What it is |
|---|---|
| `models/` | PhiCalNet (`phase_unet.py`), the differentiable `calibration_layer.py`, and the architectural variants used in the ablations |
| `data/` | Dataset readers for the single-frame and multi-frame configurations |
| `UNet/`, `TransUNet/`, `Hformer/`, `ResUNet/` | Depth-regressing baselines compared in Part I |
| `scripts/` | Phase-target precomputation and calibration validation |
| `calibration_matrices/` | Camera and projector intrinsics/extrinsics for the benchmark rig |
| `configs/` | Architecture configs |
| `train_*.py`, `eval_*.py`, `evaluate_*.py` | Training and evaluation entry points |
| `conformal_snapshot*.py` | Snapshot-ensemble conformal uncertainty quantification |
| `mechanistic_interpretability/` | GradCAM, linear probing and mutual-information analyses (Part I, UNet) and `phicalnet/` (Part II, dual-mode) |

## Setup

```bash
git clone https://github.com/oadamharoon/fpp-shortcut-diagnosis-repair.git
cd fpp-shortcut-diagnosis-repair
pip install -r requirements.txt
```

### Data

The FPP-ML-Bench dataset is published separately:

```bash
# https://huggingface.co/datasets/aharoon/fpp-ml-bench
huggingface-cli download aharoon/fpp-ml-bench --repo-type dataset --local-dir ./
```

Scripts resolve data relative to `FPPML_ROOT`, which defaults to the repository root. Point it
elsewhere if you keep the dataset outside the checkout:

```bash
export FPPML_ROOT=/path/to/data
```

The expected layout under `$FPPML_ROOT` is:

```
training_datasets/
  training_data_depth_individual_normalized/{train,val,test}/{fringe,depth,mask}
  phase_targets/{train,val,test}
calibration_matrices/          # shipped in this repo
```

Phase targets are derived from the dataset and the calibration, and are regenerated with:

```bash
python scripts/precompute_phase_targets.py
```

### Trained weights

Trained checkpoints are not distributed. Every result below is reproduced by running the
training script from scratch on the published dataset; `--seed 42` reproduces the original
runs. Scripts that take a `--pretrained` argument expect a checkpoint you have trained.

## Reproducing the published results

Set `FPPML_ROOT` and download the dataset first. The headline numbers:

| Result | Paper | Command |
|---|---|---|
| UNet depth-regression baseline, 14.54 mm object MAE | I | `python UNet/train.py` |
| Architecture comparison (UNet / TransUNet / Hformer / ResUNet) | I | `python <Arch>/train.py`, then `python eval_arch_headtohead.py` |
| Flat-plane out-of-distribution test | I | `python make_flat_plane_depth.py` |
| **PhiCalNet, 4.46 mm object MAE** | II | `python train_phase_A1_refined_v2.py` |
| PhiCalNet evaluation and qualitative figures | II | `python regen_phaseunet_viz.py` |
| Loss-term ablation | II | `python train_phicalnet_ablation.py`, then `python eval_phicalnet_ablation.py` |
| PINN baseline, lambda sweep | II | `python train_pinn_unet.py --lambda_physics <value>` |
| Learned fringe-order variants (negative result) | II | `python train_phicalnet_learnedk.py` |
| Fringe-order sensitivity sweep | II | `python diag_learnedk_and_noisy_k.py` |
| Multi-frame configurations, 3-frame 1.16 mm | II | `python train_phase_A1_multiframe.py --frames 0,6,12`, then `python eval_multiframe_grid.py` |
| Wrap-boundary residual analysis | II | `python evaluate_phase_A1_v2_outliers.py`, `python probe_wrap_correction.py` |
| Snapshot-conformal uncertainty quantification | II | `python conformal_snapshot.py` (PhiCalNet), `python conformal_snapshot_unet.py` (UNet) |
| Linear probing, edge vs depth decodability | I | `python mechanistic_interpretability/probe_networks.py --pretrained <unet.pth>` |
| Mutual information between activations and targets | I | `python mechanistic_interpretability/mutual_information.py --pretrained <unet.pth>` |
| GradCAM edge/fringe attention ratio | I | `python mechanistic_interpretability/gradcam_analysis.py --pretrained <unet.pth>` |
| Spatial probes at the nine PhiCalNet layers | II | `python mechanistic_interpretability/phicalnet/probing_analysis.py --pretrained <phicalnet.pth>` |
| Dual-mode GradCAM (phase vs depth gradient target) | II | `python mechanistic_interpretability/phicalnet/gradcam_analysis.py --pretrained <phicalnet.pth> --modes phase depth` |

PhiCalNet as published is `PhaseUNetFPP(variant='A1')` predicting `(sin phi, cos phi)`, with
fringe order supplied as declared auxiliary side information and depth produced by the fixed
differentiable calibration layer. Its objective is

```
L = 1.0 * L_circular + 0.5 * L_gradient + 0.1 * L_depth
```

where `L_circular` is the boundary-weighted geodesic phase loss (primary) and `L_depth` is an
auxiliary hybrid-L1 term on per-sample normalized depth.

## Not included

- **Trained checkpoints.** See above; results are reproduced by training.
- **Follow-on work.** Experiments belonging to related manuscripts still under review are
  omitted.

## Citing

```bibtex
@article{haroon2026diagnosis,
  title   = {Diagnosing Shape-Prior Shortcuts in Long-Range Single-Shot Fringe
             Projection Profilometry},
  author  = {Haroon, Adam and Lakshman, Anush and Fleming, Cody and Li, Beiwen},
  journal = {arXiv preprint arXiv:2606.17093},
  year    = {2026}
}

@article{haroon2026repairing,
  title   = {Repairing Shape-Prior Shortcuts in Long-Range Single-Shot Fringe
             Projection Profilometry},
  author  = {Haroon, Adam and Fleming, Cody and Li, Beiwen},
  journal = {arXiv preprint arXiv:2607.11928},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
