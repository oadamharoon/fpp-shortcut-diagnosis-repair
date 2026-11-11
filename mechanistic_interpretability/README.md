# Mechanistic Interpretability Analysis

This directory contains tools for analyzing what UNet and PhicalNet (physics-informed UNet) learn when predicting depth from fringe projection profilometry (FPP) images.

## Hypothesis

**Do neural networks learn shape priors or physical fringe-to-depth relationships?**

We investigate whether depth prediction networks:
1. **Learn shape priors** - Memorize geometric relationships between shape categories and depth patterns
2. **Learn physics** - Actually decode the fringe pattern to extract phase/depth information

## Analysis Methods

### 1. GradCAM Analysis (`gradcam_analysis.py`)
Visualizes what spatial regions the network focuses on when making predictions.

**Hypothesis Test:**
- If the network learns **shape priors**: GradCAM should highlight edges/boundaries
- If the network learns **physics**: GradCAM should highlight fringe pattern variations

```bash
# UNet
python gradcam_analysis.py --pretrained /path/to/unet.pth --num_samples 30

# PhicalNet
python phicalnet/gradcam_analysis.py --pretrained /path/to/phicalnet.pth --modes phase depth
```

### 2. Neural Probing (`probe_networks.py`)
Trains small auxiliary networks on frozen activations to test what information is decodable at each layer.

**Probe Tasks:**
- Shape classification (MLP and linear)
- Phase regression
- Depth-from-phase regression

```bash
# UNet
python probe_networks.py --pretrained /path/to/unet.pth --epochs 50

# PhicalNet
python phicalnet/probing_analysis.py --pretrained /path/to/phicalnet.pth --epochs 30
```

### 3. Mutual Information Estimation (`mutual_information.py`)
Quantifies how much information about shape vs. phase is encoded in layer activations.

**Methods:**
- MINE (Mutual Information Neural Estimation)
- Histogram-based binning estimator

```bash
python mutual_information.py --pretrained /path/to/unet.pth --method both --max_samples 500
```

## Directory Structure

```
mechanistic_interpretability/
├── __init__.py
├── README.md
├── utils.py                    # Shared utilities
├── feature_extractor.py        # UNet feature extraction
├── dataset.py                  # Datasets with shape labels
├── gradcam_analysis.py         # GradCAM for UNet
├── probe_networks.py           # Neural probing for UNet
├── mutual_information.py       # MI estimation for UNet
│
└── phicalnet/                  # PhicalNet analysis
    ├── __init__.py
    ├── feature_extractor.py
    ├── dataset.py
    ├── gradcam_analysis.py
    └── probing_analysis.py
```

## Requirements

```
torch>=1.10
numpy
scipy
matplotlib
scikit-learn
tqdm
pillow
```

## Interpretation Guide

### GradCAM Results
| Ratio (Edge/Fringe) | Interpretation |
|---------------------|----------------|
| > 1.5 | Strong shape prior evidence |
| 1.0 - 1.5 | Moderate shape prior evidence |
| < 1.0 | Weak/no shape prior evidence |

### Probing Results
- **High shape classification accuracy** (>70%) → Shape information well-encoded
- **Low phase regression loss** (<0.1) → Physical phase relationship learned
- **Shape accuracy >> Phase accuracy** → Shape priors dominate

### Mutual Information Results
- **I(act; shape) >> I(act; phase)** → Shape priors dominate
- **High I(act; shape) in early layers** → Shape encoded early
- **Low I(act; phase) throughout** → Physical relationships not learned

## Output Format

All analysis scripts produce:
1. **JSON results file** - Detailed metrics for each layer
2. **Visualization plots** - Summary figures with error bars
3. **Text report** - Interpretation and conclusions

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

## License

[Your license here]
