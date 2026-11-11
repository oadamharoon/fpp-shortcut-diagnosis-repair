import sys
#!/usr/bin/env python3
"""
Mutual Information Estimation for UNet Representations

Estimates mutual information (MI) between layer activations and different
target variables to quantify what information is encoded at each layer.

MI Estimates:
    - I(activation; shape_label): Information about object shape category
    - I(activation; phase_map): Information about physical phase values

Methods:
    1. MINE (Mutual Information Neural Estimation): Neural network-based
    2. Binning-based estimator: Discrete histogram approximation

Reference:
    Belghazi et al., "MINE: Mutual Information Neural Estimation", ICML 2018

Usage:
    python mutual_information.py --pretrained /path/to/model.pth \\
                                  --output_dir ./mi_results

Author: [Your Name]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from feature_extractor import UNetFeatureExtractor, extract_global_features
from dataset import FPPDatasetWithLabels, SHAPE_CATEGORIES
from utils import get_depth_key

import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent))
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet"):
    sys.path.insert(0, str(_p))


# =============================================================================
# MINE Implementation
# =============================================================================

class MINENetwork(nn.Module):
    """Statistics network for MINE."""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)


class MINE:
    """MINE estimator for mutual information with continuous Y."""

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256,
                 lr: float = 1e-4, device: str = 'cuda'):
        self.device = device
        self.network = MINENetwork(x_dim, y_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.ema_alpha = 0.01
        self.ema = None

    def estimate(self, x: torch.Tensor, y: torch.Tensor,
                 num_iterations: int = 500, batch_size: int = 256) -> float:
        x = x.to(self.device)
        y = y.to(self.device)
        n_samples = x.shape[0]

        self.network.train()
        mi_estimates = []

        for _ in range(num_iterations):
            actual_batch_size = min(batch_size, n_samples)
            idx = torch.randperm(n_samples)[:actual_batch_size]
            x_batch = x[idx]
            y_batch = y[idx]

            y_shuffle_idx = torch.randperm(actual_batch_size)
            y_marginal = y_batch[y_shuffle_idx]

            t_joint = self.network(x_batch, y_batch)
            t_marginal = self.network(x_batch, y_marginal)

            joint_term = t_joint.mean()
            exp_marginal = torch.exp(t_marginal)

            if self.ema is None:
                self.ema = exp_marginal.mean().detach()
            else:
                self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * exp_marginal.mean().detach()

            marginal_term = exp_marginal.mean() / self.ema
            loss = -(joint_term - torch.log(marginal_term))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                mi_est = joint_term - torch.log(exp_marginal.mean())
                mi_estimates.append(mi_est.item())

        return np.mean(mi_estimates[-100:])


class MINEWithCategories:
    """MINE estimator for categorical Y (shape labels)."""

    def __init__(self, x_dim: int, num_categories: int, hidden_dim: int = 256,
                 lr: float = 1e-4, device: str = 'cuda'):
        self.device = device
        self.num_categories = num_categories
        self.network = MINENetwork(x_dim, num_categories, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.ema_alpha = 0.01
        self.ema = None

    def estimate(self, x: torch.Tensor, labels: torch.Tensor,
                 num_iterations: int = 500, batch_size: int = 256) -> float:
        x = x.to(self.device)
        labels = labels.to(self.device)

        valid_mask = (labels >= 0) & (labels < self.num_categories)
        x = x[valid_mask]
        labels = labels[valid_mask]

        n_samples = x.shape[0]
        if n_samples < batch_size:
            batch_size = n_samples

        y = F.one_hot(labels, num_classes=self.num_categories).float()

        self.network.train()
        mi_estimates = []

        for _ in range(num_iterations):
            actual_batch_size = min(batch_size, n_samples)
            idx = torch.randperm(n_samples)[:actual_batch_size]
            x_batch = x[idx]
            y_batch = y[idx]

            y_shuffle_idx = torch.randperm(actual_batch_size)
            y_marginal = y_batch[y_shuffle_idx]

            t_joint = self.network(x_batch, y_batch)
            t_marginal = self.network(x_batch, y_marginal)

            joint_term = t_joint.mean()
            exp_marginal = torch.exp(t_marginal)

            if self.ema is None:
                self.ema = exp_marginal.mean().detach()
            else:
                self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * exp_marginal.mean().detach()

            marginal_term = exp_marginal.mean() / self.ema
            loss = -(joint_term - torch.log(marginal_term))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                mi_est = joint_term - torch.log(exp_marginal.mean())
                mi_estimates.append(mi_est.item())

        return np.mean(mi_estimates[-100:])


# =============================================================================
# Binning-based MI Estimation
# =============================================================================

def estimate_mi_binning(x: np.ndarray, y: np.ndarray, n_bins: int = 30) -> float:
    """Estimate MI using histogram-based binning."""
    if x.shape[1] > 10:
        pca = PCA(n_components=10)
        x = pca.fit_transform(x)

    x_discrete = np.zeros(x.shape[0], dtype=np.int64)
    for d in range(x.shape[1]):
        edges = np.histogram_bin_edges(x[:, d], bins=n_bins)
        x_d = np.digitize(x[:, d], edges[:-1])
        x_discrete = x_discrete * n_bins + x_d

    n_samples = len(y)
    joint_counts = {}
    x_counts = {}
    y_counts = {}

    for xi, yi in zip(x_discrete, y):
        key = (xi, yi)
        joint_counts[key] = joint_counts.get(key, 0) + 1
        x_counts[xi] = x_counts.get(xi, 0) + 1
        y_counts[yi] = y_counts.get(yi, 0) + 1

    mi = 0.0
    for (xi, yi), count in joint_counts.items():
        p_xy = count / n_samples
        p_x = x_counts[xi] / n_samples
        p_y = y_counts[yi] / n_samples
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))

    return mi


# =============================================================================
# Feature Collection
# =============================================================================

def collect_features_and_targets(
    feature_extractor: UNetFeatureExtractor,
    dataloader: DataLoader,
    device: str = 'cuda',
    max_samples: int = 500
) -> Dict[str, torch.Tensor]:
    """Collect pooled features and targets from the dataset."""
    feature_extractor.eval()
    layer_info = feature_extractor.get_layer_info()

    collected = {'shape_labels': [], 'phase_flat': []}
    for layer_name in layer_info.keys():
        collected[f'features_{layer_name}'] = []

    n_collected = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting features"):
            if n_collected >= max_samples:
                break

            fringe = batch['fringe'].to(device)
            phase = batch['phase']
            labels = batch['shape_label']

            activations = feature_extractor(fringe)
            global_features = extract_global_features(activations)

            for layer_name in layer_info.keys():
                collected[f'features_{layer_name}'].append(global_features[layer_name].cpu())

            collected['shape_labels'].append(labels)
            phase_flat = F.adaptive_avg_pool2d(phase, (30, 30)).view(phase.shape[0], -1)
            collected['phase_flat'].append(phase_flat)

            n_collected += fringe.shape[0]

    result = {}
    for key, values in collected.items():
        result[key] = torch.cat(values, dim=0)[:max_samples]

    return result


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_mi_estimation(
    pretrained_path: str,
    data_root: str,
    output_dir: str,
    dataset_type: str = "_raw",
    method: str = 'both',
    max_samples: int = 500,
    mine_iterations: int = 500,
    device: str = 'cuda',
    layers_to_analyze: Optional[List[str]] = None
) -> Dict:
    """Run MI estimation experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading pretrained UNet from {pretrained_path}")
    feature_extractor = UNetFeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor.to(device)

    depth_key = get_depth_key(dataset_type)
    base_path = Path(data_root) / f"training_data_depth{dataset_type}"

    dataset = FPPDatasetWithLabels(
        fringe_dir=str(base_path / "train" / "fringe"),
        depth_dir=str(base_path / "train" / "depth"),
        depth_key=depth_key
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"\nCollecting features (max {max_samples} samples)...")
    data = collect_features_and_targets(feature_extractor, dataloader, device, max_samples)

    layer_info = feature_extractor.get_layer_info()
    if layers_to_analyze is None:
        layers_to_analyze = list(layer_info.keys())

    results = {
        'experiment_info': {
            'pretrained_path': pretrained_path,
            'method': method,
            'max_samples': max_samples,
            'timestamp': datetime.now().isoformat(),
            'layers_analyzed': layers_to_analyze
        },
        'mi_shape': {},
        'mi_phase': {},
    }

    num_classes = len(SHAPE_CATEGORIES)
    phase_dim = data['phase_flat'].shape[1]

    for layer_name in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"Estimating MI for layer: {layer_name}")
        print('='*60)

        features = data[f'features_{layer_name}']
        feature_dim = features.shape[1]
        labels = data['shape_labels']
        phase_flat = data['phase_flat']

        if method in ['mine', 'both']:
            print("\nMINE estimation...")

            print("  I(activation; shape) ...")
            mine_shape = MINEWithCategories(feature_dim, num_classes, device=device)
            mi_shape = mine_shape.estimate(features, labels, num_iterations=mine_iterations)
            results['mi_shape'][f'{layer_name}_mine'] = float(mi_shape)
            print(f"    I(act; shape) = {mi_shape:.4f} nats")

            print("  I(activation; phase) ...")
            phase_np = phase_flat.numpy()
            pca = PCA(n_components=min(50, phase_dim))
            phase_reduced = torch.from_numpy(pca.fit_transform(phase_np)).float()

            mine_phase = MINE(feature_dim, phase_reduced.shape[1], device=device)
            mi_phase = mine_phase.estimate(features, phase_reduced, num_iterations=mine_iterations)
            results['mi_phase'][f'{layer_name}_mine'] = float(mi_phase)
            print(f"    I(act; phase) = {mi_phase:.4f} nats")

        if method in ['binning', 'both']:
            print("\nBinning estimation...")

            features_np = features.numpy()
            labels_np = labels.numpy()

            valid_mask = labels_np >= 0
            features_np = features_np[valid_mask]
            labels_np = labels_np[valid_mask]
            phase_np = phase_flat.numpy()[valid_mask]

            print("  I(activation; shape) ...")
            mi_shape_bin = estimate_mi_binning(features_np, labels_np, n_bins=20)
            results['mi_shape'][f'{layer_name}_binning'] = float(mi_shape_bin)
            print(f"    I(act; shape) = {mi_shape_bin:.4f} nats")

            print("  I(activation; phase) ...")
            pca = PCA(n_components=min(10, phase_dim))
            phase_reduced = pca.fit_transform(phase_np)
            kmeans = MiniBatchKMeans(n_clusters=50, random_state=42, n_init=3)
            phase_discrete = kmeans.fit_predict(phase_reduced)

            mi_phase_bin = estimate_mi_binning(features_np, phase_discrete, n_bins=20)
            results['mi_phase'][f'{layer_name}_binning'] = float(mi_phase_bin)
            print(f"    I(act; phase) = {mi_phase_bin:.4f} nats")

    results_path = output_dir / 'mi_estimation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Mutual Information Estimation for UNet")

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(_ROOT))
    parser.add_argument("--output_dir", type=str, default="mi_results")
    parser.add_argument("--dataset_type", type=str, default="_raw")
    parser.add_argument("--method", type=str, default="both", choices=["mine", "binning", "both"])
    parser.add_argument("--max_samples", type=int, default=240)
    parser.add_argument("--mine_iterations", type=int, default=500)
    parser.add_argument("--layers", type=str, nargs='+', default=None)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_mi_estimation(
        pretrained_path=args.pretrained,
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        method=args.method,
        max_samples=args.max_samples,
        mine_iterations=args.mine_iterations,
        device=device,
        layers_to_analyze=args.layers
    )


if __name__ == "__main__":
    main()
