import sys
#!/usr/bin/env python3
"""
Neural Probing Analysis for UNet Intermediate Representations

Trains small auxiliary networks on frozen UNet activations to test what
information is linearly decodable at each layer.

Probe Tasks:
    1. Shape Classification (MLP): Tests if shape categories are decodable
    2. Shape Classification (Linear): Tests linear separability of shapes
    3. Phase Regression: Tests if physical phase relationship is encoded
    4. Depth-from-Phase Regression: Tests if physical depth relationship is learned

Usage:
    python probe_networks.py --pretrained /path/to/model.pth \\
                             --output_dir ./probing_results

Author: [Your Name]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from feature_extractor import UNetFeatureExtractor, extract_global_features
from dataset import FPPDatasetWithLabels, SHAPE_CATEGORIES
from utils import get_depth_key, convert_to_serializable

import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent))
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet"):
    sys.path.insert(0, str(_p))


# =============================================================================
# Probe Network Architectures
# =============================================================================

class ShapeClassifierProbe(nn.Module):
    """MLP classifier for shape category prediction."""

    def __init__(self, input_dim: int, num_classes: int = 40, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearClassifierProbe(nn.Module):
    """Simple linear probe for classification."""

    def __init__(self, input_dim: int, num_classes: int = 40):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RegressionProbe(nn.Module):
    """CNN-based probe for predicting spatial maps."""

    def __init__(self, input_channels: int, output_size: Tuple[int, int] = (120, 120)):
        super().__init__()
        self.output_size = output_size
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)


# =============================================================================
# Training Functions
# =============================================================================

def train_shape_classifier(
    feature_extractor: UNetFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_name: str,
    num_epochs: int = 50,
    device: str = 'cuda',
    probe_type: str = 'mlp'
) -> Dict:
    """Train a shape classifier probe on a specific layer's activations."""
    feature_extractor.eval()
    layer_info = feature_extractor.get_layer_info()
    input_dim = layer_info[layer_name][0]
    num_classes = len(SHAPE_CATEGORIES)

    if probe_type == 'linear':
        probe = LinearClassifierProbe(input_dim, num_classes).to(device)
    else:
        probe = ShapeClassifierProbe(input_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        probe.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            fringe = batch['fringe'].to(device)
            labels = batch['shape_label'].to(device)

            valid_mask = labels >= 0
            if not valid_mask.any():
                continue

            with torch.no_grad():
                activations = feature_extractor(fringe)
                global_features = extract_global_features(activations)
                features = global_features[layer_name]

            features = features[valid_mask]
            labels = labels[valid_mask]

            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / max(train_total, 1)

        probe.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                fringe = batch['fringe'].to(device)
                labels = batch['shape_label'].to(device)

                valid_mask = labels >= 0
                if not valid_mask.any():
                    continue

                activations = feature_extractor(fringe)
                global_features = extract_global_features(activations)
                features = global_features[layer_name][valid_mask]
                labels = labels[valid_mask]

                logits = probe(features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        scheduler.step(val_acc)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")

    return {
        'layer': layer_name,
        'probe_type': probe_type,
        'best_val_acc': best_val_acc,
        'history': history
    }


def train_regression_probe(
    feature_extractor: UNetFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_name: str,
    target_key: str,
    num_epochs: int = 50,
    device: str = 'cuda',
    output_size: Tuple[int, int] = (120, 120)
) -> Dict:
    """Train a regression probe to predict spatial maps from activations."""
    feature_extractor.eval()
    layer_info = feature_extractor.get_layer_info()
    input_channels = layer_info[layer_name][0]

    probe = RegressionProbe(input_channels, output_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        probe.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            fringe = batch['fringe'].to(device)
            target = batch[target_key].to(device)
            target = F.interpolate(target, size=output_size, mode='bilinear', align_corners=False)

            with torch.no_grad():
                activations = feature_extractor(fringe)
                features = activations[layer_name]
                if features.shape[2:] != output_size:
                    features = F.interpolate(features, size=output_size, mode='bilinear', align_corners=False)

            optimizer.zero_grad()
            pred = probe(features)

            target_norm = (target - target.mean()) / (target.std() + 1e-8)
            pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
            loss = F.mse_loss(pred_norm, target_norm)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        probe.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                fringe = batch['fringe'].to(device)
                target = batch[target_key].to(device)
                target = F.interpolate(target, size=output_size, mode='bilinear', align_corners=False)

                activations = feature_extractor(fringe)
                features = activations[layer_name]
                if features.shape[2:] != output_size:
                    features = F.interpolate(features, size=output_size, mode='bilinear', align_corners=False)

                pred = probe(features)
                target_norm = (target - target.mean()) / (target.std() + 1e-8)
                pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
                loss = F.mse_loss(pred_norm, target_norm)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"  Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    return {
        'layer': layer_name,
        'target': target_key,
        'best_val_loss': best_val_loss,
        'history': history
    }


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_probing_experiment(
    pretrained_path: str,
    data_root: str,
    output_dir: str,
    dataset_type: str = "_raw",
    num_epochs: int = 50,
    batch_size: int = 4,
    device: str = 'cuda',
    layers_to_probe: Optional[List[str]] = None
) -> Dict:
    """Run full probing experiment across all layers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading pretrained UNet from {pretrained_path}")
    feature_extractor = UNetFeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor.to(device)

    depth_key = get_depth_key(dataset_type)
    base_path = Path(data_root) / f"training_data_depth{dataset_type}"

    print(f"\nLoading data from {base_path}")
    full_train_dataset = FPPDatasetWithLabels(
        fringe_dir=str(base_path / "train" / "fringe"),
        depth_dir=str(base_path / "train" / "depth"),
        depth_key=depth_key
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if layers_to_probe is None:
        layers_to_probe = ['skip1', 'skip2', 'skip3', 'skip4', 'bottleneck',
                          'up1', 'up2', 'up3', 'up4']

    results = {
        'experiment_info': {
            'pretrained_path': pretrained_path,
            'timestamp': datetime.now().isoformat(),
            'layers_probed': layers_to_probe
        },
        'shape_classification': {},
        'phase_regression': {},
        'phase_depth_regression': {}
    }

    for layer in layers_to_probe:
        print(f"\n{'='*60}")
        print(f"Probing layer: {layer}")
        print('='*60)

        print(f"\n[1/4] Shape Classification (MLP)")
        results['shape_classification'][f'{layer}_mlp'] = train_shape_classifier(
            feature_extractor, train_loader, val_loader,
            layer_name=layer, num_epochs=num_epochs, device=device, probe_type='mlp'
        )

        print(f"\n[2/4] Shape Classification (Linear)")
        results['shape_classification'][f'{layer}_linear'] = train_shape_classifier(
            feature_extractor, train_loader, val_loader,
            layer_name=layer, num_epochs=num_epochs, device=device, probe_type='linear'
        )

        print(f"\n[3/4] Phase Regression")
        results['phase_regression'][layer] = train_regression_probe(
            feature_extractor, train_loader, val_loader,
            layer_name=layer, target_key='phase', num_epochs=num_epochs, device=device
        )

        print(f"\n[4/4] Depth-from-Phase Regression")
        results['phase_depth_regression'][layer] = train_regression_probe(
            feature_extractor, train_loader, val_loader,
            layer_name=layer, target_key='phase_depth', num_epochs=num_epochs, device=device
        )

    results_path = output_dir / 'probing_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Neural Probing Analysis for UNet")

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(_ROOT))
    parser.add_argument("--output_dir", type=str, default="probing_results")
    parser.add_argument("--dataset_type", type=str, default="_raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layers", type=str, nargs='+', default=None)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_probing_experiment(
        pretrained_path=args.pretrained,
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        layers_to_probe=args.layers
    )


if __name__ == "__main__":
    main()
