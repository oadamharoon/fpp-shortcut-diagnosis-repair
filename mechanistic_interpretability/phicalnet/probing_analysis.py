import sys
#!/usr/bin/env python3
"""
Linear Probing Analysis for PhicalNet (PhaseUNetFPP)

Probes 4 spatial targets at each layer:
    1. edges    - geometric structure
    2. depth    - normalized GT depth
    3. sin_phi  - GT sine of wrapped phase
    4. cos_phi  - GT cosine of wrapped phase

Usage:
    python probing_analysis.py --pretrained /path/to/model.pth \\
                               --output_dir ./phicalnet_probing_results

Author: [Your Name]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from feature_extractor import PhicalNetFeatureExtractor
from dataset import PhicalNetDataset

import os as _os
_ROOT = Path(_os.environ.get("FPPML_ROOT", Path(__file__).resolve().parent.parent.parent))
_REPO = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO, _REPO / "models", _REPO / "UNet", _REPO / "mechanistic_interpretability"):
    sys.path.insert(0, str(_p))


class SpatialProbe(nn.Module):
    """Probe network: 4 conv layers predicting a spatial map."""

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


def train_spatial_probe(
    feature_extractor: PhicalNetFeatureExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_name: str,
    target_key: str,
    num_epochs: int = 30,
    device: str = 'cuda',
    output_size: Tuple[int, int] = (120, 120)
) -> Dict:
    """Train a probe to predict a spatial target from frozen activations."""
    feature_extractor.eval()
    layer_info = feature_extractor.get_layer_info()
    input_channels = layer_info[layer_name][0]

    probe = SpatialProbe(input_channels, output_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

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
            loss = F.mse_loss(pred, target)
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
                loss = F.mse_loss(pred, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    return {
        'layer': layer_name,
        'target': target_key,
        'best_val_loss': best_val_loss,
        'history': history
    }


def run_probing_analysis(
    pretrained_path: str,
    data_root: str,
    phase_root: str,
    output_dir: str,
    num_epochs: int = 30,
    batch_size: int = 4,
    device: str = 'cuda',
    layers_to_analyze: Optional[List[str]] = None
) -> Dict:
    """Run linear probing analysis for PhicalNet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading PhicalNet from {pretrained_path}")
    feature_extractor = PhicalNetFeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor.to(device)

    base_path = Path(data_root) / "training_data_depth_individual_normalized"
    phase_base = Path(phase_root)

    print(f"\nLoading data from {base_path}")
    full_dataset = PhicalNetDataset(
        fringe_dir=str(base_path / "train" / "fringe"),
        depth_dir=str(base_path / "train" / "depth"),
        phase_dir=str(phase_base / "train"),
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size} train, {val_size} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if layers_to_analyze is None:
        layers_to_analyze = ['skip1', 'skip2', 'skip3', 'skip4', 'bottleneck',
                            'up1', 'up2', 'up3', 'up4']

    results = {
        'experiment_info': {
            'pretrained_path': pretrained_path,
            'num_epochs': num_epochs,
            'timestamp': datetime.now().isoformat(),
            'layers_analyzed': layers_to_analyze,
            'model': 'PhicalNet (PhaseUNetFPP A1)',
        },
        'probe_results': {},
    }

    targets = ['edges', 'depth', 'sin_phi', 'cos_phi']
    target_labels = {
        'edges': 'Geometry (Edges)',
        'depth': 'Depth Values',
        'sin_phi': 'Phase sin(phi)',
        'cos_phi': 'Phase cos(phi)',
    }

    print("\n" + "=" * 70)
    print("LINEAR PROBING - PHICALNET")
    print("=" * 70)

    for layer in layers_to_analyze:
        print(f"\n--- Layer: {layer} ---")
        results['probe_results'][layer] = {}

        for target in targets:
            print(f"\n  Training probe for: {target_labels[target]}")
            probe_result = train_spatial_probe(
                feature_extractor, train_loader, val_loader,
                layer_name=layer, target_key=target,
                num_epochs=num_epochs, device=device
            )
            results['probe_results'][layer][target] = probe_result

    results_path = output_dir / 'phicalnet_probing_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY: PROBE VALIDATION LOSS (lower = better)")
    print("=" * 90)
    print(f"{'Layer':<12} {'Edges':<12} {'Depth':<12} {'sin_phi':<12} {'cos_phi':<12}")
    print("-" * 90)

    for layer in layers_to_analyze:
        values = [results['probe_results'][layer][t]['best_val_loss'] for t in targets]
        print(f"{layer:<12} {values[0]:<12.6f} {values[1]:<12.6f} {values[2]:<12.6f} {values[3]:<12.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PhicalNet Probing Analysis")

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=str(_ROOT))
    parser.add_argument("--phase_root", type=str, default=str(_ROOT / "training_datasets" / "phase_targets"))
    parser.add_argument("--output_dir", type=str, default="phicalnet_probing_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layers", type=str, nargs='+', default=None)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_probing_analysis(
        pretrained_path=args.pretrained,
        data_root=args.data_root,
        phase_root=args.phase_root,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        layers_to_analyze=args.layers,
    )


if __name__ == "__main__":
    main()
