"""
Regime sweep experiment (Stage A: exploratory).

Sweeps width × noise_rate × epochs to find regimes with substantial memorization.
Outputs a phase diagram of memorization fraction vs. capacity/training time.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json
import itertools
from tqdm import tqdm
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from data.corruption import corrupt_labels_random, CorruptionProvenance
from analysis.memorization import (
    extract_per_example_metrics,
    aggregate_temporal_metrics,
    save_temporal_artifacts,
    compute_memorization_onset,
)


def train_and_evaluate(config, hidden_dim, noise_rate, epochs, init_seed, corruption_seed, loader_seed, device):
    """Train a single model and return memorization metrics."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(init_seed)

    # Model
    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=hidden_dim,
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['lr'],
                           weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    provenance = None
    if noise_rate > 0:
        train_dataset, provenance = corrupt_labels_random(train_dataset, noise_rate, corruption_seed)

    torch.manual_seed(loader_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    # Eval loader (fixed order for per-example tracking)
    eval_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    checkpoint_results = []

    best_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Per-example metrics
        epoch_metrics = extract_per_example_metrics(
            model, eval_loader, provenance, device, criterion)
        checkpoint_results.append(epoch_metrics)

        if test_acc > best_acc:
            best_acc = test_acc

    # Aggregate temporal metrics
    if checkpoint_results and provenance:
        aggregated = aggregate_temporal_metrics(checkpoint_results, provenance)

        # Compute memorization fraction
        n_changed = len(provenance.changed_indices)
        n_memorized = aggregated['n_memorized']
        mem_frac = n_memorized / max(n_changed, 1)

        return {
            'hidden_dim': hidden_dim,
            'noise_rate': noise_rate,
            'epochs': epochs,
            'init_seed': init_seed,
            'corruption_seed': corruption_seed,
            'loader_seed': loader_seed,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_test_acc': best_acc,
            'n_changed': n_changed,
            'n_memorized': n_memorized,
            'memorization_fraction': mem_frac,
            'n_forgotten': aggregated['n_forgotten'],
            'mean_csl': float(aggregated['csl'].mean()),
            'mean_forgetting': float(aggregated['forgetting'].mean()),
        }
    else:
        return {
            'hidden_dim': hidden_dim,
            'noise_rate': noise_rate,
            'epochs': epochs,
            'init_seed': init_seed,
            'corruption_seed': corruption_seed,
            'loader_seed': loader_seed,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_test_acc': best_acc,
            'n_changed': 0,
            'n_memorized': 0,
            'memorization_fraction': 0.0,
            'n_forgotten': 0,
            'mean_csl': 0.0,
            'mean_forgetting': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Regime sweep: width × noise × epochs')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--output-dir', type=str, default='outputs/regime_sweep')
    parser.add_argument('--seeds-per-config', type=int, default=3)
    parser.add_argument('--quick', action='store_true', help='Run reduced sweep for testing')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Get sweep parameters from config
    sweep_cfg = config.get('regime_sweep', {})
    hidden_dims = sweep_cfg.get('hidden_dims', [16, 32, 64, 128, 256, 512])
    noise_rates = sweep_cfg.get('noise_rates', [0.0, 0.1, 0.2, 0.4, 0.6])
    epochs_list = sweep_cfg.get('epochs_list', [20, 50, 100, 200])

    if args.quick:
        hidden_dims = [16, 64, 256]
        noise_rates = [0.1, 0.2, 0.4]
        epochs_list = [20, 100]

    seeds_per_config = args.seeds_per_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    configs = list(itertools.product(hidden_dims, noise_rates, epochs_list))
    print(f"Total configurations: {len(configs)}")
    print(f"Seeds per config: {seeds_per_config}")
    print(f"Total runs: {len(configs) * seeds_per_config}")

    all_results = []

    for i, (hidden_dim, noise_rate, epochs) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] hidden_dim={hidden_dim}, noise_rate={noise_rate}, epochs={epochs}")

        for seed_idx in range(seeds_per_config):
            init_seed = config['initialization_seeds'][seed_idx]
            corruption_seed = config['corruption_seeds'][seed_idx]
            loader_seed = config['loader_seeds'][seed_idx]

            result = train_and_evaluate(
                config, hidden_dim, noise_rate, epochs,
                init_seed, corruption_seed, loader_seed, device
            )
            all_results.append(result)

            print(f"  Seed {seed_idx}: test_acc={result['test_acc']:.2f}%, "
                  f"mem_frac={result['memorization_fraction']:.4f} "
                  f"({result['n_memorized']}/{result['n_changed']})")

    # Save results
    with open(output_dir / 'regime_sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n=== REGIME SWEEP SUMMARY ===")
    print(f"{'hidden':>6} {'noise':>5} {'epochs':>6} {'test_acc':>8} {'mem_frac':>10} {'n_mem':>6}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['hidden_dim']:>6} {r['noise_rate']:>5} {r['epochs']:>6} "
              f"{r['test_acc']:>8.2f} {r['memorization_fraction']:>10.4f} {r['n_memorized']:>6}")

    # Find promising regimes
    print("\n=== PROMISING REGIMES (memorization_fraction > 0.1) ===")
    promising = [r for r in all_results if r['memorization_fraction'] > 0.1]
    for r in sorted(promising, key=lambda x: -x['memorization_fraction']):
        print(f"  h={r['hidden_dim']:3d} noise={r['noise_rate']:.1f} epochs={r['epochs']:3d} "
              f"mem={r['memorization_fraction']:.3f} test_acc={r['test_acc']:.2f}%")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()