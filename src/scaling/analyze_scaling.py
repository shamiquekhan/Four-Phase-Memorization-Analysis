"""
Scaling analysis: runs all 4 phases across hidden sizes.
Tracks σ, monosemanticity, circuit sparsity, and accuracy vs width.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.metrics import (
    compute_davies_bouldin, compute_calinski_harabasz, compute_monosemanticity,
    compute_circuit_sparsity, extract_hidden_activations, compute_sigma_and_fdr
)
from utils.stats import compute_ci, SEEDS


def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def get_test_loader(batch_size=500, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_scaling_analysis(results_dir, hidden_dims, seeds, device='cpu'):
    """Run all 4 phases of metrics across hidden sizes."""
    import torch
    test_loader = get_test_loader()
    results = {}

    for h in hidden_dims:
        print(f"\n=== Analyzing hidden size: {h} ===")
        results[h] = {
            'davies_bouldin': [], 'calinski_harabasz': [], 'mono_fraction': [],
            'avg_max_corr': [], 'circuit_size': [], 'sparsity': [], 'accuracy': [],
            'sigma': [], 'fdr': []
        }

        for seed in seeds:
            model_path = Path(results_dir) / f"hidden_{h}" / f"seed_{seed}" / 'final_model.pt'
            if not model_path.exists():
                print(f"  Model not found: {model_path}")
                continue

            model = MNISTNet(784, h, 10).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Phase 1: Geometric (Davies-Bouldin index - dimension-invariant)
            hidden_acts, labels = extract_hidden_activations(model, test_loader, device)
            db = compute_davies_bouldin(hidden_acts, labels)
            ch = compute_calinski_harabasz(hidden_acts, labels)
            results[h]['davies_bouldin'].append(db)
            results[h]['calinski_harabasz'].append(ch)

            # FDR and sigma (dimension-invariant separability + legacy)
            sf = compute_sigma_and_fdr(hidden_acts, labels)
            results[h]['sigma'].append(sf['sigma'])
            results[h]['fdr'].append(sf['fdr'])

            # Phase 2: Representational (monosemanticity)
            mono, max_corrs = compute_monosemanticity(hidden_acts, labels)
            results[h]['mono_fraction'].append(mono)
            results[h]['avg_max_corr'].append(float(np.mean(max_corrs)))

            # Phase 3: Circuit sparsity
            cs, sp, _ = compute_circuit_sparsity(model, test_loader, device)
            results[h]['circuit_size'].append(cs)
            results[h]['sparsity'].append(sp)

            # Accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            results[h]['accuracy'].append(100. * correct / total)

            print(f"  seed={seed}: DB={db:.4f}, mono={mono:.3f}, "
                  f"circuit={cs:.1f}, sparsity={sp:.3f}, acc={results[h]['accuracy'][-1]:.2f}%")

        # Aggregate
        for metric in ['davies_bouldin', 'calinski_harabasz', 'mono_fraction', 'avg_max_corr', 'circuit_size', 'sparsity', 'accuracy', 'sigma', 'fdr']:
            vals = results[h][metric]
            if vals:
                mean, ci_low, ci_high = compute_ci(vals)
                results[h][f'{metric}_mean'] = mean
                results[h][f'{metric}_ci'] = (ci_high - mean)

    return results


def plot_scaling_results(results, output_path):
    """Generate scaling analysis figure (FDR, monosemanticity, circuit size, sparsity vs hidden dim)."""
    hidden_dims = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_config = [
        ('fdr', 'Fisher Discriminant Ratio', axes[0, 0], 'higher = better separation'),
        ('mono_fraction', 'Monosemantic Fraction', axes[0, 1], 'higher is better'),
        ('circuit_size', 'Mean Circuit Size (neurons/class)', axes[1, 0], 'smaller = sparser'),
        ('sparsity', 'Network Sparsity', axes[1, 1], 'higher = sparser'),
    ]

    for metric, ylabel, ax, note in metrics_config:
        means = [results[h][f'{metric}_mean'] for h in hidden_dims]
        cis = [results[h][f'{metric}_ci'] for h in hidden_dims]
        ax.errorbar(hidden_dims, means, yerr=cis, fmt='o-', capsize=5, linewidth=2)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}\n({note})')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Scaling Analysis: Interpretability Metrics vs Model Width', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'figure8_scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path / 'figure8_scaling_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Scaling analysis across hidden sizes')
    parser.add_argument('--results-dir', type=str, default='outputs/scaling')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/scaling')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:3])
    args = parser.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = run_scaling_analysis(args.results_dir, args.hidden_dims, args.seeds, device)

    # Save results
    serializable = {}
    for h, data in results.items():
        serializable[str(h)] = {}
        for k, v in data.items():
            if isinstance(v, list):
                serializable[str(h)][k] = v
            else:
                serializable[str(h)][k] = v

    with open(output_path / 'scaling_analysis.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Print summary table
    print("\n=== SCALING ANALYSIS SUMMARY ===")
    header = f"{'Hidden':>6} {'FDR':>10} {'σ':>8} {'Mono%':>10} {'Circuit':>10} {'Acc%':>10}"
    print(header)
    print('-' * len(header))
    for h in args.hidden_dims:
        if h not in results:
            continue
        r = results[h]
        print(f"{h:>6}  {r['fdr_mean']:.4f}  "
              f"{r['sigma_mean']:.4f}  "
              f"{r['mono_fraction_mean']:.3f}     "
              f"{r['circuit_size_mean']:.1f}       "
              f"{r['accuracy_mean']:.2f}")

    # Plot
    plot_scaling_results(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()