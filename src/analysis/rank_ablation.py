"""
Rank ablation study: analyze effect of rank-k approximations on memorization.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet, CIFAR10MLP
from utils.stats import compute_ci, SEEDS


def get_low_rank_approximation(W, k):
    """Get rank-k approximation of weight matrix via SVD."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
    return W_k, S


def _get_weight(model, layer_name):
    if layer_name == 'fc1':
        return model.fc1.weight.data
    elif layer_name == 'fc2':
        return model.fc2.weight.data
    elif layer_name == 'fc3':
        return model.fc3.weight.data
    raise ValueError(f"Unknown layer: {layer_name}")


def _set_weight(model, layer_name, W):
    with torch.no_grad():
        if layer_name == 'fc1':
            model.fc1.weight.data.copy_(W)
        elif layer_name == 'fc2':
            model.fc2.weight.data.copy_(W)
        elif layer_name == 'fc3':
            model.fc3.weight.data.copy_(W)


def apply_rank_ablation(model, layer_name, k):
    """Replace weight matrix with rank-k approximation."""
    W = _get_weight(model, layer_name)
    W_k, S = get_low_rank_approximation(W, k)
    _set_weight(model, layer_name, W_k)
    return S


def evaluate_model(model, dataloader, device, criterion):
    """Evaluate model and return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(dataloader), 100. * correct / total


def get_singular_values(model, layer_name):
    """Get singular values of weight matrix."""
    with torch.no_grad():
        W = _get_weight(model, layer_name)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return S.cpu().numpy()


def run_rank_ablation(model, train_loader, test_loader, device, criterion, layer_name, max_rank=None):
    """Run rank ablation for a specific layer."""
    W_orig = _get_weight(model, layer_name).clone()

    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    full_rank = len(S)

    if max_rank is None:
        max_rank = full_rank

    ranks = list(range(1, min(max_rank, full_rank) + 1, max(1, full_rank // 20)))
    if full_rank not in ranks:
        ranks.append(full_rank)
    ranks = sorted(set(ranks))

    results = {
        'singular_values': S.tolist(),
        'ranks': ranks,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'svd_spectrum': S.tolist()
    }

    for k in ranks:
        W_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
        _set_weight(model, layer_name, W_k)

        # Evaluate
        train_loss, train_acc = evaluate_model(model, train_loader, device, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(f"  Rank {k}/{full_rank}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    _set_weight(model, layer_name, W_orig)

    return results


# ROME rank-r ablation: decompose edit via SVD, measure recovery vs side effects

def rome_edit_fc2(model, source_idx, target_idx, device, scale=1.0):
    """Apply ROME rank-1 edit to fc2 for source→target class.

    Uses the ROME rank-one update formula from Meng et al. (2022):
    W += scale * (v - W·u) · u^T / (u^T · u)
    where u is one-hot for target, v is hidden activation delta.
    """
    fc2 = model.fc2
    W = fc2.weight.data.clone()
    u = torch.zeros(fc2.out_features, device=device)
    u[target_idx] = 1.0
    v = torch.zeros(fc2.in_features, device=device)
    v[source_idx] = 1.0
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    W_new = W + scale * delta
    return W_new, W


def run_rome_rank_ablation(model, test_loader, device, source_class, target_class, scale=1.0):
    """Apply ROME edit and decompose via SVD; measure recovery vs side effects per rank.

    Returns dict with per-rank accuracy for every class.
    """
    fc2 = model.fc2
    W_orig = fc2.weight.data.clone()
    W_edited, W_old = rome_edit_fc2(model, source_class, target_class, device, scale)
    edit_delta = W_edited - W_old
    U, S, Vh = torch.linalg.svd(edit_delta, full_matrices=False)
    full_rank = len(S)
    ranks = sorted(set(list(range(1, min(full_rank, 16) + 1)) + [full_rank]))

    results = {'singular_values': S.cpu().tolist(), 'ranks': ranks, 'per_class_acc': {}}

    for k in ranks:
        delta_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
        W_k = W_orig + delta_k
        with torch.no_grad():
            fc2.weight.data.copy_(W_k)
        model.eval()
        class_correct = {c: 0 for c in range(10)}
        class_total = {c: 0 for c in range(10)}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                for c in range(10):
                    m = y == c
                    class_correct[c] += (preds[m] == y[m]).sum().item()
                    class_total[c] += m.sum().item()
        results['per_class_acc'][k] = {
            str(c): class_correct[c] / max(class_total[c], 1) for c in range(10)
        }

    with torch.no_grad():
        fc2.weight.data.copy_(W_orig)
    return results


def aggregate_rome_rank_results(all_results):
    """Aggregate ROME rank ablation across seeds into summary dict."""
    ranks = None
    for sk, res in all_results.items():
        if res:
            ranks = res['ranks']
            break
    if ranks is None:
        return {}

    agg = {
        'ranks': ranks,
        'source_class': None,
        'target_class': None,
    }
    for sk in all_results:
        if all_results.get(sk):
            agg['source_class'] = all_results[sk].get('source_class')
            agg['target_class'] = all_results[sk].get('target_class')
            break

    # Per rank: average recovery (accuracy on source), side effect (mean drop on others)
    agg['recovery'] = []
    agg['side_effects'] = []
    for k in ranks:
        recovs, sides = [], []
        for sk in all_results:
            if not all_results.get(sk):
                continue
            pac = all_results[sk]['per_class_acc'].get(k, {})
            source_acc = float(pac.get(str(agg.get('source_class', 0)), 0))
            recovs.append(source_acc)
            other_accs = [float(pac[str(c)]) for c in range(10)
                          if c != agg.get('source_class', 0) and str(c) in pac]
            sides.append(1.0 - np.mean(other_accs) if other_accs else 0)
        agg['recovery'].append(np.mean(recovs))
        agg['side_effects'].append(np.mean(sides))
    return agg


def get_data_loaders(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Rank ablation study')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/rank_ablation')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:5])
    parser.add_argument('--layer', type=str, default='fc2', choices=['fc1', 'fc2', 'fc3'])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(config['training']['batch_size'], config['training']['num_workers'])
    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {seed: {} for seed in args.seeds}
    
    for seed in args.seeds:
        checkpoint_path = Path(args.checkpoint_dir) / f"seed_{seed}" / 'final_model.pt'
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found for seed {seed}")
            continue
        
        model = MNISTNet(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            activation=config['model']['activation']
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"\nSeed {seed}: Rank ablation on {args.layer}")
        results = run_rank_ablation(model, train_loader, test_loader, device, criterion, args.layer)
        all_results[seed] = results
    
    # Aggregate: find rank where test accuracy drops significantly
    print("\n=== Rank Ablation Summary ===")
    for seed in args.seeds:
        if all_results[seed]:
            ranks = all_results[seed]['ranks']
            test_accs = all_results[seed]['test_acc']
            
            # Find rank where accuracy drops below 90% of full rank
            full_acc = test_accs[-1]
            threshold = 0.9 * full_acc
            
            critical_rank = None
            for r, acc in zip(ranks, test_accs):
                if acc < threshold:
                    critical_rank = r
                    break
            
            print(f"Seed {seed}: Full rank acc={full_acc:.2f}%, "
                  f"Critical rank (90% threshold)={critical_rank}")
    
    with open(output_dir / f'rank_ablation_{args.layer}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()