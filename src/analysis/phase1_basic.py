"""
Phase 1: Basic memorization analysis - epoch-wise accuracy, loss, weight norms.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.stats import compute_ci, SEEDS
from utils.metrics import compute_sigma_and_fdr, extract_hidden_activations


def load_model_and_history(checkpoint_path, config, device):
    """Load model and training history."""
    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    history_path = checkpoint_path.parent / 'history.pt'
    history = torch.load(history_path) if history_path.exists() else None
    
    return model, history


def compute_weight_norms(model):
    """Compute weight matrix norms for each layer."""
    with torch.no_grad():
        fc1_norm = model.fc1.weight.data.norm(p='fro').item()
        fc2_norm = model.fc2.weight.data.norm(p='fro').item()
        fc1_spectral = torch.linalg.norm(model.fc1.weight.data, ord=2).item()
        fc2_spectral = torch.linalg.norm(model.fc2.weight.data, ord=2).item()
    return {
        'fc1_frobenius': fc1_norm,
        'fc2_frobenius': fc2_norm,
        'fc1_spectral': fc1_spectral,
        'fc2_spectral': fc2_spectral
    }


def compute_gradient_norms(model, dataloader, criterion, device, num_batches=5):
    """Compute average gradient norms."""
    model.train()
    grad_norms = []
    
    for i, (data, target) in enumerate(dataloader):
        if i >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)
    
    return np.mean(grad_norms), np.std(grad_norms)


def compute_weight_norms(model):
    """Compute weight matrix norms for each layer."""
    with torch.no_grad():
        fc1_norm = model.fc1.weight.data.norm(p='fro').item()
        fc2_norm = model.fc2.weight.data.norm(p='fro').item()
        fc1_spectral = torch.linalg.norm(model.fc1.weight.data, ord=2).item()
        fc2_spectral = torch.linalg.norm(model.fc2.weight.data, ord=2).item()
    return {
        'fc1_frobenius': fc1_norm,
        'fc2_frobenius': fc2_norm,
        'fc1_spectral': fc1_spectral,
        'fc2_spectral': fc2_spectral
    }


def analyze_checkpoint(model, history, dataloader, criterion, device, epoch):
    """Analyze a single checkpoint: norms + SPECTRAL metrics + FDR.

    v2: adds stable rank / effective rank / spectral entropy per layer so
    that rank claims are measured directly instead of inferred from the
    spectral norm (which conflates scale with rank).
    """
    from utils.metrics import compute_spectral_metrics
    weight_norms = compute_weight_norms(model)
    _sp_fc1 = compute_spectral_metrics(model.fc1.weight.data)
    _sp_fc2 = compute_spectral_metrics(model.fc2.weight.data)
    spectral = {
        'fc1_stable_rank': _sp_fc1['stable_rank'],
        'fc2_stable_rank': _sp_fc2['stable_rank'],
        'fc1_effective_rank': _sp_fc1['effective_rank'],
        'fc2_effective_rank': _sp_fc2['effective_rank'],
        'fc1_spectral_entropy': _sp_fc1['spectral_entropy'],
        'fc2_spectral_entropy': _sp_fc2['spectral_entropy'],
        'fc2_cum_energy_top1': _sp_fc2['cumulative_energy'].get(1),
        'fc2_cum_energy_top4': _sp_fc2['cumulative_energy'].get(4),
    }
    grad_mean, grad_std = compute_gradient_norms(model, dataloader, criterion, device, num_batches=5)

    # FDR and sigma from hidden activations
    hidden_acts, labels = extract_hidden_activations(model, dataloader, device)
    sf = compute_sigma_and_fdr(hidden_acts, labels)

    results = {
        'epoch': epoch,
        **weight_norms,
        **spectral,
        'grad_norm_mean': grad_mean,
        'grad_norm_std': grad_std,
        'sigma': sf['sigma'],
        'fdr': sf['fdr'],
    }
    
    if history:
        results['train_loss'] = history['train_loss'][epoch] if epoch < len(history['train_loss']) else None
        results['train_acc'] = history['train_acc'][epoch] if epoch < len(history['train_acc']) else None
        results['test_loss'] = history['test_loss'][epoch] if epoch < len(history['test_loss']) else None
        results['test_acc'] = history['test_acc'][epoch] if epoch < len(history['test_acc']) else None
    
    return results


def get_data_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Basic memorization analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase1')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:10])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_data_loader(config['training']['batch_size'], config['training']['num_workers'])
    criterion = nn.CrossEntropyLoss()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {seed: [] for seed in args.seeds}
    
    for seed in args.seeds:
        checkpoint_dir = Path(args.checkpoint_dir) / f"seed_{seed}"
        final_checkpoint = checkpoint_dir / 'final_model.pt'
        
        if not final_checkpoint.exists():
            print(f"Checkpoint not found for seed {seed}")
            continue
        
        model, history = load_model_and_history(final_checkpoint, config, device)
        model.eval()
        
        # Analyze final checkpoint
        results = analyze_checkpoint(model, history, test_loader, criterion, device, 
                                   config['training']['epochs'] - 1)
        all_results[seed].append(results)
        
        test_acc = results.get('test_acc')
        test_acc_str = f"{test_acc:.2f}" if test_acc is not None else "N/A"
        print(f"Seed {seed}: Test Acc = {test_acc_str}%, "
              f"FC1 Frobenius = {results['fc1_frobenius']:.4f}, "
              f"FC2 Frobenius = {results['fc2_frobenius']:.4f}")
    
    # Aggregate across seeds
    print("\n=== Aggregated Results (10 seeds) ===")
    metrics = ['test_acc', 'fc1_frobenius', 'fc2_frobenius', 'fc1_spectral', 'fc2_spectral',
               'grad_norm_mean',
               'fc1_stable_rank', 'fc2_stable_rank', 'fc1_effective_rank',
               'fc2_effective_rank', 'fc1_spectral_entropy', 'fc2_spectral_entropy',
               'fc2_cum_energy_top1', 'fc2_cum_energy_top4']
    
    for metric in metrics:
        values = [all_results[s][0][metric] for s in args.seeds if all_results[s] and all_results[s][0].get(metric) is not None]
        if values:
            mean, ci_low, ci_high = compute_ci(values)
            print(f"{metric}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Save detailed results
    import json
    with open(output_dir / 'phase1_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()