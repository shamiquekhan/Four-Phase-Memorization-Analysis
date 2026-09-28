"""
Temporal training script with per-epoch checkpointing and per-example logging (v2).

Saves:
  - Model checkpoint every epoch
  - Per-example losses, margins, predictions, gradients
  - CSL, forgetting, memorization onset tracking
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
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from data.corruption import corrupt_labels_random, CorruptionProvenance
from analysis.memorization import (
    extract_per_example_metrics,
    aggregate_temporal_metrics,
    save_temporal_artifacts,
    compute_gradient_conflict,
)


def get_data_loaders(config, noise_rate=0.0, corruption_seed=7001, loader_seed=9001):
    """Get MNIST loaders with optional label corruption."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    provenance = None
    if noise_rate > 0:
        train_dataset, provenance = corrupt_labels_random(train_dataset, noise_rate, corruption_seed)

    # Use separate loader seed
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
    return train_loader, test_loader, provenance


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save full training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Temporal training with per-epoch logging')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--init-seed', type=int, default=42)
    parser.add_argument('--corruption-seed', type=int, default=7001)
    parser.add_argument('--loader-seed', type=int, default=9001)
    parser.add_argument('--noise-rate', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='outputs/temporal')
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds (decoupled)
    torch.manual_seed(args.init_seed)
    np.random.seed(args.init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.init_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Init seed: {args.init_seed}, Corruption seed: {args.corruption_seed}, Loader seed: {args.loader_seed}")
    print(f"Noise rate: {args.noise_rate}, Optimizer: {args.optimizer}")

    # Build model
    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=config['training']['lr'],
                               weight_decay=config['training']['weight_decay'])
    elif args.optimizer == 'sgd':
        sgd_cfg = [o for o in config['optimizers'] if o['name'] == 'sgd'][0]
        optimizer = optim.SGD(model.parameters(),
                              lr=sgd_cfg['lr'],
                              momentum=sgd_cfg['momentum'],
                              weight_decay=sgd_cfg['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()

    # Data
    train_loader, test_loader, provenance = get_data_loaders(
        config, args.noise_rate, args.corruption_seed, args.loader_seed)

    # Output directory
    output_dir = Path(args.output_dir) / f"noise_{args.noise_rate}_opt_{args.optimizer}" / f"init_{args.init_seed}_corr_{args.corruption_seed}_load_{args.loader_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save provenance
    if provenance:
        provenance.save(output_dir / 'corruption_provenance.json')

    # Save config
    run_config = {
        'init_seed': args.init_seed,
        'corruption_seed': args.corruption_seed,
        'loader_seed': args.loader_seed,
        'noise_rate': args.noise_rate,
        'optimizer': args.optimizer,
        'config': config,
    }
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

    # Temporal logging config
    temporal_cfg = config.get('temporal', {})
    checkpoint_every = temporal_cfg.get('checkpoint_every', 1)
    log_per_example = temporal_cfg.get('log_per_example', True)
    log_grad_conflict = temporal_cfg.get('log_grad_conflict', True)
    max_tracin_examples = temporal_cfg.get('max_tracin_examples', 2000)

    # Storage for temporal metrics
    checkpoint_results = []
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # Per-epoch eval loader (no shuffle, fixed order for per-example tracking)
    eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    best_acc = 0

    for epoch in range(config['training']['epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['training']['epochs']} ===")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config['training'].get('grad_clip')
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Per-example metrics on training set (for temporal dynamics)
        if log_per_example:
            print("  Computing per-example metrics...")
            epoch_metrics = extract_per_example_metrics(
                model, eval_loader, provenance, device, criterion)
            checkpoint_results.append(epoch_metrics)

            # Gradient conflict
            if log_grad_conflict and provenance:
                grad_conflict = compute_gradient_conflict(
                    model, eval_loader, provenance, device, 'fc1')
                epoch_metrics['gradient_conflict'] = grad_conflict

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, output_dir / f'epoch_{epoch+1:03d}.pt')

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }, output_dir / 'best_model.pt')

    # Final save
    save_checkpoint(model, optimizer, config['training']['epochs'] - 1, {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }, output_dir / 'final_model.pt')

    torch.save(history, output_dir / 'history.pt')

    # Aggregate and save temporal metrics
    if checkpoint_results:
        print("\nAggregating temporal metrics...")
        aggregated = aggregate_temporal_metrics(checkpoint_results, provenance)
        save_temporal_artifacts(checkpoint_results, aggregated, output_dir, args.init_seed)

        print(f"Memorized examples: {aggregated['n_memorized']}")
        print(f"Examples with forgetting: {aggregated['n_forgotten']}")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()