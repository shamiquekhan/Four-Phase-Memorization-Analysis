"""
Training script for corrupted MNIST (label noise).
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from data.corruption import corrupt_labels_random


def get_data_loaders(batch_size=128, noise_rate=0.2, num_workers=4, seed=42):
    """Get MNIST train and test data loaders with guaranteed-changed label corruption."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Corrupt training labels (guarantees new != original; saves provenance)
    train_dataset, provenance = corrupt_labels_random(train_dataset, noise_rate, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, provenance


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
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
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on test set."""
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


def audit_behavioral_memorization(model, dataset, provenance, device, batch_size=512):
    """
    Behavioral memorization on training data (non-circular definition):
      memorized_i = (training_label_i != original_label_i)
                    AND (prediction_i == training_label_i)
                    AND (prediction_i != original_label_i)

    Returns dict of rates over: changed samples, clean samples, all samples.
    """
    if provenance.kind != 'random':
        return None

    model.eval()
    changed_idx = np.array(provenance.changed_indices, dtype=int)
    orig_by_idx = {i: int(o) for i, o in zip(provenance.changed_indices,
                                             provenance.original_labels)}
    targets = np.array(dataset.targets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for data, _ in tqdm(loader, desc="Auditing", leave=False):
            preds.append(model(data.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()

    if len(preds) != len(targets):
        raise RuntimeError(f"preds {len(preds)} != targets {len(targets)}")

    n = len(targets)
    is_changed = np.zeros(n, dtype=bool)
    is_changed[changed_idx] = True

    orig_labels = targets.copy()
    for i in changed_idx:
        orig_labels[i] = orig_by_idx[i]

    pred_eq_train = preds == targets
    pred_eq_orig = preds == orig_labels

    memorized = is_changed & pred_eq_train & ~pred_eq_orig
    fit_noisy = is_changed & pred_eq_train
    fit_original = is_changed & pred_eq_orig

    return {
        'n_samples': int(n),
        'n_changed': int(is_changed.sum()),
        'memorized_count': int(memorized.sum()),
        'memorized_fraction_of_changed': float(memorized.sum() / max(is_changed.sum(), 1)),
        'memorized_fraction_of_all': float(memorized.sum() / n),
        'changed_fit_noisy_label': float(fit_noisy.sum() / max(is_changed.sum(), 1)),
        'changed_fit_original_label': float(fit_original.sum() / max(is_changed.sum(), 1)),
        'clean_train_acc': float((~is_changed & pred_eq_train).sum() / max((~is_changed).sum(), 1)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train MNISTNet on corrupted MNIST')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-rate', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='outputs/corrupted')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Label noise rate: {args.noise_rate}")
    
    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader, provenance = get_data_loaders(
        batch_size=config['training']['batch_size'],
        noise_rate=args.noise_rate,
        num_workers=config['training']['num_workers'],
        seed=args.seed
    )
    corrupt_indices = np.array(provenance.changed_indices)

    output_dir = Path(args.output_dir) / f"noise_{args.noise_rate}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'corrupt_indices.npy', corrupt_indices)
    provenance.save(output_dir / 'corruption_provenance.json')
    print(f"Corruption provenance: {len(provenance.changed_indices)} changed "
          f"(rate {len(provenance.changed_indices)/provenance.n_samples:.4f})")
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_checkpoint(output_dir / 'best_model.pt', {
                'epoch': epoch, 
                'test_acc': test_acc,
                'noise_rate': args.noise_rate,
                'corrupt_indices': corrupt_indices.tolist()
            })
    
    model.save_checkpoint(output_dir / 'final_model.pt', {
        'epoch': config['training']['epochs'], 
        'test_acc': test_acc,
        'noise_rate': args.noise_rate
    })
    torch.save(history, output_dir / 'history.pt')
    
    # Behavioral memorization audit (non-circular definition)
    train_dataset = train_loader.dataset
    audit = audit_behavioral_memorization(model, train_dataset, provenance, device)
    if audit is not None:
        with open(output_dir / 'memorization_audit.json', 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"Behavioral memorization: {audit['memorized_count']}/{audit['n_changed']} "
              f"changed labels memorized ({audit['memorized_fraction_of_changed']:.1%})")
    
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()