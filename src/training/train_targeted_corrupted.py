"""
Training script for targeted source→target label corruption.
Used for ROME validation: corrupts specific class pairs so ROME can recover them.
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
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from data.corruption import apply_targeted_corruption


def get_data_loaders(batch_size=128, source=7, target=1, num_workers=4, seed=42):
    """Get MNIST loaders with targeted source→target label corruption."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_dataset, provenance = apply_targeted_corruption(train_dataset, source, target, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, provenance


def train_epoch(model, loader, optimizer, criterion, device):
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


def main():
    parser = argparse.ArgumentParser(description='Train with targeted source→target corruption')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source', type=int, default=7)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='outputs/targeted_corrupted')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Targeted corruption: class {args.source} → class {args.target}")

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
        source=args.source, target=args.target,
        num_workers=config['training']['num_workers'],
        seed=args.seed
    )

    output_dir = Path(args.output_dir) / f"src{args.source}_tgt{args.target}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'corrupt_mask.npy', np.array(provenance.changed_indices, dtype=bool))
    provenance.save(output_dir / 'corruption_provenance.json')

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
                'epoch': epoch, 'test_acc': test_acc,
                'source': args.source, 'target': args.target,
                'corrupt_indices': provenance.changed_indices
            })

    model.save_checkpoint(output_dir / 'final_model.pt', {
        'epoch': config['training']['epochs'], 'test_acc': test_acc,
        'source': args.source, 'target': args.target,
        'corrupt_indices': provenance.changed_indices
    })
    torch.save(history, output_dir / 'history.pt')

    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
