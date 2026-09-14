#!/usr/bin/env python3
"""
Temporal training: saves checkpoints at specified epochs for temporal dynamics analysis.
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
from data.corruption import corrupt_labels_random


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
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
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-rate', type=float, default=0.2)
    parser.add_argument('--output-dir', type=str, default='outputs/temporal')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--checkpoint-epochs', type=int, nargs='+', default=[0, 1, 2, 4, 8, 12, 16, 20])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, noise={args.noise_rate}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    provenance = None
    if args.noise_rate > 0:
        train_dataset, provenance = corrupt_labels_random(train_dataset, args.noise_rate, args.seed)
        corrupt_indices = np.array(provenance.changed_indices)
    else:
        corrupt_indices = None

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir) / (f"noise_{args.noise_rate}" if args.noise_rate > 0 else "clean") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if provenance:
        provenance.save(output_dir / 'corruption_provenance.json')

    checkpoint_epochs = set(args.checkpoint_epochs)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{args.epochs}: Train {train_acc:.2f}%, Test {test_acc:.2f}%")

        if epoch in checkpoint_epochs:
            ckpt_dir = output_dir / f"epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(ckpt_dir / 'model.pt', {
                'epoch': epoch, 'test_acc': test_acc,
                'noise_rate': args.noise_rate
            })
            if provenance:
                provenance.save(ckpt_dir / 'corruption_provenance.json')

    torch.save(history, output_dir / 'history.pt')
    print(f"Temporal checkpoints saved to {output_dir}")


if __name__ == '__main__':
    main()