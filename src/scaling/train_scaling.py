"""
Scaling experiment: train models with varying hidden dimensions.
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
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.stats import SEEDS
from data.corruption import corrupt_labels_random


def get_data_loaders(batch_size=128, num_workers=4, noise_rate=0.0, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    provenance = None
    if noise_rate > 0:
        # v2: guaranteed-change corruption with provenance (seed-DEPENDENT
        # so each scaling seed gets its own corruption draw, consistent with
        # the main corrupted training)
        train_dataset, provenance = corrupt_labels_random(train_dataset, noise_rate, seed)

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
    parser = argparse.ArgumentParser(description='Scaling experiment: train models with varying hidden dimensions')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[16, 32, 64, 128, 256])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:5])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--noise-rate', type=float, default=0.0,
                        help='Label noise rate (0 = clean; v2 adds corruption + provenance)')
    parser.add_argument('--output-dir', type=str, default='outputs/scaling')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir) / (f"noise_{args.noise_rate}" if args.noise_rate > 0 else "clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for hidden_dim in args.hidden_dims:
        print(f"\n=== Hidden Dim: {hidden_dim} (noise={args.noise_rate}) ===")
        all_results[hidden_dim] = {}

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            train_loader, test_loader, provenance = get_data_loaders(
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                noise_rate=args.noise_rate,
                seed=seed
            )
            
            model = MNISTNet(
                input_dim=config['model']['input_dim'],
                hidden_dim=hidden_dim,
                output_dim=config['model']['output_dim'],
                activation=config['model']['activation']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
            
            seed_dir = output_dir / f"hidden_{hidden_dim}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if provenance is not None:
                provenance.save(seed_dir / 'corruption_provenance.json')
                np.save(seed_dir / 'corrupt_indices.npy', np.array(provenance.changed_indices))
            
            best_acc = 0
            history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    model.save_checkpoint(seed_dir / 'best_model.pt', {'epoch': epoch, 'test_acc': test_acc})
            
            model.save_checkpoint(seed_dir / 'final_model.pt', {'epoch': args.epochs, 'test_acc': test_acc})
            torch.save(history, seed_dir / 'history.pt')
            
            all_results[hidden_dim][seed] = {
                'best_test_acc': best_acc,
                'final_test_acc': test_acc,
                'history': history
            }

            # v2: behavioral memorization audit (noisy-label fit) on train set
            if provenance is not None:
                import importlib.util
                _tc_path = Path(__file__).parent.parent / 'training' / 'train_corrupted.py'
                _spec = importlib.util.spec_from_file_location('train_corrupted', _tc_path)
                _tc = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_tc)
                try:
                    audit = _tc.audit_behavioral_memorization(
                        model, train_loader.dataset, provenance, device)
                    all_results[hidden_dim][seed]['behavioral_memorization'] = audit
                    print(f"    behavioral memorization: "
                          f"{audit['memorized_fraction_of_changed']*100:.2f}% of changed")
                except Exception as e:
                    print(f"    behavioral audit failed: {e}")

            print(f"  Seed {seed}: Best Test Acc = {best_acc:.2f}%")
    
    # Aggregate results
    print("\n=== Scaling Results Summary ===")
    summary = {}
    for hidden_dim in args.hidden_dims:
        accs = [all_results[hidden_dim][s]['best_test_acc'] for s in args.seeds]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary[hidden_dim] = {'mean': mean_acc, 'std': std_acc, 'seeds': accs}
        print(f"Hidden {hidden_dim}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    with open(output_dir / 'scaling_results.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    
    with open(output_dir / 'scaling_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()