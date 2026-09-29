import sys
sys.path.append('src')
import yaml
with open('configs/experiment_config.yaml') as f:
    config = yaml.safe_load(f)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.model import MNISTNet
from data.corruption import corrupt_labels_random
from analysis.memorization import extract_per_example_metrics, aggregate_temporal_metrics
from pathlib import Path
import json

device = torch.device('cuda')
output_dir = Path('outputs/targeted_sweep')
output_dir.mkdir(parents=True, exist_ok=True)

# Key configs from roadmap that should yield >10% memorization
test_configs = [
    {'hidden_dim': 64, 'noise_rate': 0.4, 'epochs': 50},
    {'hidden_dim': 64, 'noise_rate': 0.6, 'epochs': 50},
    {'hidden_dim': 128, 'noise_rate': 0.4, 'epochs': 50},
    {'hidden_dim': 128, 'noise_rate': 0.6, 'epochs': 50},
    {'hidden_dim': 256, 'noise_rate': 0.4, 'epochs': 50},
    {'hidden_dim': 256, 'noise_rate': 0.6, 'epochs': 50},
]

seeds = [(42, 7001, 9001), (123, 7002, 9002), (456, 7003, 9003)]

results = []
results_file = output_dir / 'targeted_sweep_results.json'

# Load existing
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)

completed = set((r['hidden_dim'], r['noise_rate'], r['epochs'], r['init_seed']) for r in results)

for cfg in test_configs:
    for init_seed, corruption_seed, loader_seed in seeds:
        if (cfg['hidden_dim'], cfg['noise_rate'], cfg['epochs'], init_seed) in completed:
            print(f"SKIP h={cfg['hidden_dim']} noise={cfg['noise_rate']} epochs={cfg['epochs']} seed={init_seed}")
            continue
        
        print(f"\nRunning: h={cfg['hidden_dim']} noise={cfg['noise_rate']} epochs={cfg['epochs']} seed={init_seed}")
        
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        
        model = MNISTNet(
            input_dim=config['model']['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            output_dim=config['model']['output_dim'],
            activation=config['model']['activation']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(),
                               lr=config['training']['lr'],
                               weight_decay=config['training']['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        criterion_none = nn.CrossEntropyLoss(reduction='none')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_dataset, provenance = corrupt_labels_random(train_dataset, cfg['noise_rate'], corruption_seed)
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
        eval_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
        
        checkpoint_results = []
        for epoch in range(cfg['epochs']):
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
            
            if epoch % 10 == 0:
                epoch_metrics = extract_per_example_metrics(model, eval_loader, provenance, device, criterion_none)
                checkpoint_results.append(epoch_metrics)
                print(f"  Epoch {epoch}: train_acc={train_acc:.2f}%")
        
        if checkpoint_results:
            aggregated = aggregate_temporal_metrics(checkpoint_results, provenance)
            n_changed = len(provenance.changed_indices)
            n_memorized = aggregated['n_memorized']
            mem_frac = n_memorized / max(n_changed, 1)
            
            result = {
                'hidden_dim': cfg['hidden_dim'],
                'noise_rate': cfg['noise_rate'],
                'epochs': cfg['epochs'],
                'init_seed': init_seed,
                'corruption_seed': corruption_seed,
                'loader_seed': loader_seed,
                'train_acc': train_acc,
                'n_changed': n_changed,
                'n_memorized': n_memorized,
                'memorization_fraction': mem_frac,
                'n_forgotten': aggregated['n_forgotten'],
                'mean_csl': float(aggregated['csl'].mean()),
                'mean_forgetting': float(aggregated['forgetting'].mean()),
            }
            results.append(result)
            
            # Save incrementally
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"  Result: mem_frac={mem_frac:.4f} ({n_memorized}/{n_changed})")
        else:
            print(f"  No checkpoints saved")

print("\n=== TARGETED SWEEP COMPLETE ===")
for r in results:
    print(f"h={r['hidden_dim']:3d} noise={r['noise_rate']:.1f} epochs={r['epochs']:2d} seed={r['init_seed']}: mem={r['memorization_fraction']:.4f} test_acc={r['train_acc']:.2f}%")
