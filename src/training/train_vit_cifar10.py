"""
Train Vision Transformer on CIFAR-10.
Supports clean training and corrupted label training.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.vit_model import create_vit_model
from utils.stats import compute_ci


def get_data_loaders(batch_size=128, num_workers=4, image_size=224):
    """Get CIFAR-10 data loaders with ViT-appropriate transforms."""
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_corrupted_data_loaders(
    noise_rate=0.2,
    batch_size=128,
    num_workers=4,
    image_size=224,
    seed=42
):
    """Get CIFAR-10 data loaders with corrupted labels."""
    np.random.seed(seed)
    
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    # Corrupt labels
    targets = np.array(train_dataset.targets)
    n_samples = len(targets)
    n_corrupt = int(n_samples * noise_rate)
    corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
    
    for idx in corrupt_indices:
        original = targets[idx]
        new_label = np.random.choice([c for c in range(10) if c != original])
        targets[idx] = new_label
    
    train_dataset.targets = targets.tolist()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset, test_dataset, corrupt_indices


def evaluate(model, test_loader, device, per_class=False):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {c: 0 for c in range(10)}
    per_class_total = {c: 0 for c in range(10)}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            
            if per_class:
                for p, t in zip(preds, y):
                    per_class_total[t.item()] += 1
                    if p == t:
                        per_class_correct[t.item()] += 1
    
    acc = correct / total
    if per_class:
        per_class_acc = {c: per_class_correct[c] / max(per_class_total[c], 1) for c in range(10)}
        return acc, per_class_acc
    return acc


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train ViT on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/vit')
    parser.add_argument('--noise-rate', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--model-name', type=str, default='google/vit-base-patch16-224')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}, Noise rate: {args.noise_rate}")
    
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    if args.noise_rate > 0:
        train_loader, test_loader, train_dataset, test_dataset, corrupt_indices = get_corrupted_data_loaders(
            noise_rate=args.noise_rate,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed
        )
        np.save(output_dir / 'corrupt_indices.npy', corrupt_indices)
        print(f"Corrupted {len(corrupt_indices)} samples")
    else:
        train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
            batch_size=args.batch_size,
            image_size=args.image_size
        )
    
    # Model
    model = create_vit_model(
        model_name=args.model_name,
        num_classes=10,
        image_size=args.image_size,
        pretrained=True
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # Training loop
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        test_acc = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_checkpoint(output_dir / 'best_model.pt', {'epoch': epoch, 'test_acc': test_acc})
        
        # Save latest
        model.save_checkpoint(output_dir / 'latest_model.pt', {'epoch': epoch, 'test_acc': test_acc})
    
    # Final evaluation
    final_acc, per_class_acc = evaluate(model, test_loader, device, per_class=True)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print("Per-class accuracy:")
    for c in range(10):
        print(f"  Class {c}: {per_class_acc[c]:.4f}")
    
    # Save history
    import json
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()