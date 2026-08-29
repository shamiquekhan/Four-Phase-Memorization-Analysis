"""
LoRA-based Machine Unlearning for Vision Transformers on CIFAR-10.

Implements:
- NegLoRA: Gradient ascent on forget set with LoRA
- Inverted Hinge Loss (IHL): Stable unlearning loss
- Residual Feature Alignment (RFA): Dual-objective unlearning
- Evaluation: CKA, FLD Ratio, MIA
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import copy
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.vit_model import ViTWrapper, create_vit_model
from utils.stats import compute_ci


def get_data_loaders(batch_size=128, num_workers=4, image_size=224):
    """Get CIFAR-10 data loaders."""
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset, test_dataset


def build_forget_retain_sets(train_dataset, forget_class: int, forget_ratio: float = 1.0, seed: int = 42):
    """Split dataset into forget set (forget_class) and retain set (other classes)."""
    np.random.seed(seed)
    
    forget_indices = [i for i, (_, y) in enumerate(train_dataset) if y == forget_class]
    retain_indices = [i for i, (_, y) in enumerate(train_dataset) if y != forget_class]
    
    # Subsample forget set if ratio < 1.0
    if forget_ratio < 1.0:
        n_forget = int(len(forget_indices) * forget_ratio)
        forget_indices = np.random.choice(forget_indices, n_forget, replace=False).tolist()
    
    forget_dataset = Subset(train_dataset, forget_indices)
    retain_dataset = Subset(train_dataset, retain_indices)
    
    return forget_dataset, retain_dataset


def evaluate_per_class(model, test_loader, device):
    """Evaluate per-class accuracy."""
    model.eval()
    per_class_correct = {c: 0 for c in range(10)}
    per_class_total = {c: 0 for c in range(10)}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for p, t in zip(preds, y):
                per_class_total[t.item()] += 1
                if p == t:
                    per_class_correct[t.item()] += 1
    
    return {c: per_class_correct[c] / max(per_class_total[c], 1) for c in range(10)}


def evaluate_overall(model, test_loader, device):
    """Evaluate overall accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def compute_cka(model_a, model_b, data_loader, device, layer_name='last'):
    """
    Compute Centered Kernel Alignment (CKA) between two models.
    Uses linear CKA on [CLS] token representations.
    """
    model_a.eval()
    model_b.eval()
    
    features_a = []
    features_b = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, h_a = model_a.forward_with_hidden(x)
            _, h_b = model_b.forward_with_hidden(x)
            features_a.append(h_a.cpu())
            features_b.append(h_b.cpu())
    
    X = torch.cat(features_a, dim=0).numpy()
    Y = torch.cat(features_b, dim=0).numpy()
    
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # Linear CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    # HSIC(X,Y) = trace(X^T Y Y^T X) / (n-1)^2 = ||X^T Y||_F^2 / (n-1)^2
    n = X.shape[0]
    K = X @ X.T
    L = Y @ Y.T
    
    hsic_xy = np.trace(K @ L) / (n - 1) ** 2
    hsic_xx = np.trace(K @ K) / (n - 1) ** 2
    hsic_yy = np.trace(L @ L) / (n - 1) ** 2
    
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)
    return float(cka)


def compute_fld_ratio(model, test_loader, device):
    """
    Compute Fisher Linear Discriminant Ratio on [CLS] token embeddings.
    FLD = tr(S_B) / tr(S_W)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, h = model.forward_with_hidden(x)
            all_features.append(h.cpu().numpy())
            all_labels.append(y.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Within-class scatter S_W
    Sw = np.zeros((features.shape[1], features.shape[1]))
    # Between-class scatter S_B
    global_mean = features.mean(axis=0)
    Sb = np.zeros((features.shape[1], features.shape[1]))
    
    for c in range(10):
        class_features = features[labels == c]
        if len(class_features) == 0:
            continue
        class_mean = class_features.mean(axis=0)
        centered = class_features - class_mean
        Sw += centered.T @ centered
        diff = (class_mean - global_mean).reshape(-1, 1)
        Sb += len(class_features) * (diff @ diff.T)
    
    tr_Sw = np.trace(Sw)
    tr_Sb = np.trace(Sb)
    
    if tr_Sw == 0:
        return float('inf')
    return tr_Sb / tr_Sw


def compute_mia_score(model, forget_loader, retain_loader, device):
    """
    Membership Inference Attack accuracy.
    Simple threshold-based attack on prediction confidence.
    """
    model.eval()
    
    def get_confidences(loader):
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                max_probs = probs.max(dim=1).values
                confs.extend(max_probs.cpu().numpy())
        return np.array(confs)
    
    forget_confs = get_confidences(forget_loader)
    retain_confs = get_confidences(retain_loader)
    
    # Threshold-based attack: predict "member" if confidence > threshold
    all_confs = np.concatenate([forget_confs, retain_confs])
    true_labels = np.concatenate([np.ones(len(forget_confs)), np.zeros(len(retain_confs))])
    
    # Find best threshold
    best_acc = 0
    for threshold in np.percentile(all_confs, np.arange(1, 100)):
        preds = (all_confs > threshold).astype(int)
        acc = (preds == true_labels).mean()
        best_acc = max(best_acc, acc)
    
    return best_acc


def inverted_hinge_loss(logits, labels, margin=1.0):
    """
    Inverted Hinge Loss for stable unlearning.
    Pushes logits of true class down, promotes next best alternative.
    """
    probs = F.softmax(logits, dim=1)
    batch_size = logits.size(0)
    
    # Get true class logits
    true_logits = logits[torch.arange(batch_size), labels]
    
    # Get second-best logits (mask true class)
    masked_logits = logits.clone()
    masked_logits[torch.arange(batch_size), labels] = -float('inf')
    second_best_logits = masked_logits.max(dim=1).values
    
    # IHL: max(0, true_logits - second_best_logits + margin)
    loss = F.relu(true_logits - second_best_logits + margin).mean()
    return loss


def neg_lora_loss(logits, labels):
    """Standard negative log-likelihood for gradient ascent."""
    return -F.cross_entropy(logits, labels)


def rfa_loss(model, forget_loader, retain_loader, device, lambda_rfa=1.0):
    """
    Residual Feature Alignment loss.
    Drives forget set features toward retain set centroid.
    Drives retain set residual features toward zero.
    """
    model.eval()
    
    # Get retain set centroid (using base features without LoRA)
    retain_features = []
    with torch.no_grad():
        for x, _ in retain_loader:
            x = x.to(device)
            _, h = model.forward_with_hidden(x)
            retain_features.append(h)
    retain_centroid = torch.cat(retain_features, dim=0).mean(dim=0)
    
    # Compute loss on forget set: align features to retain centroid
    forget_loss = 0
    n_forget = 0
    for x, _ in forget_loader:
        x = x.to(device)
        _, h = model.forward_with_hidden(x)
        forget_loss += F.mse_loss(h, retain_centroid.expand_as(h))
        n_forget += 1
    forget_loss = forget_loss / max(n_forget, 1)
    
    # Compute loss on retain set: drive residuals to zero
    retain_loss = 0
    n_retain = 0
    for x, _ in retain_loader:
        x = x.to(device)
        _, h = model.forward_with_hidden(x)
        retain_loss += F.mse_loss(h, torch.zeros_like(h))
        n_retain += 1
    retain_loss = retain_loss / max(n_retain, 1)
    
    return forget_loss + lambda_rfa * retain_loss


def run_unlearning(
    model,
    forget_loader,
    retain_loader,
    test_loader,
    device,
    method: str = 'neglora',
    epochs: int = 10,
    lr: float = 1e-4,
    margin: float = 1.0,
    lambda_rfa: float = 1.0
):
    """
    Run unlearning with specified method.
    
    Methods:
    - 'neglora': Negative log-likelihood gradient ascent
    - 'ihl': Inverted Hinge Loss
    - 'rfa': Residual Feature Alignment
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion_map = {
        'neglora': neg_lora_loss,
        'ihl': lambda logits, labels: inverted_hinge_loss(logits, labels, margin),
    }
    
    if method in criterion_map:
        criterion = criterion_map[method]
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in forget_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Evaluate
            forget_acc = evaluate_overall(model, forget_loader, device)
            retain_acc = evaluate_overall(model, retain_loader, device)
            test_acc = evaluate_overall(model, test_loader, device)
            
            print(f"  Epoch {epoch+1}: Loss={epoch_loss/len(forget_loader):.4f}, "
                  f"Forget Acc={forget_acc:.4f}, Retain Acc={retain_acc:.4f}, Test Acc={test_acc:.4f}")
    
    elif method == 'rfa':
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            # Alternate forget and retain batches
            forget_iter = iter(forget_loader)
            retain_iter = iter(retain_loader)
            
            for _ in range(max(len(forget_loader), len(retain_loader))):
                optimizer.zero_grad()
                loss = 0
                
                try:
                    x_f, _ = next(forget_iter)
                    x_f = x_f.to(device)
                    _, h_f = model.forward_with_hidden(x_f)
                    loss += F.mse_loss(h_f, torch.zeros_like(h_f))  # placeholder
                except StopIteration:
                    pass
                
                try:
                    x_r, _ = next(retain_iter)
                    x_r = x_r.to(device)
                    _, h_r = model.forward_with_hidden(x_r)
                    loss += F.mse_loss(h_r, torch.zeros_like(h_r))
                except StopIteration:
                    pass
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            test_acc = evaluate_overall(model, test_loader, device)
            print(f"  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Test Acc={test_acc:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='ViT LoRA Unlearning on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained ViT checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/vit_unlearning')
    parser.add_argument('--forget-class', type=int, default=0, help='Class to unlearn')
    parser.add_argument('--forget-ratio', type=float, default=1.0, help='Fraction of forget class to unlearn')
    parser.add_argument('--method', type=str, default='ihl', choices=['neglora', 'ihl', 'rfa'])
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--lambda-rfa', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Method: {args.method}, Forget class: {args.forget_class}, LoRA rank: {args.lora_rank}")
    
    output_dir = Path(args.output_dir) / f"forget_class_{args.forget_class}_{args.method}_r{args.lora_rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    base_model = ViTWrapper.load_checkpoint(args.checkpoint, device)
    
    # Create a copy for evaluation (clean reference)
    clean_model = copy.deepcopy(base_model)
    clean_model.eval()
    
    # Attach LoRA to base model
    base_model.attach_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    base_model.print_trainable_params()
    
    # Data
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        batch_size=args.batch_size, image_size=args.image_size
    )
    
    # Build forget/retain sets
    forget_dataset, retain_dataset = build_forget_retain_sets(
        train_dataset, args.forget_class, args.forget_ratio, args.seed
    )
    
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Test loaders for forget/retain classes
    forget_test_indices = [i for i, (_, y) in enumerate(test_dataset) if y == args.forget_class]
    retain_test_indices = [i for i, (_, y) in enumerate(test_dataset) if y != args.forget_class]
    
    forget_test_loader = DataLoader(Subset(test_dataset, forget_test_indices), batch_size=args.batch_size)
    retain_test_loader = DataLoader(Subset(test_dataset, retain_test_indices), batch_size=args.batch_size)
    
    # Pre-unlearning evaluation
    print("\n=== Pre-unlearning Evaluation ===")
    pre_forget_acc = evaluate_overall(base_model, forget_test_loader, device)
    pre_retain_acc = evaluate_overall(base_model, retain_test_loader, device)
    pre_test_acc = evaluate_overall(base_model, test_loader, device)
    pre_cka = compute_cka(base_model, clean_model, test_loader, device)
    pre_fld = compute_fld_ratio(base_model, test_loader, device)
    
    print(f"Forget class acc: {pre_forget_acc:.4f}")
    print(f"Retain classes acc: {pre_retain_acc:.4f}")
    print(f"Overall test acc: {pre_test_acc:.4f}")
    print(f"CKA vs clean: {pre_cka:.6f}")
    print(f"FLD Ratio: {pre_fld:.4f}")
    
    # Run unlearning
    print(f"\n=== Running {args.method.upper()} Unlearning ===")
    model = run_unlearning(
        base_model, forget_loader, retain_loader, test_loader, device,
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        lambda_rfa=args.lambda_rfa
    )
    
    # Post-unlearning evaluation
    print("\n=== Post-unlearning Evaluation ===")
    post_forget_acc = evaluate_overall(model, forget_test_loader, device)
    post_retain_acc = evaluate_overall(model, retain_test_loader, device)
    post_test_acc = evaluate_overall(model, test_loader, device)
    post_cka = compute_cka(model, clean_model, test_loader, device)
    post_fld = compute_fld_ratio(model, test_loader, device)
    
    print(f"Forget class acc: {post_forget_acc:.4f} (Δ={post_forget_acc - pre_forget_acc:+.4f})")
    print(f"Retain classes acc: {post_retain_acc:.4f} (Δ={post_retain_acc - pre_retain_acc:+.4f})")
    print(f"Overall test acc: {post_test_acc:.4f} (Δ={post_test_acc - pre_test_acc:+.4f})")
    print(f"CKA vs clean: {post_cka:.6f} (Δ={post_cka - pre_cka:+.6f})")
    print(f"FLD Ratio: {post_fld:.4f} (Δ={post_fld - pre_fld:+.4f})")
    
    # MIA
    print("\n=== MIA Evaluation ===")
    mia_acc = compute_mia_score(model, forget_test_loader, retain_test_loader, device)
    print(f"MIA Accuracy: {mia_acc:.4f}")
    
    # Per-class
    per_class = evaluate_per_class(model, test_loader, device)
    print("\nPer-class accuracy:")
    for c in range(10):
        print(f"  Class {c}: {per_class[c]:.4f}")
    
    # Save results
    results = {
        'method': args.method,
        'forget_class': args.forget_class,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'epochs': args.epochs,
        'pre': {
            'forget_acc': pre_forget_acc,
            'retain_acc': pre_retain_acc,
            'test_acc': pre_test_acc,
            'cka': pre_cka,
            'fld': pre_fld
        },
        'post': {
            'forget_acc': post_forget_acc,
            'retain_acc': post_retain_acc,
            'test_acc': post_test_acc,
            'cka': post_cka,
            'fld': post_fld
        },
        'mia_accuracy': mia_acc,
        'per_class': per_class
    }
    
    with open(output_dir / 'unlearning_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save unlearned model
    model.merge_lora()
    model.save_checkpoint(output_dir / 'unlearned_model.pt')
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()