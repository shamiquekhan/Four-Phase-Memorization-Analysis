"""
Phase 5: LoRA vs ROME Subspace Comparison

Compares gradient-learned low-rank adapters (LoRA) against closed-form 
rank-one model editing (ROME) on the targeted corruption configurations.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json
import copy
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from models.lora_layer import LoRALinear, attach_lora, detach_lora
from analysis.subspace_overlap import (
    principal_angles, 
    subspace_overlap_score, 
    subspace_overlap_score_all_angles,
    frobenius_norm,
    compute_rome_delta_W
)
from utils.stats import compute_ci
from utils.metrics import evaluate_class_accuracy

# Import ROME functions - need to add parent to path for sibling module
sys.path.append(str(Path(__file__).parent))
from multiclass_rome import run_rome_experiment, compute_rome_edit


CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
    {'source': 5, 'target': 6, 'label': '5→6'},
    {'source': 0, 'target': 8, 'label': '0→8'},
]

DEFAULT_RANKS = [1, 2, 4, 8]
DEFAULT_N_EXAMPLES = 100
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-2
EARLY_STOP_PATIENCE = 3


def build_correction_set(train_dataset, source_class: int, n_examples: int = DEFAULT_N_EXAMPLES, seed: int = 42):
    """Pulls n_examples correctly-labeled examples of source_class from the original label set."""
    rng = torch.Generator().manual_seed(seed)
    class_indices = [i for i, (_, y) in enumerate(train_dataset) if y == source_class]
    if len(class_indices) < n_examples:
        n_examples = len(class_indices)
    perm = torch.randperm(len(class_indices), generator=rng)[:n_examples]
    selected = [class_indices[i] for i in perm]
    return Subset(train_dataset, selected)


def get_train_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_per_class(model, test_loader, device):
    """Evaluate accuracy per class."""
    model.eval()
    per_class_correct = {c: 0 for c in range(10)}
    per_class_total = {c: 0 for c in range(10)}
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for p, t in zip(preds, y):
                per_class_total[t.item()] += 1
                if p == t:
                    per_class_correct[t.item()] += 1
    per_class_acc = {c: per_class_correct[c] / max(per_class_total[c], 1) for c in range(10)}
    return per_class_acc


def train_lora_correction(
    model, 
    lora_layer: LoRALinear, 
    correction_loader: DataLoader, 
    device,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: int = EARLY_STOP_PATIENCE
) -> Tuple[List[float], int]:
    """
    Train LoRA on correction set.
    
    Returns:
        loss_history: List of training losses per epoch
        epochs_trained: Number of epochs until early stopping
    """
    optimizer = torch.optim.Adam(lora_layer.get_lora_params(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.eval()  # base network stays frozen
    
    history = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x, y in correction_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)
        
        # Early stopping based on loss (more stable than accuracy for this task)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return history, len(history)


def run_lora_correction(
    model,
    correction_loader: DataLoader,
    test_loader: DataLoader,
    source_class: int,
    target_class: int,
    layer_name: str = 'fc2',
    rank: int = 1,
    alpha: Optional[float] = None,
    device: str = 'cpu',
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    n_examples: int = DEFAULT_N_EXAMPLES,
    seed: int = 42
) -> Dict:
    """
    Run full LoRA correction pipeline for a single config/rank/seed.
    
    Returns dict with recovery, side effects, edit norm, epochs, and delta_W.
    """
    # Baseline accuracy
    pre_accs = evaluate_per_class(model, test_loader, device)
    pre_source_acc = pre_accs[source_class]
    
    # Attach LoRA
    lora_layer = attach_lora(model, layer_name, rank, alpha)
    lora_layer.to(device)
    
    # Verify delta_W is zero at initialization
    init_delta_norm = frobenius_norm(lora_layer.delta_W())
    if init_delta_norm > 1e-10:
        print(f"  WARNING: Initial delta_W norm = {init_delta_norm:.6f} (should be ~0)")
    
    # Train LoRA
    loss_history, epochs_trained = train_lora_correction(
        model, lora_layer, correction_loader, device, epochs, lr
    )
    
    # Post-correction accuracy
    post_accs = evaluate_per_class(model, test_loader, device)
    post_source_acc = post_accs[source_class]
    
    # Compute metrics
    recovery = post_source_acc - pre_source_acc
    
    other_classes = [c for c in range(10) if c != source_class]
    side_effects = float(np.mean([abs(post_accs[c] - pre_accs[c]) for c in other_classes]))
    
    # LoRA delta_W
    delta_W = lora_layer.delta_W()
    edit_norm = frobenius_norm(delta_W)
    
    # Detach LoRA to restore original model
    detach_lora(model, layer_name)
    
    return {
        'recovery': recovery,
        'side_effects': side_effects,
        'edit_norm': edit_norm,
        'epochs_trained': epochs_trained,
        'loss_history': loss_history,
        'pre_accs': pre_accs,
        'post_accs': post_accs,
        'delta_W': delta_W.cpu().numpy() if isinstance(delta_W, torch.Tensor) else delta_W,
    }


def run_rome_correction(
    model,
    test_loader: DataLoader,
    source_class: int,
    target_class: int,
    layer_name: str = 'fc2',
    device: str = 'cpu'
) -> Dict:
    """
    Run ROME correction for comparison.
    Uses the existing multiclass_rome functions.
    """
    recovery, side_effects, magnitude, pre_accs, meta = run_rome_experiment(
        model, test_loader, device, target_class, layer_name
    )
    
    # Also get the delta_W for subspace comparison
    delta_W = compute_rome_delta_W(model, target_class, layer_name)
    
    return {
        'recovery': recovery,
        'side_effects': side_effects,
        'edit_norm': magnitude,
        'pre_accs': pre_accs,
        'post_accs': {c: pre_accs[c] + (recovery if c == target_class else 0) for c in range(10)},
        'delta_W': delta_W.cpu().numpy() if isinstance(delta_W, torch.Tensor) else delta_W,
        'layer': layer_name,
    }


def run_single_experiment(
    cfg: Dict,
    seed: int,
    rank: int,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str,
    layer_name: str = 'fc2',
    n_examples: int = DEFAULT_N_EXAMPLES,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    alpha: Optional[float] = None
) -> Dict:
    """
    Run one complete experiment: ROME + LoRA for a given config, seed, rank.
    """
    label = cfg['label']
    source = cfg['source']
    target = cfg['target']
    
    print(f"\n  Config: {label}, Seed: {seed}, Rank: {rank}, Layer: {layer_name}")
    
    # Load model
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = MNISTNet(
        config['model']['input_dim'],
        config['model']['hidden_dim'],
        config['model']['output_dim']
    ).to(device)
    
    ckpt_path = checkpoint_dir / f"src{source}_tgt{target}" / f"seed_{seed}" / 'final_model.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Data loaders
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    
    # Build correction set (clean examples of source class)
    train_dataset = train_loader.dataset
    correction_subset = build_correction_set(train_dataset, source, n_examples, seed)
    correction_loader = DataLoader(correction_subset, batch_size=min(32, n_examples), shuffle=True)
    
    # Create a copy for ROME (since it modifies weights in-place)
    model_rome = copy.deepcopy(model)
    
    # Run ROME - target the source class (the corrupted class we want to recover)
    print(f"    Running ROME...")
    rome_results = run_rome_correction(model_rome, test_loader, source, source, layer_name, device)
    print(f"      ROME: recovery={rome_results['recovery']:.4f}, side_effects={rome_results['side_effects']:.4f}, norm={rome_results['edit_norm']:.4f}")
    
    # Run LoRA - train on clean examples of source class
    print(f"    Running LoRA...")
    lora_results = run_lora_correction(
        model, correction_loader, test_loader, source, source,
        layer_name, rank, alpha, device, epochs, lr, n_examples, seed
    )
    print(f"      LoRA: recovery={lora_results['recovery']:.4f}, side_effects={lora_results['side_effects']:.4f}, norm={lora_results['edit_norm']:.4f}, epochs={lora_results['epochs_trained']}")
    
    # Subspace overlap
    rome_delta = rome_results['delta_W']
    lora_delta = lora_results['delta_W']
    
    if isinstance(rome_delta, torch.Tensor):
        rome_delta = rome_delta.cpu().numpy()
    if isinstance(lora_delta, torch.Tensor):
        lora_delta = lora_delta.cpu().numpy()
    
    overlap_details = subspace_overlap_score_all_angles(rome_delta, lora_delta)
    overlap = overlap_details['overlap_score']
    
    print(f"      Subspace overlap: {overlap:.4f} (angle: {overlap_details['min_angle_deg']:.2f}°)")
    
    return {
        'config': label,
        'seed': seed,
        'rank': rank,
        'layer': layer_name,
        'rome': rome_results,
        'lora': lora_results,
        'subspace_overlap': overlap_details,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 5: LoRA vs ROME subspace comparison')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/targeted_corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/phase5')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--ranks', type=int, nargs='+', default=DEFAULT_RANKS)
    parser.add_argument('--layer', type=str, default='fc2', choices=['fc1', 'fc2'])
    parser.add_argument('--n-examples', type=int, default=DEFAULT_N_EXAMPLES)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--n-workers', type=int, default=4)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use all 10 seeds by default
    if args.seeds is None:
        seeds = config['seeds']
    else:
        seeds = args.seeds
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Ranks: {args.ranks}")
    print(f"Layer: {args.layer}")
    print(f"Correction examples per class: {args.n_examples}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    all_results = []
    
    for cfg in CORRUPTION_CONFIGS:
        for rank in args.ranks:
            for seed in seeds:
                try:
                    result = run_single_experiment(
                        cfg, seed, rank, checkpoint_dir, output_dir, device,
                        layer_name=args.layer,
                        n_examples=args.n_examples,
                        epochs=args.epochs,
                        lr=args.lr,
                        alpha=args.alpha
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'config': cfg['label'],
                        'seed': seed,
                        'rank': rank,
                        'layer': args.layer,
                        'error': str(e)
                    })
    
    # Save raw results
    raw_results_path = output_dir / 'raw_results.json'
    with open(raw_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump([convert(r) for r in all_results], f, indent=2)
    
    print(f"\nRaw results saved to {raw_results_path}")
    
    # Generate aggregated table (Table III format)
    print("\n=== AGGREGATED RESULTS (Table III) ===")
    
    # Group by config, rank
    aggregated = {}
    for r in all_results:
        if 'error' in r:
            continue
        key = (r['config'], r['rank'])
        if key not in aggregated:
            aggregated[key] = {'rome_rec': [], 'lora_rec': [], 'rome_se': [], 'lora_se': [], 
                              'rome_norm': [], 'lora_norm': [], 'overlap': [], 'epochs': []}
        aggregated[key]['rome_rec'].append(r['rome']['recovery'])
        aggregated[key]['lora_rec'].append(r['lora']['recovery'])
        aggregated[key]['rome_se'].append(r['rome']['side_effects'])
        aggregated[key]['lora_se'].append(r['lora']['side_effects'])
        aggregated[key]['rome_norm'].append(r['rome']['edit_norm'])
        aggregated[key]['lora_norm'].append(r['lora']['edit_norm'])
        aggregated[key]['overlap'].append(r['subspace_overlap']['overlap_score'])
        aggregated[key]['epochs'].append(r['lora']['epochs_trained'])
    
    # Print table
    header = f"{'Config':>8} {'Rank':>4} {'ROME Rec':>14} {'LoRA Rec':>14} {'ROME SE':>12} {'LoRA SE':>12} {'ROME Norm':>12} {'LoRA Norm':>12} {'Overlap':>10} {'Epochs':>8}"
    print(header)
    print('-' * len(header))
    
    table_rows = []
    for (config_label, rank), vals in sorted(aggregated.items()):
        def fmt_ci(vals_list):
            if not vals_list:
                return "N/A"
            mean, lo, hi = compute_ci(vals_list)
            return f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        
        def fmt_mean(vals_list):
            if not vals_list:
                return "N/A"
            return f"{np.mean(vals_list):.4f}"
        
        rome_rec_ci = fmt_ci(vals['rome_rec'])
        lora_rec_ci = fmt_ci(vals['lora_rec'])
        rome_se = fmt_mean(vals['rome_se'])
        lora_se = fmt_mean(vals['lora_se'])
        rome_norm = fmt_mean(vals['rome_norm'])
        lora_norm = fmt_mean(vals['lora_norm'])
        overlap = fmt_mean(vals['overlap'])
        epochs = fmt_mean(vals['epochs'])
        
        row = f"{config_label:>8} {rank:>4} {rome_rec_ci:>14} {lora_rec_ci:>14} {rome_se:>12} {lora_se:>12} {rome_norm:>12} {lora_norm:>12} {overlap:>10} {epochs:>8}"
        print(row)
        table_rows.append({
            'config': config_label,
            'rank': rank,
            'rome_recovery_ci': rome_rec_ci,
            'lora_recovery_ci': lora_rec_ci,
            'rome_side_effects': rome_se,
            'lora_side_effects': lora_se,
            'rome_edit_norm': rome_norm,
            'lora_edit_norm': lora_norm,
            'subspace_overlap': overlap,
            'epochs_trained': epochs,
        })
    
    # Save aggregated table
    table_path = output_dir / 'table3_lora_vs_rome.csv'
    import pandas as pd
    df = pd.DataFrame(table_rows)
    df.to_csv(table_path, index=False)
    print(f"\nTable saved to {table_path}")
    
    # Statistical comparison: paired t-test ROME vs LoRA recovery per config/rank
    from scipy import stats
    print("\n=== PAIRED T-TEST: ROME vs LoRA Recovery ===")
    n_comparisons = len(aggregated)
    alpha_corrected = 0.05 / max(n_comparisons, 1)
    
    for (config_label, rank), vals in sorted(aggregated.items()):
        if len(vals['rome_rec']) > 1 and len(vals['lora_rec']) > 1:
            t_stat, p_val = stats.ttest_rel(vals['rome_rec'], vals['lora_rec'])
            sig = "***" if p_val < alpha_corrected else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  {config_label} rank={rank}: t={t_stat:.3f}, p={p_val:.4f} {sig} (Bonferroni α={alpha_corrected:.4f})")
    
    print(f"\nPhase 5 complete. Results in {output_dir}")


if __name__ == '__main__':
    main()