"""
Multi-layer ROME for MNISTNet and CIFARNet.
Implements two strategies:
  1. Sequential: apply ROME to fc2, then re-apply to fc1 given the updated model
  2. Joint: alternating optimization for n_iters rounds

Strategy 1 is cheaper; strategy 2 is closer to optimal.
"""
import argparse
import json
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.metrics import evaluate_class_accuracy
from utils.stats import compute_ci, SEEDS
from analysis.multiclass_rome import (
    compute_rank1_edit, apply_rank1_edit, get_test_dataset, build_edit_eval_loaders,
)

CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
    {'source': 5, 'target': 6, 'label': '5→6'},
    {'source': 0, 'target': 8, 'label': '0→8'},
]


def sequential_multilayer_rome(
    model,
    edit_loader,
    src_class: int,
    tgt_class: int,
    layers: list = ['fc2', 'fc1'],
    device: str = 'cpu',
):
    """
    Apply rank-one edits sequentially: layers[0] first, then layers[1] on the updated model.
    Edit directions are built from edit_loader only. Returns (edited_model, [(layer_name, delta), ...]).
    """
    model_copy = deepcopy(model)
    deltas = []

    for layer_name in layers:
        delta, u, v, _ = compute_rank1_edit(
            model_copy, edit_loader, device, tgt_class, layer_name=layer_name
        )
        if delta is None:
            continue
        apply_rank1_edit(model_copy, delta, layer_name)
        deltas.append((layer_name, delta))

    return model_copy, deltas


def joint_multilayer_rome(
    model,
    edit_loader,
    eval_loader,
    src_class: int,
    tgt_class: int,
    n_iters: int = 3,
    device: str = 'cpu',
):
    """
    Iterative joint editing: alternate between editing fc2 and fc1 for n_iters rounds.
    Each round re-computes the update given the current state of both layers.
    Edits built from edit_loader; recovery evaluated on eval_loader only.

    Returns (edited_model, recovery_per_iter).
    """
    model_copy = deepcopy(model)
    recovery_per_iter = []

    baseline_acc = evaluate_class_accuracy(model, eval_loader, src_class, device)

    for i in range(n_iters):
        delta2, _, _, _ = compute_rank1_edit(model_copy, edit_loader, device, tgt_class, 'fc2')
        if delta2 is not None:
            apply_rank1_edit(model_copy, delta2, 'fc2')
        delta1, _, _, _ = compute_rank1_edit(model_copy, edit_loader, device, tgt_class, 'fc1')
        if delta1 is not None:
            apply_rank1_edit(model_copy, delta1, 'fc1')

        acc = evaluate_class_accuracy(model_copy, eval_loader, src_class, device)
        recovery_per_iter.append(acc - baseline_acc)

    return model_copy, recovery_per_iter


def run_multilayer_rome_experiment(
    model,
    edit_loader,
    eval_loader,
    corruption_configs: list,
    device: str = 'cpu',
) -> dict:
    """
    Run sequential and joint multi-layer edits for each corruption config.
    Compare to single-layer baseline. All edits built from edit_loader;
    all recovery measured on eval_loader only.
    """
    results = {}

    for cfg in corruption_configs:
        src, tgt, label = cfg['source'], cfg['target'], cfg['label']
        print(f"\n  Corruption config: {label}")
        key = label.replace('→', '_to_')

        # Single-layer (fc2 only) baseline
        base_acc = evaluate_class_accuracy(model, eval_loader, src, device)
        delta_single, _, _, _ = compute_rank1_edit(model, edit_loader, device, tgt, 'fc2')
        if delta_single is not None:
            model_temp = deepcopy(model)
            apply_rank1_edit(model_temp, delta_single, 'fc2')
            single_acc = evaluate_class_accuracy(model_temp, eval_loader, src, device)
            single_rec = single_acc - base_acc
        else:
            single_rec = 0.0

        # Sequential multi-layer
        m_seq, _ = sequential_multilayer_rome(model, edit_loader, src, tgt,
                                               layers=['fc2', 'fc1'], device=device)
        seq_acc = evaluate_class_accuracy(m_seq, eval_loader, src, device)
        seq_rec = seq_acc - base_acc

        # Joint multi-layer (3 iterations)
        m_joint, iter_recoveries = joint_multilayer_rome(model, edit_loader, eval_loader, src, tgt,
                                                          n_iters=3, device=device)
        joint_rec = iter_recoveries[-1] if iter_recoveries else 0.0

        print(f"    Single-layer: {single_rec:+.4f}  Sequential: {seq_rec:+.4f}  "
              f"Joint (3 iter): {joint_rec:+.4f}")
        print(f"    Iter recoveries: {[f'{r:+.4f}' for r in iter_recoveries]}")

        results[key] = {
            'single_layer_recovery': single_rec,
            'sequential_recovery': seq_rec,
            'joint_recovery': joint_rec,
            'joint_iter_recoveries': iter_recoveries,
        }

    return results


def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description='Multi-layer ROME')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/targeted_corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/multilayer_rome')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:5])
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disjoint edit/eval split of the test set
    test_dataset = get_test_dataset()
    edit_loader, eval_loader, split = build_edit_eval_loaders(
        test_dataset, split_seed=42, batch_size=config['training']['batch_size']
    )
    split.save(output_dir / 'split_provenance.json')
    print(f"Edit/eval split: {len(split.edit_indices)} edit / {len(split.eval_indices)} eval (disjoint)")

    corruption_configs = config.get('phase4', {}).get('corruption_configs', CORRUPTION_CONFIGS)

    all_results = {}
    for cfg in corruption_configs:
        label = cfg['label']
        key = label.replace('→', '_to_')
        all_results[key] = {
            'single_layer_recovery': [],
            'sequential_recovery': [],
            'joint_recovery': [],
        }

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        for cfg in corruption_configs:
            src, tgt, label = cfg['source'], cfg['target'], cfg['label']
            key = label.replace('→', '_to_')
            ckpt_path = Path(args.checkpoint_dir) / f"src{src}_tgt{tgt}" / f"seed_{seed}" / 'final_model.pt'
            if not ckpt_path.exists():
                print(f"  Checkpoint not found: {ckpt_path}")
                continue

            model = MNISTNet(
                config['model']['input_dim'],
                config['model']['hidden_dim'],
                config['model']['output_dim'],
            ).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            result = run_multilayer_rome_experiment(
                model, edit_loader, eval_loader, [cfg], device=device
            )
            all_results[key]['single_layer_recovery'].append(result[key]['single_layer_recovery'])
            all_results[key]['sequential_recovery'].append(result[key]['sequential_recovery'])
            all_results[key]['joint_recovery'].append(result[key]['joint_recovery'])

    # Summary
    print("\n=== MULTI-LAYER RANK-ONE EDIT SUMMARY (EVAL split only) ===")
    header = f"{'Config':>8} {'Single':>12} {'Sequential':>14} {'Joint':>12}"
    print(header)
    print('-' * len(header))
    for key in all_results:
        r = all_results[key]
        s_mean, s_lo, s_hi = compute_ci(r['single_layer_recovery'])
        seq_mean, seq_lo, seq_hi = compute_ci(r['sequential_recovery'])
        j_mean, j_lo, j_hi = compute_ci(r['joint_recovery'])
        print(f"{key:>8}  {s_mean:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}]  "
              f"{seq_mean:+.4f} [{seq_lo:+.4f}, {seq_hi:+.4f}]  "
              f"{j_mean:+.4f} [{j_lo:+.4f}, {j_hi:+.4f}]")

    with open(output_dir / 'multilayer_rome_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
