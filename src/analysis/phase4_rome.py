"""
Phase 4: Class-mean rank-one delta-norm analysis (legacy exploratory metric).

NOTE ON TERMINOLOGY (v2): this module computes a ROME-INSPIRED closed-form
rank-one edit statistic adapted to a shallow classifier. It is NOT the
original ROME algorithm of Meng et al. (2022), which relies on causal
tracing, key/value covariance statistics, and a preservation-constrained
objective defined for transformer MLP blocks.

What is computed here, for each class c and layer W:
  1. u = mean activation of the layer input for class-c examples (EDIT split)
  2. v = desired layer output for class c
       - fc2: one-hot target class vector
       - fc1: mean post-activation representation of class-c examples
  3. delta = ((v - W u) u^T) / (u.u + eps)
  4. delta_norm = ||delta||_F  (the reported statistic)

Protocol (v2): edit directions are built ONLY from the EDIT half of the
stratified test split; nothing in this module evaluates on the EVAL half.
Split provenance is saved to the output directory.

The multiclass_rome.py module performs the causal recovery experiments;
this module reports the delta-norm scaling statistics (clean vs corrupted,
noise-rate dose-response).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.model import MNISTNet
from utils.stats import compute_ci, SEEDS
from data.splits import SplitProvenance
from data.corruption import corrupt_labels_random
from multiclass_rome import (
    compute_rank1_edit,
    get_test_dataset,
    build_edit_eval_loaders,
)


def compute_delta_norms(model, edit_loader, device, layer='fc2'):
    """Rank-one delta-norm statistic for all classes, EDIT split only.

    Returns {class: {'delta_norm': float, 'effect_on_target': float}} where
    effect_on_target = (delta @ u).sum(), the raw change in the class logit
    direction under the edit (descriptive only, not a recovery result).
    """
    results = {}
    for c in range(model.output_dim):
        delta, u, v, used_layer = compute_rank1_edit(model, edit_loader, device, c, layer)
        if delta is None:
            continue
        with torch.no_grad():
            effect = (delta @ u).sum().item()
        results[c] = {
            'delta_norm': float(delta.norm(p='fro').item()),
            'effect_on_target': effect,
            'layer': used_layer,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 4: rank-one delta-norm analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase4')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds (defaults to config seeds; never range(n))')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--layer', type=str, default='fc2', choices=['fc1', 'fc2'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seeds = args.seeds if args.seeds is not None else SEEDS[:config.get('n_primary_seeds', 10)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # EDIT split only (disjoint from EVAL; nothing here touches EVAL)
    test_dataset = get_test_dataset()
    edit_loader, eval_loader, split = build_edit_eval_loaders(
        test_dataset, split_seed=args.split_seed,
        batch_size=config['training']['batch_size']
    )
    split.save(output_dir / 'split_provenance.json')
    assert set(split.edit_indices).isdisjoint(split.eval_indices)

    all_results = {str(seed): {} for seed in seeds}

    for seed in seeds:
        checkpoint_path = Path(args.checkpoint_dir) / f"seed_{seed}" / 'final_model.pt'
        if not checkpoint_path.exists():
            print(f"Checkpoint not found for seed {seed}")
            continue

        model = MNISTNet(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            activation=config['model']['activation']
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        fc1_results = compute_delta_norms(model, edit_loader, device, 'fc1')
        fc2_results = compute_delta_norms(model, edit_loader, device, 'fc2')

        all_results[str(seed)] = {'fc1': fc1_results, 'fc2': fc2_results}

        fc1_norms = [r['delta_norm'] for r in fc1_results.values()]
        fc2_norms = [r['delta_norm'] for r in fc2_results.values()]
        print(f"Seed {seed}: FC1 mean delta-norm={np.mean(fc1_norms):.4f}, "
              f"FC2 mean delta-norm={np.mean(fc2_norms):.4f}")

    print(f"\n=== Aggregated rank-one delta-norm results ===")
    for layer in ['fc1', 'fc2']:
        print(f"\n  {layer}:")
        for c in range(10):
            vals = [all_results[str(s)][layer][c]['delta_norm']
                    for s in seeds
                    if all_results.get(str(s)) and c in all_results[str(s)][layer]]
            if vals:
                mean, lo, hi = compute_ci(vals)
                print(f"    Class {c}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    with open(output_dir / 'phase4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / 'run_manifest.json', 'w') as f:
        json.dump({
            'seeds': seeds,
            'split_seed': args.split_seed,
            'layer_default': args.layer,
            'edit_n': len(split.edit_indices),
            'eval_n': len(split.eval_indices),
            'method': 'class-mean rank-one closed-form edit (ROME-inspired; not Meng et al. 2022)',
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
