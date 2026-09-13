"""
Multi-class rank-one intervention validation using pre-trained corrupted checkpoints.

Method note (v2): this is NOT the original ROME algorithm of Meng et al. (2022)
(no causal tracing, no key/value covariance constraint, no preservation term).
It is a ROME-INSPIRED closed-form rank-one edit adapted to a shallow classifier:
delta = ((v - W u) u^T) / (u.u + eps). All functions below are named rank1_*
to avoid misattribution.

Protocol note (v2): edit directions are constructed from an EDIT loader drawn
from a disjoint half of the test set; final evaluation uses only the EVAL half.
Split indices are saved to split_provenance.json in the output dir.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json

from copy import deepcopy
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.model import MNISTNet
from utils.stats import compute_ci, SEEDS
from utils.metrics import evaluate_class_accuracy
from data.splits import split_edit_eval, SplitProvenance


CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
    {'source': 5, 'target': 6, 'label': '5→6'},
    {'source': 0, 'target': 8, 'label': '0→8'},
]


def _get_hidden_activations(model, data, device):
    """Extract hidden (fc1_post) activations for ROME on fc2."""
    model.eval()
    with torch.no_grad():
        out = model.forward_with_all_layers(data)
        return out['fc1_post_activation']


def _get_input_activations(model, data, device):
    """Extract flattened input activations for ROME on fc1."""
    with torch.no_grad():
        return data.view(data.size(0), -1)


def compute_rank1_edit(model, edit_loader, device, target_class, layer='fc2'):
    """Compute rank-one edit for a specified layer, from EDIT-set examples only.

    For fc2: u = mean hidden activation, v = one-hot target output.
    For fc1: u = mean input activation, v = mean target-class hidden
              activation (shifts source inputs toward target-like hidden reps).

    Returns:
        (delta, u, v, layer)
    """
    model.eval()

    if layer == 'fc2':
        target_key = []
        with torch.no_grad():
            for data, target in edit_loader:
                data = data.to(device)
                key = _get_hidden_activations(model, data, device)
                mask = target == target_class
                if mask.any():
                    target_key.append(key[mask].cpu())

        if not target_key:
            return None, None, None, layer

        u = torch.cat(target_key).mean(0).to(device)
        v = torch.zeros(model.output_dim, device=device)
        v[target_class] = 1.0
        W = model.fc2.weight.data

    elif layer == 'fc1':
        target_inputs, target_hiddens = [], []
        with torch.no_grad():
            for data, target in edit_loader:
                data = data.to(device)
                inp = _get_input_activations(model, data, device)
                hidden = _get_hidden_activations(model, data, device)
                mask = target == target_class
                if mask.any():
                    target_inputs.append(inp[mask].cpu())
                    target_hiddens.append(hidden[mask].cpu())

        if not target_inputs:
            return None, None, None, layer

        u = torch.cat(target_inputs).mean(0).to(device)
        v = torch.cat(target_hiddens).mean(0).to(device)
        W = model.fc1.weight.data

    else:
        raise ValueError(f"Unknown layer: {layer}")

    # Rank-one closed-form update (adapting the rank-one principle of ROME,
    # Meng et al. 2022, to a shallow classifier; NOT the original algorithm)
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    return delta, u, v, layer


def apply_rank1_edit(model, delta, layer='fc2'):
    """Apply precomputed rank-one edit to specified layer."""
    with torch.no_grad():
        if layer == 'fc2':
            model.fc2.weight.data += delta
        elif layer == 'fc1':
            model.fc1.weight.data += delta


def apply_random_edit(model, layer_name: str, delta_norm: float, seed: int = None):
    """
    Apply a random rank-one edit of exactly delta_norm Frobenius norm
    to the named weight matrix. Used as null baseline for ROME.

    Args:
        model:       trained MNISTNet (or CIFARNet)
        layer_name:  'fc1' or 'fc2' or 'fc3'
        delta_norm:  target Frobenius norm of the perturbation (match ROME's delta)
        seed:        random seed for reproducibility

    Returns:
        edited model (deep copy, original unchanged)
    """
    if seed is not None:
        torch.manual_seed(seed)

    model_copy = deepcopy(model)
    W = getattr(model_copy, layer_name).weight.data

    u = torch.randn(W.shape[1])
    v = torch.randn(W.shape[0])
    u = u / u.norm()
    v = v / v.norm()

    delta = delta_norm * torch.outer(v, u)
    W.add_(delta)

    return model_copy


def run_rank1_with_random_baseline(
    model,
    edit_loader,
    eval_loader,
    src_class: int,
    tgt_class: int,
    layer_name: str = 'fc2',
    n_random_trials: int = 20,
    device: str = 'cpu'
) -> dict:
    """
    Run rank-one edit (built from edit_loader) and compare against random
    rank-one baselines of matched Frobenius norm, both evaluated on eval_loader.

    Returns dict with recovery stats and standardized effect (z) against the
    random null. Signal ratio is reported only when the null mean is nonzero;
    'inf' ratios are not used.
    """
    baseline_acc = evaluate_class_accuracy(model, eval_loader, src_class, device)

    # Rank-one edit (constructed from EDIT split only)
    delta, u, v, used_layer = compute_rank1_edit(model, edit_loader, device, tgt_class, layer_name)
    if delta is None:
        return {"error": "rank-one edit failed"}
    delta_norm = delta.norm(p='fro').item()

    W_orig = getattr(model, used_layer).weight.data.clone()
    apply_rank1_edit(model, delta, used_layer)
    edited_acc = evaluate_class_accuracy(model, eval_loader, src_class, device)
    getattr(model, used_layer).weight.data.copy_(W_orig)
    rank1_recovery = edited_acc - baseline_acc

    # Random baseline: norm-matched random rank-one edits, evaluated on EVAL split
    random_recoveries = []
    for trial in range(n_random_trials):
        model_rand = apply_random_edit(model, used_layer, delta_norm, seed=trial)
        rand_acc = evaluate_class_accuracy(model_rand, eval_loader, src_class, device)
        random_recoveries.append(rand_acc - baseline_acc)

    random_mean = float(np.mean(random_recoveries))
    random_std = float(np.std(random_recoveries))
    if abs(random_std) > 1e-9:
        z_score = (rank1_recovery - random_mean) / random_std
        # one-sided empirical null p-value (at least one extreme trial below)
        p_null = (1 + np.sum(np.array(random_recoveries) >= rank1_recovery)) / (n_random_trials + 1)
    else:
        z_score = None
        p_null = None

    return {
        "rank1_recovery": rank1_recovery,
        "random_recovery_mean": random_mean,
        "random_recovery_std": random_std,
        "random_recovery_trials": random_recoveries,
        "delta_norm": delta_norm,
        "z_vs_null": z_score,
        "p_null_empirical": p_null,
        "n_random_trials": n_random_trials,
    }


def find_broken_class(model, test_loader, device):
    """Find the class with lowest accuracy in corrupted model."""
    accs = {}
    for c in range(10):
        accs[c] = evaluate_class_accuracy(model, test_loader, c, device)
    worst = min(accs, key=accs.get)
    return worst, accs


def run_rank1_experiment(model, edit_loader, eval_loader, device, target_class, layer='fc2'):
    """Run rank-one edit (built from edit_loader) and evaluate on eval_loader only."""
    pre_accs = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}

    delta, u, v, used_layer = compute_rank1_edit(model, edit_loader, device, target_class, layer)
    if delta is None:
        return 0.0, 0.0, 0.0, pre_accs, {}

    weight_attr = 'fc2.weight' if used_layer == 'fc2' else 'fc1.weight'
    W_orig = getattr(model, weight_attr.split('.')[0]).weight.data.clone()
    apply_rank1_edit(model, delta, used_layer)

    post_accs = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}

    recovery = post_accs[target_class] - pre_accs[target_class]
    other_classes = [c for c in range(10) if c != target_class]
    side_effects = float(np.mean([abs(post_accs[c] - pre_accs[c]) for c in other_classes]))
    magnitude = (getattr(model, weight_attr.split('.')[0]).weight.data - W_orig).norm(p='fro').item()

    getattr(model, weight_attr.split('.')[0]).weight.data.copy_(W_orig)

    return recovery, side_effects, magnitude, pre_accs, {'layer': used_layer, 'delta_norm': delta.norm().item(), 'post_accs': post_accs}


def run_multi_layer_rank1(model, edit_loader, eval_loader, device, target_class):
    """Apply rank-one edits to fc2, then fc1 sequentially; compare to single-layer.

    Edit construction uses edit_loader; all accuracy evaluation uses eval_loader.
    """
    results = {}

    # fc2-only
    r_fc2, s_fc2, m_fc2, pre, meta2 = run_rank1_experiment(model, edit_loader, eval_loader, device, target_class, 'fc2')
    results['fc2_only'] = {'recovery': r_fc2, 'side_effects': s_fc2, 'magnitude': m_fc2}

    # fc1-only
    r_fc1, s_fc1, m_fc1, pre, meta1 = run_rank1_experiment(model, edit_loader, eval_loader, device, target_class, 'fc1')
    results['fc1_only'] = {'recovery': r_fc1, 'side_effects': s_fc1, 'magnitude': m_fc1}

    # Both layers (fc2 then fc1 sequentially)
    delta2, u2, v2, _ = compute_rank1_edit(model, edit_loader, device, target_class, 'fc2')
    W2_orig = model.fc2.weight.data.clone()
    apply_rank1_edit(model, delta2, 'fc2')
    delta1, u1, v1, _ = compute_rank1_edit(model, edit_loader, device, target_class, 'fc1')
    W1_orig = model.fc1.weight.data.clone()
    apply_rank1_edit(model, delta1, 'fc1')
    post_both = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}
    model.fc1.weight.data.copy_(W1_orig)
    model.fc2.weight.data.copy_(W2_orig)
    results['both_layers'] = {
        'recovery': post_both[target_class] - pre[target_class],
        'side_effects': float(np.mean([abs(post_both[c] - pre[c]) for c in range(10) if c != target_class])),
        'magnitude': (delta2.norm().item() + delta1.norm().item())
    }

    return results


def get_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST('./data', train=False, download=True, transform=transform)


def build_edit_eval_loaders(dataset, split_seed=42, batch_size=128):
    """
    Stratified edit/eval split of the test set: each class contributes half
    of its examples to EDIT and half to EVAL, guaranteeing both splits
    contain every class while remaining globally disjoint.
    """
    targets = np.array(dataset.targets)
    edit_idx, eval_idx = [], []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        cls_split = split_edit_eval(cls_idx, eval_fraction=0.5, seed=split_seed, split_name=f'class_{c}')
        edit_idx.extend(cls_split.edit_indices)
        eval_idx.extend(cls_split.eval_indices)

    split = SplitProvenance(
        split_name='test_edit_eval_stratified',
        edit_indices=edit_idx,
        eval_indices=eval_idx,
        seed=split_seed,
    )
    assert set(edit_idx).isdisjoint(eval_idx)
    assert len(edit_idx) + len(eval_idx) == len(targets)

    from torch.utils.data import Subset
    edit_loader = DataLoader(Subset(dataset, split.edit_indices),
                             batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(Subset(dataset, split.eval_indices),
                             batch_size=batch_size, shuffle=False)
    return edit_loader, eval_loader, split


def main():
    parser = argparse.ArgumentParser(description='Multi-class rank-one intervention validation')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/targeted_corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/multiclass_rome')
    parser.add_argument('--clean-dir', type=str, default='outputs/clean')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:3])
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Seed for the edit/eval test-set partition (fixed across all runs)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disjoint edit/eval split of the test set (stratified by class)
    test_dataset = get_test_dataset()
    edit_loader, eval_loader, split = build_edit_eval_loaders(
        test_dataset, split_seed=args.split_seed, batch_size=config['training']['batch_size']
    )
    split.save(output_dir / 'split_provenance.json')
    print(f"Edit/eval split: {len(split.edit_indices)} edit / {len(split.eval_indices)} eval "
          f"(disjoint, stratified, seed {args.split_seed})")

    corruption_configs = config.get('phase4', {}).get('corruption_configs', CORRUPTION_CONFIGS)

    # Run rank-one edits on each corruption config
    all_results = {}
    multi_results = {}
    random_baseline_results = {}
    for cfg in corruption_configs:
        label = cfg['label']
        target = cfg['source']
        print(f"\n=== Corruption config: {label} ===")
        all_results[label] = {'recovery': [], 'side_effects': [], 'magnitude': [],
                              'pre_accs': [], 'post_accs': []}
        multi_results[label] = {'fc2_only': {'recovery': [], 'side_effects': [], 'magnitude': []},
                                'fc1_only': {'recovery': [], 'side_effects': [], 'magnitude': []},
                                'both_layers': {'recovery': [], 'side_effects': [], 'magnitude': []}}
        random_baseline_results[label] = {'rank1_recovery': [], 'random_mean': [],
                                          'random_std': [], 'z_vs_null': [],
                                          'p_null_empirical': []}

        for seed in args.seeds:
            ckpt_path = Path(args.checkpoint_dir) / f"src{cfg['source']}_tgt{cfg['target']}" / f"seed_{seed}" / 'final_model.pt'
            if not ckpt_path.exists():
                print(f"  Checkpoint not found: {ckpt_path}")
                continue

            model = MNISTNet(
                config['model']['input_dim'],
                config['model']['hidden_dim'],
                config['model']['output_dim']
            ).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # fc2-only (edit built from EDIT split, evaluated on EVAL split)
            recovery, side_effects, magnitude, pre_accs, meta = run_rank1_experiment(
                model, edit_loader, eval_loader, device, target, 'fc2'
            )
            all_results[label]['recovery'].append(recovery)
            all_results[label]['side_effects'].append(side_effects)
            all_results[label]['magnitude'].append(magnitude)
            if 'post_accs' in meta:
                all_results[label]['post_accs'].append(meta['post_accs'])

            # Multi-layer comparison (fc1-only, fc2-only, both)
            layer_results = run_multi_layer_rank1(model, edit_loader, eval_loader, device, target)
            for k in ['fc2_only', 'fc1_only', 'both_layers']:
                multi_results[label][k]['recovery'].append(layer_results[k]['recovery'])
                multi_results[label][k]['side_effects'].append(layer_results[k]['side_effects'])
                multi_results[label][k]['magnitude'].append(layer_results[k]['magnitude'])

            # Random baseline comparison (norm-matched null, EVAL split)
            random_baseline = run_rank1_with_random_baseline(
                deepcopy(model), edit_loader, eval_loader, target, target,
                layer_name='fc2', n_random_trials=20, device=device
            )
            if 'error' not in random_baseline:
                random_baseline_results[label]['rank1_recovery'].append(random_baseline['rank1_recovery'])
                random_baseline_results[label]['random_mean'].append(random_baseline['random_recovery_mean'])
                random_baseline_results[label]['random_std'].append(random_baseline['random_recovery_std'])
                if random_baseline['z_vs_null'] is not None:
                    random_baseline_results[label]['z_vs_null'].append(random_baseline['z_vs_null'])
                    random_baseline_results[label]['p_null_empirical'].append(random_baseline['p_null_empirical'])

            print(f"  seed={seed} fc2={layer_results['fc2_only']['recovery']:+.4f} "
                  f"fc1={layer_results['fc1_only']['recovery']:+.4f} "
                  f"both={layer_results['both_layers']['recovery']:+.4f} "
                  f"z_null={random_baseline.get('z_vs_null')}")

    # Summary
    print("\n=== MULTI-CLASS RANK-ONE INTERVENTION SUMMARY (fc2-only, EVAL split only) ===")
    header = f"{'Config':>8} {'Recovery':>18} {'Side Effects':>16} {'Magnitude':>12}"
    print(header)
    print('-' * len(header))
    for label in all_results:
        r_mean, r_lo, r_hi = compute_ci(all_results[label]['recovery'])
        s_mean, s_lo, s_hi = compute_ci(all_results[label]['side_effects'])
        m_mean, m_lo, m_hi = compute_ci(all_results[label]['magnitude'])
        print(f"{label:>8}  {r_mean:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]  "
              f"{s_mean:.4f} [{s_lo:.4f}, {s_hi:.4f}]  "
              f"{m_mean:.4f} [{m_lo:.4f}, {m_hi:.4f}]")

    print("\n=== MULTI-LAYER RANK-ONE COMPARISON (EVAL split only) ===")
    for k in ['fc2_only', 'fc1_only', 'both_layers']:
        print(f"\n  {k}:")
        hl = f"{'Config':>8} {'Recovery':>18} {'Side Effects':>16} {'Magnitude':>12}"
        print(f"    {hl}")
        print(f"    {'-' * len(hl)}")
        for label in multi_results:
            r_mean, r_lo, r_hi = compute_ci(multi_results[label][k]['recovery'])
            s_mean, s_lo, s_hi = compute_ci(multi_results[label][k]['side_effects'])
            m_mean, m_lo, m_hi = compute_ci(multi_results[label][k]['magnitude'])
            print(f"    {label:>8}  {r_mean:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]  "
                  f"{s_mean:.4f} [{s_lo:.4f}, {s_hi:.4f}]  "
                  f"{m_mean:.4f} [{m_lo:.4f}, {m_hi:.4f}]")

    # Random baseline summary (no 'inf' ratios; z-scores and empirical p-values)
    print("\n=== RANDOM-NULL BASELINE SUMMARY (fc2, EVAL split) ===")
    header = f"{'Config':>8} {'Rank-1 Rec':>12} {'Random Rec':>16} {'z vs null':>12} {'p (empirical)':>14}"
    print(header)
    print('-' * len(header))
    for label in random_baseline_results:
        r = random_baseline_results[label]
        if r['rank1_recovery']:
            r1_mean = np.mean(r['rank1_recovery'])
            rand_mean = np.mean(r['random_mean'])
            rand_std = np.mean(r['random_std'])
            z_vals = r['z_vs_null']
            z_mean = np.mean(z_vals) if z_vals else float('nan')
            p_vals = r['p_null_empirical']
            p_worst = max(p_vals) if p_vals else float('nan')
            print(f"{label:>8}  {r1_mean:+.4f}     {rand_mean:+.4f}±{rand_std:.4f}  "
                  f"{z_mean:+.2f}       {p_worst:.3f}")
        else:
            print(f"{label:>8}  (no data)")

    with open(output_dir / 'multiclass_rome_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / 'multilayer_rome_comparison.json', 'w') as f:
        json.dump(multi_results, f, indent=2, default=str)
    with open(output_dir / 'random_baseline_results.json', 'w') as f:
        json.dump(random_baseline_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()