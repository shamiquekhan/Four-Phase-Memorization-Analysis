"""
CIFAR-10 replication (v2 protocol): three claims under the fixed methodology.

  1. Cross-model CKA drift: 1 - CKA(clean_layer, corrupted_layer) on identical
     inputs, plus cross-seed noise-floor controls (v0 measured within-model
     layer similarity instead, which is a different quantity).
  2. Rank-one intervention: edit directions built ONLY from the EDIT half of
     a fixed stratified test split; recovery measured ONLY on the disjoint
     EVAL half (v0 built and evaluated on the same full test set).
  3. Rank-ablation gap on fc3, evaluated on the EVAL half only.

Method note: the "rank-one edit" is a ROME-inspired closed-form update
adapted to a shallow MLP; it is NOT the original ROME algorithm of
Meng et al. (2022).
"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.model import CIFAR10MLP
from models.cifarnet import CIFARNet
from utils.metrics import linear_cka, evaluate_class_accuracy
from utils.stats import compute_ci, paired_t_test, SEEDS
from analysis.rank_ablation import _get_weight, _set_weight
from data.splits import split_edit_eval, SplitProvenance

CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
]

CKA_LAYERS = ['input', 'fc1_pre_activation', 'fc1_post_activation',
              'fc2_pre_activation', 'fc2_post_activation', 'output']


def get_cifar10_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    return datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)


def build_edit_eval_loaders(split_seed=42, batch_size=256):
    """Stratified disjoint EDIT/EVAL split of the CIFAR-10 test set."""
    ds = get_cifar10_test_dataset()
    targets = np.array(ds.targets)
    edit_idx, eval_idx = [], []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        sp = split_edit_eval(cls_idx, eval_fraction=0.5, seed=split_seed,
                             split_name=f'class_{c}')
        edit_idx.extend(sp.edit_indices)
        eval_idx.extend(sp.eval_indices)
    split = SplitProvenance('cifar10_test_edit_eval_stratified', edit_idx, eval_idx,
                            seed=split_seed)
    assert set(edit_idx).isdisjoint(eval_idx)
    edit_loader = DataLoader(Subset(ds, split.edit_indices), batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(Subset(ds, split.eval_indices), batch_size=batch_size, shuffle=False)
    return edit_loader, eval_loader, split


def get_drift_loader(batch_size=256):
    """Deterministic loader over the FULL test set for CKA drift analysis
    (representation measurement, not intervention evaluation)."""
    ds = get_cifar10_test_dataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_activations(model, dataloader, device, max_samples=5000):
    model.eval()
    all_layers = {k: [] for k in CKA_LAYERS}
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            for k in all_layers:
                all_layers[k].append(out[k].cpu())
            if sum(x.size(0) for x in all_layers['input']) >= max_samples:
                break
    return {k: torch.cat(v)[:max_samples] for k, v in all_layers.items()}


def cross_model_drift(model_a, model_b, loader, device):
    """1 - CKA between two models' layer activations on identical inputs."""
    acts_a = get_activations(model_a, loader, device)
    acts_b = get_activations(model_b, loader, device)
    return {layer: 1.0 - linear_cka(acts_a[layer], acts_b[layer])
            for layer in CKA_LAYERS}


def run_cka_drift(clean_models, corrupted_models, loader, device):
    """v2 primary: cross-model drift + seed noise-floor controls."""
    print("\n=== CIFAR-10 v2: Cross-Model CKA Drift (clean <-> corrupted, 0.2) ===")
    import itertools
    results = {'paired_drift': {}, 'seed_floors': {}}

    paired = {}
    for seed in sorted(set(clean_models) & set(corrupted_models)):
        paired[seed] = cross_model_drift(clean_models[seed], corrupted_models[seed], loader, device)
    if paired:
        print("  Paired clean<->corrupted drift per layer:")
        for layer in CKA_LAYERS:
            vals = [paired[s][layer] for s in sorted(paired)]
            m, lo, hi = compute_ci(vals)
            print(f"    drift({layer:22s}): {m:.4f} [{lo:.4f}, {hi:.4f}]")
            results['paired_drift'][layer] = {'mean': m, 'ci_lo': lo, 'ci_hi': hi}

    for name, models in [('clean_clean', clean_models), ('corrupted_corrupted', corrupted_models)]:
        floors = []
        for a, b in itertools.combinations(sorted(models), 2):
            floors.append(cross_model_drift(models[a], models[b], loader, device))
        if floors:
            print(f"  Seed noise floor ({name}):")
            results['seed_floors'][name] = {}
            for layer in CKA_LAYERS:
                vals = [f[layer] for f in floors]
                m, lo, hi = compute_ci(vals)
                print(f"    {layer:22s}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
                results['seed_floors'][name][layer] = m
    return results


def _compute_rank1_edit_cifar(model, edit_loader, device, target_class, layer='fc3'):
    """ROME-inspired closed-form rank-one edit (EDIT loader only)."""
    model.eval()
    target_key = []
    with torch.no_grad():
        for data, target in edit_loader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            key = out['fc2_post_activation']
            mask = target == target_class
            if mask.any():
                target_key.append(key[mask].cpu())
    if not target_key:
        return None, None, None, layer
    u = torch.cat(target_key).mean(0).to(device)
    v = torch.zeros(model.output_dim, device=device)
    v[target_class] = 1.0
    W = model.fc3.weight.data
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    return delta, u, v, layer


def run_rank1_replication(clean_models, corrupted_models, edit_loader, eval_loader, device):
    """v2: edits from EDIT split; recovery on EVAL split only."""
    print("\n=== CIFAR-10 v2: Rank-One Intervention (EDIT/EVAL firewall) ===")
    results = {}
    for cfg in CORRUPTION_CONFIGS:
        src, tgt, label = cfg['source'], cfg['target'], cfg['label']
        clean_recs, corr_recs, corr_side = [], [], []
        for seed in sorted(corrupted_models):
            model = corrupted_models[seed]
            baseline = evaluate_class_accuracy(model, eval_loader, src, device)
            pre = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}
            delta, _, _, _ = _compute_rank1_edit_cifar(model, edit_loader, device, tgt, 'fc3')
            if delta is None:
                continue
            W_orig = model.fc3.weight.data.clone()
            model.fc3.weight.data += delta
            post = evaluate_class_accuracy(model, eval_loader, src, device)
            post_all = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}
            model.fc3.weight.data.copy_(W_orig)
            corr_recs.append(post - baseline)
            corr_side.append(float(np.mean([abs(post_all[c] - pre[c]) for c in range(10) if c != src])))
        for seed in sorted(clean_models):
            model = clean_models[seed]
            baseline = evaluate_class_accuracy(model, eval_loader, src, device)
            delta, _, _, _ = _compute_rank1_edit_cifar(model, edit_loader, device, tgt, 'fc3')
            if delta is None:
                continue
            W_orig = model.fc3.weight.data.clone()
            model.fc3.weight.data += delta
            post = evaluate_class_accuracy(model, eval_loader, src, device)
            model.fc3.weight.data.copy_(W_orig)
            clean_recs.append(post - baseline)
        if clean_recs and corr_recs:
            cm, clo, chi = compute_ci(clean_recs)
            rm, rlo, rhi = compute_ci(corr_recs)
            sm, slo, shi = compute_ci(corr_side)
            _, p_val = paired_t_test(corr_recs, clean_recs)
            print(f"  {label:>6}  Clean: {cm:+.4f} [{clo:+.4f}, {chi:+.4f}]  "
                  f"Corrupted: {rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]  "
                  f"side-effects: {sm:.4f}  p={p_val:.4f}")
            results[label] = {'clean_recovery': cm, 'corrupted_recovery': rm,
                              'corrupted_side_effects': sm, 'p': p_val,
                              'n_clean': len(clean_recs), 'n_corr': len(corr_recs)}
    return results


def run_rank_ablation_replication(clean_models, corrupted_models, eval_loader, device):
    """v2: rank-k truncation of fc3, evaluated on EVAL half only."""
    print("\n=== CIFAR-10 v2: Rank Ablation (fc3, EVAL split only) ===")
    results = {}
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
        clean_accs, corr_accs = [], []
        for models, accs in [(clean_models, clean_accs), (corrupted_models, corr_accs)]:
            for seed in sorted(models):
                model = models[seed]
                W_orig = _get_weight(model, 'fc3').clone()
                U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
                k_eff = min(k, len(S))
                W_k = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]
                _set_weight(model, 'fc3', W_k)
                correct = total = 0
                with torch.no_grad():
                    for x, y in eval_loader:
                        x, y = x.to(device), y.to(device)
                        correct += (model(x).argmax(1) == y).sum().item()
                        total += len(y)
                accs.append(100. * correct / total)
                _set_weight(model, 'fc3', W_orig)
        if clean_accs and corr_accs:
            cm, clo, chi = compute_ci(clean_accs)
            rm, rlo, rhi = compute_ci(corr_accs)
            gap = cm - rm
            print(f"  Rank {k:2d}:  Clean={cm:.2f}% [{clo:.2f}, {chi:.2f}]  "
                  f"Corrupted={rm:.2f}% [{rlo:.2f}, {rhi:.2f}]  Gap={gap:.2f}pp")
            results[k] = {'clean_acc': cm, 'corrupted_acc': rm, 'gap': gap}
    return results


def load_model(ckpt_path: str, model_type: str, device):
    if model_type == 'cifar10mlp':
        return CIFAR10MLP.load_checkpoint(ckpt_path, device)
    elif model_type == 'cifarnet':
        return CIFARNet.load_checkpoint(ckpt_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 v2 replication')
    parser.add_argument('--clean-dir', type=str, default='outputs/cifar10/clean')
    parser.add_argument('--corrupted-dir', type=str, default='outputs/cifar10/corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/replication')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS[:5])
    parser.add_argument('--model-type', type=str, default='cifar10mlp', choices=['cifar10mlp', 'cifarnet'])
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edit_loader, eval_loader, split = build_edit_eval_loaders(split_seed=args.split_seed)
    split.save(output_dir / 'split_provenance.json')
    print(f"Edit/eval split: {len(split.edit_indices)} edit / {len(split.eval_indices)} eval "
          f"(disjoint, stratified, seed {args.split_seed})")

    drift_loader = get_drift_loader()

    clean_models, corrupted_models = {}, {}
    for seed in args.seeds:
        ckpt = Path(args.clean_dir) / f"seed_{seed}" / 'final_model.pt'
        if ckpt.exists():
            clean_models[seed] = load_model(str(ckpt), args.model_type, device)
        ckpt = Path(args.corrupted_dir) / f"noise_0.2" / f"seed_{seed}" / 'final_model.pt'
        if ckpt.exists():
            corrupted_models[seed] = load_model(str(ckpt), args.model_type, device)

    print(f"Loaded {len(clean_models)} clean / {len(corrupted_models)} corrupted models")

    results = {}
    results['cka_drift'] = run_cka_drift(clean_models, corrupted_models, drift_loader, device)
    results['rank1'] = run_rank1_replication(clean_models, corrupted_models, edit_loader, eval_loader, device)
    results['rank_ablation'] = run_rank_ablation_replication(clean_models, corrupted_models, eval_loader, device)

    with open(output_dir / 'cifar_replication_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
