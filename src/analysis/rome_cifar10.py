"""
CIFAR-10 ROME analysis: rank-one edit magnitude and multi-class recovery.
Replicates Phase 4 findings on CIFAR-10 with a 3-layer MLP.
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

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import CIFAR10MLP
from utils.stats import compute_ci, paired_t_test
from utils.metrics import evaluate_class_accuracy


def _get_hidden_activations(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model.forward_with_all_layers(data)
        return out['fc2_post_activation']


def compute_rank1_edit(model, dataloader, device, target_class, layer='fc3'):
    """ROME-inspired rank-one edit adapted for CIFAR-10 MLP (built from EDIT loader)."""
    model.eval()
    if layer == 'fc3':
        target_key = []
        with torch.no_grad():
            for data, target in dataloader:
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
        W = model.fc3.weight.data
    elif layer == 'fc2':
        target_inputs, target_hiddens = [], []
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                out = model.forward_with_all_layers(data)
                inp = out['fc1_post_activation']
                hidden = out['fc2_post_activation']
                mask = target == target_class
                if mask.any():
                    target_inputs.append(inp[mask].cpu())
                    target_hiddens.append(hidden[mask].cpu())
        if not target_inputs:
            return None, None, None, layer
        u = torch.cat(target_inputs).mean(0).to(device)
        v = torch.cat(target_hiddens).mean(0).to(device)
        W = model.fc2.weight.data
    else:
        raise ValueError(f"Unknown layer: {layer}")

    # Rank-one closed-form update (ROME-inspired adaptation, not the original algorithm)
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    return delta, u, v, layer


def apply_rank1_edit(model, delta, layer='fc3'):
    with torch.no_grad():
        if layer == 'fc3':
            model.fc3.weight.data += delta
        elif layer == 'fc2':
            model.fc2.weight.data += delta


def run_rank1_experiment(model, edit_loader, eval_loader, device, target_class, layer='fc3'):
    pre_accs = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}
    delta, u, v, used_layer = compute_rank1_edit(model, edit_loader, device, target_class, layer)
    if delta is None:
        return 0.0, 0.0, 0.0, pre_accs, {}

    if used_layer == 'fc3':
        W_orig = model.fc3.weight.data.clone()
    else:
        W_orig = model.fc2.weight.data.clone()
    apply_rank1_edit(model, delta, used_layer)
    post_accs = {c: evaluate_class_accuracy(model, eval_loader, c, device) for c in range(10)}
    recovery = post_accs[target_class] - pre_accs[target_class]
    other_classes = [c for c in range(10) if c != target_class]
    side_effects = float(np.mean([abs(post_accs[c] - pre_accs[c]) for c in other_classes]))

    if used_layer == 'fc3':
        magnitude = (model.fc3.weight.data - W_orig).norm(p='fro').item()
        model.fc3.weight.data.copy_(W_orig)
    else:
        magnitude = (model.fc2.weight.data - W_orig).norm(p='fro').item()
        model.fc2.weight.data.copy_(W_orig)

    return recovery, side_effects, magnitude, pre_accs, {'layer': used_layer, 'delta_norm': delta.norm().item()}


def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 rank-one intervention validation')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--clean-dir', type=str, default='outputs/cifar10/clean')
    parser.add_argument('--corrupted-dir', type=str, default='outputs/cifar10/corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/analysis/rome')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratified disjoint edit/eval split of the CIFAR-10 test set
    test_dataset = test_loader.dataset
    targets = np.array(test_dataset.targets)
    edit_idx, eval_idx = [], []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        rng = np.random.default_rng(42)
        perm = rng.permutation(cls_idx)
        half = len(perm) // 2
        eval_idx.extend(perm[:half].tolist())
        edit_idx.extend(perm[half:].tolist())
    assert set(edit_idx).isdisjoint(eval_idx)
    from data.splits import SplitProvenance
    split = SplitProvenance('cifar10_test_edit_eval_stratified', edit_idx, eval_idx, seed=42)
    split.save(output_dir / 'split_provenance.json')
    print(f"Edit/eval split: {len(edit_idx)} edit / {len(eval_idx)} eval (disjoint, stratified)")

    from torch.utils.data import Subset
    edit_loader = DataLoader(Subset(test_dataset, split.edit_indices), batch_size=128, shuffle=False)
    eval_loader = DataLoader(Subset(test_dataset, split.eval_indices), batch_size=128, shuffle=False)

    clean_results = {c: {'fc3_recovery': [], 'fc3_magnitude': []} for c in range(10)}
    corrupted_results = {c: {'fc3_recovery': [], 'fc3_magnitude': []} for c in range(10)}

    for seed in args.seeds:
        ckpt_clean = Path(args.clean_dir) / f"seed_{seed}" / 'final_model.pt'
        if ckpt_clean.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_clean), device)
            for c in range(10):
                rec, se, mag, pre, meta = run_rank1_experiment(model, edit_loader, eval_loader, device, c, 'fc3')
                clean_results[c]['fc3_recovery'].append(rec)
                clean_results[c]['fc3_magnitude'].append(mag)
            print(f"Clean seed={seed}: rank-one edits done")

        ckpt_corr = Path(args.corrupted_dir) / f"noise_0.2" / f"seed_{seed}" / 'final_model.pt'
        if ckpt_corr.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_corr), device)
            for c in range(10):
                rec, se, mag, pre, meta = run_rank1_experiment(model, edit_loader, eval_loader, device, c, 'fc3')
                corrupted_results[c]['fc3_recovery'].append(rec)
                corrupted_results[c]['fc3_magnitude'].append(mag)
            print(f"Corrupted seed={seed}: rank-one edits done")

    print("\n=== CIFAR-10 Rank-One Recovery: Clean vs Corrupted (fc3, EVAL split only) ===")
    for c in range(10):
        clean_rec = clean_results[c]['fc3_recovery']
        corr_rec = corrupted_results[c]['fc3_recovery']
        if clean_rec and corr_rec:
            cm, clo, chi = compute_ci(clean_rec)
            rm, rlo, rhi = compute_ci(corr_rec)
            t_stat, p_val = paired_t_test(clean_rec, corr_rec)
            print(f"Class {c}:  Clean={cm:+.4f} [{clo:+.4f}, {chi:+.4f}]  "
                  f"Corrupted={rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]  p={p_val:.4f}")

    print("\n=== CIFAR-10 ROME Delta-Norm: Clean vs Corrupted (fc3) ===")
    for c in range(10):
        clean_mag = clean_results[c]['fc3_magnitude']
        corr_mag = corrupted_results[c]['fc3_magnitude']
        if clean_mag and corr_mag:
            cm, clo, chi = compute_ci(clean_mag)
            rm, rlo, rhi = compute_ci(corr_mag)
            t_stat, p_val = paired_t_test(clean_mag, corr_mag)
            print(f"Class {c}:  Clean={cm:.4f} [{clo:.4f}, {chi:.4f}]  "
                  f"Corrupted={rm:.4f} [{rlo:.4f}, {rhi:.4f}]  p={p_val:.4f}")

    results = {
        'clean': {str(c): clean_results[c] for c in range(10)},
        'corrupted': {str(c): corrupted_results[c] for c in range(10)}
    }
    with open(output_dir / 'cifar10_rome_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
