#!/usr/bin/env python3
"""
Temporal dynamics analysis: loads epoch checkpoints and computes
metrics at each epoch for clean vs corrupted models.

Metrics:
- Cross-model CKA drift (clean_epoch vs corrupted_epoch, same epoch)
- Spectral metrics (stable rank, effective rank, entropy, cumulative energy)
- fc2 delta-norm (rank-one edit, EDIT split)
- Behavioral memorization (corrupted only)
- Group gradient anti-alignment
- Test accuracy
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
from models.model import MNISTNet
from utils.metrics import linear_cka, compute_spectral_metrics, evaluate_class_accuracy
from utils.stats import compute_ci, SEEDS
from data.corruption import CorruptionProvenance
from data.splits import split_edit_eval, SplitProvenance


CKPT_EPOCHS = [0, 1, 2, 4, 8, 12, 16, 20]
CKA_LAYERS = ['input', 'fc1_pre_activation', 'fc1_post_activation', 'output']
SEEDS = [42, 123, 456, 789, 1024]


def get_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return datasets.MNIST('./data', train=False, download=True, transform=transform)


def build_edit_eval_loaders(split_seed=42, batch_size=256):
    ds = get_test_dataset()
    targets = np.array(ds.targets)
    edit_idx, eval_idx = [], []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        sp = split_edit_eval(cls_idx, eval_fraction=0.5, seed=split_seed, split_name=f'class_{c}')
        edit_idx.extend(sp.edit_indices)
        eval_idx.extend(sp.eval_indices)
    split = SplitProvenance('mnist_test_edit_eval_stratified', edit_idx, eval_idx, seed=split_seed)
    assert set(edit_idx).isdisjoint(eval_idx)
    from torch.utils.data import Subset
    edit_loader = DataLoader(Subset(ds, split.edit_indices), batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(Subset(ds, split.eval_indices), batch_size=batch_size, shuffle=False)
    return edit_loader, eval_loader, split


def get_drift_loader(batch_size=256):
    ds = get_test_dataset()
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_model(ckpt_path, config, device):
    model = MNISTNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        activation=config['model']['activation']
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_activations(model, dataloader, device, max_samples=5000):
    model.eval()
    layers = CKA_LAYERS  # use the module-level constant
    acts = {k: [] for k in layers}
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            for k in layers:
                acts[k].append(out[k].cpu())
            if sum(x.size(0) for x in acts['input']) >= max_samples:
                break
    return {k: torch.cat(v)[:max_samples] for k, v in acts.items()}


def cross_model_drift(model_clean, model_corr, loader, device):
    acts_c = get_activations(model_clean, loader, device)
    acts_r = get_activations(model_corr, loader, device)
    return {layer: 1.0 - linear_cka(acts_c[layer], acts_r[layer]) for layer in CKA_LAYERS}


def compute_rank1_delta_norm(model, edit_loader, device, layer='fc2'):
    model.eval()
    norms = []
    for c in range(model.output_dim):
        target_key = []
        with torch.no_grad():
            for data, target in edit_loader:
                data = data.to(device)
                acts = model.forward_with_all_layers(data)
                mask = target == c
                if mask.any():
                    target_key.append(acts['fc1_post_activation'][mask].cpu())
        if not target_key:
            continue
        u = torch.cat(target_key).mean(0).to(device)
        v = torch.zeros(model.output_dim, device=device)
        v[c] = 1.0
        W = model.fc2.weight.data
        delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
        norms.append(delta.norm(p='fro').item())
    return np.mean(norms) if norms else 0.0


def evaluate_group_gradient_alignment(model, train_loader, provenance, device):
    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)
    clean_grad, corrupt_grad = None, None
    offset = 0
    for data, target in train_loader:
        x, y = data.to(device), target.to(device)
        for i in range(len(x)):
            model.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1])
            loss.backward()
            g = model.fc1.weight.grad.detach().clone().flatten()
            if (offset + i) in changed_idx:
                corrupt_grad = g if corrupt_grad is None else corrupt_grad + g
            else:
                clean_grad = g if clean_grad is None else clean_grad + g
        offset += len(x)
    if clean_grad is None or corrupt_grad is None:
        return None
    return torch.nn.functional.cosine_similarity(clean_grad.unsqueeze(0), corrupt_grad.unsqueeze(0)).item()


def compute_tracin(model, train_loader, provenance, device, max_examples=1000, seed=42):
    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)
    orig_by_idx = {int(i): int(o) for i, o in zip(provenance.changed_indices, provenance.original_labels)}
    rng = np.random.default_rng(seed)
    n_total = len(train_loader.dataset)
    candidates = np.array(sorted(changed_idx))
    if len(candidates) > max_examples:
        candidates = rng.choice(candidates, size=max_examples, replace=False)
    sims = []
    offset = 0
    sel = set(int(c) for c in candidates)
    for data, target in train_loader:
        x, y = data.to(device), target.to(device)
        for i in range(len(x)):
            gi = offset + i
            if gi in sel:
                model.zero_grad()
                torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
                g_noisy = model.fc1.weight.grad.detach().clone().flatten()
                model.zero_grad()
                y_orig = torch.tensor([orig_by_idx[gi]], device=device)
                torch.nn.functional.cross_entropy(model(x[i:i+1]), y_orig).backward()
                g_orig = model.fc1.weight.grad.detach().clone().flatten()
                sims.append(torch.nn.functional.cosine_similarity(g_noisy.unsqueeze(0), g_orig.unsqueeze(0)).item())
            offset += len(x)
            if offset > max(candidates):
                break
        if offset > max(candidates):
            break
    if not sims:
        return None
    sims = np.array(sims)
    return {
        'mean_cosine': float(sims.mean()),
        'frac_anti_aligned': float((sims < 0).mean()),
    }


def load_provenance(ckpt_dir):
    prov_path = ckpt_dir / 'corruption_provenance.json'
    if prov_path.exists():
        return CorruptionProvenance(**json.loads(prov_path.read_text()))
    return None


def main():
    parser = argparse.ArgumentParser(description='Temporal dynamics analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--temporal-dir', type=str, default='outputs/temporal')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/temporal')
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edit_loader, eval_loader, split = build_edit_eval_loaders(split_seed=args.split_seed)
    drift_loader = get_drift_loader()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset_clean = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataset_corr = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # We'll rebuild the corrupted dataset per epoch from the saved provenance
    # (the provenance is the same for all epochs of a given seed)

    all_results = {}

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        seed_results = {}
        clean_dir = Path(args.temporal_dir) / 'clean' / f"seed_{seed}"
        corr_dir = Path(args.temporal_dir) / 'noise_0.2' / f"seed_{seed}"

        if not clean_dir.exists() or not corr_dir.exists():
            print(f"  Missing dirs for seed {seed}")
            continue

        # Load provenance (same for all epochs)
        prov = load_provenance(corr_dir)
        if prov is None:
            print(f"  No provenance for seed {seed}")
            continue

        # Rebuild corrupted train dataset for behavioral metrics
        targets = np.array(train_dataset_corr.targets)
        targets[prov.changed_indices] = prov.corrupted_labels
        train_dataset_corr.targets = targets.tolist()
        corr_train_loader = DataLoader(train_dataset_corr, batch_size=config['training']['batch_size'],
                                       shuffle=False, num_workers=2)

        seed_results = {}
        for epoch in CKPT_EPOCHS:
            print(f"  Epoch {epoch}...")
            ckpt_clean = clean_dir / f"epoch_{epoch}" / 'model.pt'
            ckpt_corr = corr_dir / f"epoch_{epoch}" / 'model.pt'
            if not ckpt_clean.exists() or not ckpt_corr.exists():
                print(f"    Missing checkpoint for epoch {epoch}")
                continue

            m_clean = load_model(ckpt_clean, config, device)
            m_corr = load_model(ckpt_corr, config, device)

            # Cross-model CKA drift
            drift = cross_model_drift(m_clean, m_corr, drift_loader, device)

            # Spectral metrics
            sp_fc1 = compute_spectral_metrics(m_clean.fc1.weight.data)
            sp_fc2_c = compute_spectral_metrics(m_clean.fc2.weight.data)
            sp_fc2_r = compute_spectral_metrics(m_corr.fc2.weight.data)

            # fc2 delta-norm (EDIT split)
            dn_clean = compute_rank1_delta_norm(m_clean, edit_loader, device)
            dn_corr = compute_rank1_delta_norm(m_corr, edit_loader, device)

            # Behavioral memorization (corrupted only)
            prov_epoch = load_provenance(corr_dir / f"epoch_{epoch}")
            if prov_epoch is None:
                prov_epoch = prov  # same provenance
            bm = None
            try:
                bm = evaluate_behavioral_memorization(m_corr, train_dataset_corr, prov_epoch, device)
            except:
                pass

            # Group gradient alignment
            ga = evaluate_group_gradient_alignment(m_corr, corr_train_loader, prov, device)

            # TracIn
            tr = compute_tracin(m_corr, corr_train_loader, prov, device)

            # Test accuracy (from history)
            # We'll load from history later

            epoch_results = {
                'drift': drift,
                'clean_fc2_stable_rank': sp_fc2_c['stable_rank'],
                'clean_fc2_effective_rank': sp_fc2_c['effective_rank'],
                'clean_fc2_spectral_entropy': sp_fc2_c['spectral_entropy'],
                'corr_fc2_stable_rank': sp_fc2_r['stable_rank'],
                'corr_fc2_effective_rank': sp_fc2_r['effective_rank'],
                'corr_fc2_spectral_entropy': sp_fc2_r['spectral_entropy'],
                'delta_norm_clean': dn_clean,
                'delta_norm_corrupted': dn_corr,
                'delta_norm_ratio': dn_clean / max(dn_corr, 1e-9),
                'behavioral_memorization': bm,
                'gradient_alignment_group': ga,
                'tracin': tr,
            }
            seed_results[epoch] = epoch_results
            print(f"    epoch {epoch}: drift_output={drift.get('output', 0):.4f}, "
                  f"dn_ratio={epoch_results['delta_norm_ratio']:.2f}, "
                  f"GA={ga:.4f}" if ga else f"    epoch {epoch}: no GA")

        all_results[str(seed)] = seed_results

    # Aggregate across seeds
    print("\n=== Aggregated Temporal Dynamics ===")
    for epoch in CKPT_EPOCHS:
        drifts = {layer: [] for layer in CKA_LAYERS}
        dn_ratios = []
        for seed in args.seeds:
            if str(seed) in all_results and epoch in all_results[str(seed)]:
                r = all_results[str(seed)][epoch]
                for layer in CKA_LAYERS:
                    drifts[layer].append(r['drift'][layer])
                dn_ratios.append(r['delta_norm_ratio'])

        print(f"\nEpoch {epoch}:")
        for layer in CKA_LAYERS:
            if drifts[layer]:
                m, lo, hi = compute_ci(drifts[layer])
                print(f"  drift({layer}): {m:.4f} [{lo:.4f}, {hi:.4f}]")
        if dn_ratios:
            m, lo, hi = compute_ci(dn_ratios)
            print(f"  delta_norm_ratio: {m:.2f}x [{lo:.2f}, {hi:.2f}]")

    # Save
    with open(output_dir / 'temporal_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


def evaluate_behavioral_memorization(model, dataset, provenance, device, batch_size=512):
    """Non-circular memorization definition."""
    model.eval()
    targets = np.array(dataset.targets)
    n = len(targets)
    changed_idx = np.array(provenance.changed_indices, dtype=int)
    orig_labels = targets.copy()
    for i, o in zip(provenance.changed_indices, provenance.original_labels):
        orig_labels[i] = o

    preds = []
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data, _ in loader:
            preds.append(model(data.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()

    is_changed = np.zeros(n, dtype=bool)
    is_changed[changed_idx] = True
    pred_eq_noisy = preds == targets
    pred_eq_orig = preds == orig_labels
    memorized = is_changed & pred_eq_noisy & ~pred_eq_orig

    return {
        'memorized_fraction_of_changed': float(memorized.sum() / max(is_changed.sum(), 1)),
        'changed_fit_original_label': float((is_changed & pred_eq_orig).sum() / max(is_changed.sum(), 1)),
    }


if __name__ == '__main__':
    import yaml
    main()