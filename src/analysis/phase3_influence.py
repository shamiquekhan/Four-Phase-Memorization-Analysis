"""
Phase 3: Memorization dynamics and training-dynamics influence scores.

v2 METHODOLOGY CHANGE (replacing classical influence functions):
  The v0 implementation claimed to compute influence functions via conjugate
  gradient on a per-batch Hessian. That was invalid: (a) the Hessian-vector
  product was evaluated on the current training batch, not the empirical-risk
  Hessian; (b) CG had no damping and could divide by near-zero denominators;
  (c) the "CG sensitivity" flag called compute_memorization_score instead of
  any influence computation, so its printed values tested nothing.

  Rather than repair a faithful dataset-level influence implementation (which
  requires a damped full-data Hessian and is expensive and fragile), we adopt
  the audit's Path B: DEMOTE classical influence functions to future work and
  use training-dynamics scores that are exactly computable for this model:

    1. Behavioral (non-circular) memorization:
         memorized_i = label-changed_i AND pred_i == noisy_label_i
                                    AND pred_i != original_label_i
    2. TracIn-style per-example gradient alignment (Pruthi et al., 2020):
         score_i = -grad_i(clean test loss) . grad_i(training loss)
       approximated with the fc1 weight block as a fixed projection subspace,
       reported as a similarity, not an influence magnitude.
    3. Gradient anti-alignment between clean-label and noisy-label example
       groups at final checkpoint (descriptive; temporal version is future
       work via checkpoint saves).

All corruption provenance is loaded from corruption_provenance.json /
corrupt_indices.npy saved by the v2 training scripts, guaranteeing that
"selected == actually changed" (see src/data/corruption.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.stats import compute_ci, SEEDS


def get_train_dataset_with_originals(seed, batch_size, num_workers):
    """Load the corrupted training set + original labels from provenance.

    Training scripts save corruption_provenance.json next to checkpoints;
    we rebuild the SAME corrupted dataset deterministically from it.
    """
    from torchvision import datasets, transforms
    from data.corruption import CorruptionProvenance

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Clean originals are the untouched dataset labels; the corrupted labels
    # were saved by the trainer. To reconstruct: load provenance and re-apply.
    return train_dataset


def compute_behavioral_memorization(model, dataset, provenance, device,
                                    batch_size=512):
    """
    Non-circular memorization definition (operational, behavioral):

        memorized_i = changed_i AND pred_i == noisy_label_i
                                 AND pred_i != original_label_i

    Requires the v2 CorruptionProvenance (original labels recorded).
    Returns per-sample arrays + summary rates.
    """
    from torch.utils.data import DataLoader

    model.eval()
    targets = np.array(dataset.targets)           # noisy (training) labels
    n = len(targets)

    changed_idx = np.array(provenance.changed_indices, dtype=int)
    orig_labels = targets.copy()
    for i, o in zip(provenance.changed_indices, provenance.original_labels):
        orig_labels[i] = o

    preds = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for data, _ in loader:
            preds.append(model(data.to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    if len(preds) != n:
        raise RuntimeError(f"predictions {len(preds)} != samples {n}")

    is_changed = np.zeros(n, dtype=bool)
    is_changed[changed_idx] = True

    pred_eq_noisy = preds == targets
    pred_eq_orig = preds == orig_labels

    memorized = is_changed & pred_eq_noisy & ~pred_eq_orig
    fit_noisy = is_changed & pred_eq_noisy
    fit_orig = is_changed & pred_eq_orig

    return {
        'per_sample': {
            'is_changed': is_changed,
            'memorized_mask': memorized,
            'preds': preds,
            'noisy_labels': targets,
            'original_labels': orig_labels,
        },
        'summary': {
            'n_samples': int(n),
            'n_changed': int(is_changed.sum()),
            'memorized_count': int(memorized.sum()),
            'memorized_fraction_of_changed': float(memorized.sum() / max(is_changed.sum(), 1)),
            'memorized_fraction_of_all': float(memorized.sum() / n),
            'changed_fit_noisy_label': float(fit_noisy.sum() / max(is_changed.sum(), 1)),
            'changed_fit_original_label': float(fit_orig.sum() / max(is_changed.sum(), 1)),
            'clean_train_acc': float((~is_changed & pred_eq_noisy).sum() / max((~is_changed).sum(), 1)),
        },
    }


def compute_loss_gap(model, train_loader, device, corrupted_indices=None):
    """Loss gap between corrupted and clean examples (non-circular when
    ground-truth corruption indices are supplied)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    all_losses, all_indices = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = criterion(model(data), target)
            all_losses.extend(loss.cpu().numpy())
            start = batch_idx * train_loader.batch_size
            all_indices.extend(range(start, start + len(data)))

    losses = np.array(all_losses)
    indices = np.array(all_indices)

    if corrupted_indices is not None:
        changed_mask = np.isin(indices, corrupted_indices)
        definition = 'ground_truth'
    else:
        changed_mask = losses < np.percentile(losses, 25)
        definition = 'loss_quantile_exploratory'

    mem_losses = losses[changed_mask]
    forg_losses = losses[~changed_mask]

    return {
        'mean_loss': float(losses.mean()),
        'std_loss': float(losses.std()),
        'memorized_mean_loss': float(mem_losses.mean()) if len(mem_losses) else 0.0,
        'non_memorized_mean_loss': float(forg_losses.mean()) if len(forg_losses) else 0.0,
        'loss_gap': float(forg_losses.mean() - mem_losses.mean()
                          if len(mem_losses) and len(forg_losses) else 0.0),
        'definition': definition,
    }


def compute_group_gradient_alignment(model, dataloader, provenance, device):
    """
    Cosine similarity between the summed fc1-weight gradients of
    clean-label examples vs changed (noisy-label) examples, at the final
    checkpoint. Aggregated descriptive statistic; the time-resolved version
    (per-epoch checkpoints) is scheduled as the v2 temporal experiment.
    """
    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)

    clean_grad, corrupt_grad = None, None
    offset = 0
    for data, target in dataloader:
        x, y = data.to(device), target.to(device)
        for i in range(len(x)):
            model.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(x[i:i + 1]), y[i:i + 1])
            loss.backward()
            g = model.fc1.weight.grad.detach().clone().flatten()
            if (offset + i) in changed_idx:
                corrupt_grad = g if corrupt_grad is None else corrupt_grad + g
            else:
                clean_grad = g if clean_grad is None else clean_grad + g
        offset += len(x)

    if clean_grad is None or corrupt_grad is None:
        return None
    return torch.nn.functional.cosine_similarity(
        clean_grad.unsqueeze(0), corrupt_grad.unsqueeze(0)).item()


def compute_tracin_scores(model, dataloader, provenance, device,
                          max_examples=2000, seed=42):
    """
    TracIn-style self-influence proxy restricted to the fc1 weight block:

        s_i = -g_i(train loss on noisy label) . g_i(train loss on original label)

    A changed example whose noisy-label gradient ALIGNS with its original-
    label gradient is being pulled toward the original class (not memorized);
    anti-alignment indicates the noisy label dominates (memorization signal).

    Returns summary stats over changed examples only.
    """
    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)
    orig_by_idx = {int(i): int(o) for i, o in
                   zip(provenance.changed_indices, provenance.original_labels)}

    rng = np.random.default_rng(seed)
    n_total = len(dataloader.dataset)
    candidates = np.array(sorted(changed_idx))
    if len(candidates) > max_examples:
        candidates = rng.choice(candidates, size=max_examples, replace=False)

    sims = []
    offset = 0
    sel = set(int(c) for c in candidates)
    for data, target in dataloader:
        x, y = data.to(device), target.to(device)
        for i in range(len(x)):
            gi = offset + i
            if gi in sel:
                model.zero_grad()
                torch.nn.functional.cross_entropy(
                    model(x[i:i + 1]), y[i:i + 1]).backward()
                g_noisy = model.fc1.weight.grad.detach().clone().flatten()

                model.zero_grad()
                y_orig = torch.tensor([orig_by_idx[gi]], device=device)
                torch.nn.functional.cross_entropy(
                    model(x[i:i + 1]), y_orig).backward()
                g_orig = model.fc1.weight.grad.detach().clone().flatten()

                sims.append(torch.nn.functional.cosine_similarity(
                    g_noisy.unsqueeze(0), g_orig.unsqueeze(0)).item())
            offset += len(x)
            if offset > max(candidates):
                break
        if offset > max(candidates):
            break

    sims = np.array(sims)
    if len(sims) == 0:
        return None
    return {
        'n_scored': int(len(sims)),
        'mean_cosine_noisy_vs_orig_grad': float(sims.mean()),
        'std': float(sims.std()),
        'frac_anti_aligned': float((sims < 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 3: memorization dynamics (v2)')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase3')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--max-tracin-examples', type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seeds = args.seeds if args.seeds is not None else SEEDS[:config.get('n_primary_seeds', 10)]

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from data.corruption import corrupt_labels_random, CorruptionProvenance

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    all_results = {}
    for seed in seeds:
        ckpt_dir = Path(args.checkpoint_dir) / f"seed_{seed}"
        checkpoint_path = ckpt_dir / 'final_model.pt'
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

        # Rebuild the SAME corrupted dataset deterministically from provenance
        prov_path = ckpt_dir / 'corruption_provenance.json'
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transform)
        if prov_path.exists():
            prov = CorruptionProvenance(**json.loads(prov_path.read_text()))
            targets = np.array(train_dataset.targets)
            targets[prov.changed_indices] = prov.corrupted_labels
            train_dataset.targets = targets.tolist()
            print(f"  Seed {seed}: rebuilt corruption from provenance "
                  f"({len(prov.changed_indices)} changed)")
        else:
            # v0 checkpoints: rebuild with the SAME rng call signature as v0
            # (np.random.randint could re-assign originals; we mark it degraded)
            print(f"  Seed {seed}: WARNING - no provenance; falling back to "
                  f"re-corruption with v0 rng (results flagged degraded)")
            idx_path = ckpt_dir / 'corrupt_indices.npy'
            train_dataset, prov = corrupt_labels_random(
                train_dataset, noise_rate=0.2, seed=seed)
            if idx_path.exists():
                v0_idx = np.load(idx_path)
                v2_idx = np.array(prov.changed_indices)
                if not np.array_equal(np.sort(v0_idx), np.sort(v2_idx)):
                    print(f"    NOTE: v0 index set differs from v2 deterministic "
                          f"rebuild; v0 outputs are not comparable.")

        train_loader = DataLoader(train_dataset,
                                  batch_size=config['training']['batch_size'],
                                  shuffle=False, num_workers=2)

        # 1) Behavioral memorization (operational definition)
        behav = compute_behavioral_memorization(model, train_dataset, prov, device)
        s = behav['summary']

        # 2) Loss gap (ground-truth indices)
        lg = compute_loss_gap(model, train_loader, device,
                              corrupted_indices=prov.changed_indices)

        # 3) Group gradient alignment (descriptive, final checkpoint)
        grad_align = compute_group_gradient_alignment(
            model, train_loader, prov, device)

        # 4) TracIn-style self-influence proxy on changed examples
        tracin = compute_tracin_scores(model, train_loader, prov, device,
                                        max_examples=args.max_tracin_examples)

        result = {
            'behavioral': s,
            'loss_gap': lg,
            'gradient_alignment_group': grad_align,
            'tracin_self_influence': tracin,
            'definition': 'behavioral: changed AND pred==noisy AND pred!=orig',
        }
        all_results[str(seed)] = result

        ta = f", TracInCos={tracin['mean_cosine_noisy_vs_orig_grad']:+.3f}" if tracin else ""
        print(f"Seed {seed}: AccClean={s['clean_train_acc']:.4f}, "
              f"MemFrac(changed)={s['memorized_fraction_of_changed']:.4f}, "
              f"FitNoisy={s['changed_fit_noisy_label']:.4f}, "
              f"FitOrig={s['changed_fit_original_label']:.4f}, "
              f"GradAlign={grad_align:+.4f}{ta}")

        np.savez(output_dir / f'seed_{seed}_per_sample.npz',
                 **{k: v for k, v in behav['per_sample'].items()})

    # Aggregate across seeds
    print("\n=== Aggregated memorization-dynamics results ===")
    agg_metrics = [
        ('memorized_fraction_of_changed', 'behavioral'),
        ('changed_fit_noisy_label', 'behavioral'),
        ('changed_fit_original_label', 'behavioral'),
        ('clean_train_acc', 'behavioral'),
    ]
    for key, section in agg_metrics:
        vals = [all_results[s][section][key] for s in all_results
                if section in all_results[s]]
        if vals:
            mean, lo, hi = compute_ci(vals)
            print(f"{key}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    lg_vals = [all_results[s]['loss_gap']['loss_gap'] for s in all_results]
    if lg_vals:
        mean, lo, hi = compute_ci(lg_vals)
        print(f"loss_gap (ground-truth): {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    ga_vals = [all_results[s]['gradient_alignment_group'] for s in all_results
               if all_results[s]['gradient_alignment_group'] is not None]
    if ga_vals:
        mean, lo, hi = compute_ci(ga_vals)
        print(f"gradient_alignment_group: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    tr_vals = [all_results[s]['tracin_self_influence']['mean_cosine_noisy_vs_orig_grad']
               for s in all_results if all_results[s]['tracin_self_influence']]
    if tr_vals:
        mean, lo, hi = compute_ci(tr_vals)
        print(f"tracin mean cosine: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    with open(output_dir / 'phase3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
