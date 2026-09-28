"""
Low-rank weight editing with EDIT_MEM / EVAL_MEM firewall (v2).

Implements minimum-norm rank-k updates for targeted memorization reversal.

Protocol:
  - EDIT_MEM: subset of memorized corrupted training examples (used to construct edit)
  - EVAL_MEM: held-out memorized corrupted examples (evaluate generalization)
  - FINAL_TEST: clean test set (evaluate collateral damage)

This replaces the test-set ROME protocol with a proper train-set edit/eval split.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
from copy import deepcopy


def split_edit_eval_mem(provenance,
                        memorized_indices: List[int],
                        eval_fraction: float = 0.5,
                        seed: int = 42) -> Tuple[List[int], List[int], Dict]:
    """
    Split memorized corrupted examples into EDIT_MEM and EVAL_MEM.

    Returns:
        edit_indices, eval_indices, split_provenance_dict
    """
    rng = np.random.default_rng(seed)
    idx = np.array(sorted(int(i) for i in memorized_indices))
    idx = rng.permutation(idx)
    n_eval = int(len(idx) * eval_fraction)
    eval_idx = idx[:n_eval].tolist()
    edit_idx = idx[n_eval:].tolist()

    prov = {
        'split_name': 'edit_eval_mem',
        'seed': seed,
        'n_edit': len(edit_idx),
        'n_eval': len(eval_idx),
        'edit_indices': edit_idx,
        'eval_indices': eval_idx,
        'memorized_source': True,
    }
    return edit_idx, eval_idx, prov


def compute_min_norm_edit(model: nn.Module,
                           edit_loader: DataLoader,
                           device: torch.device,
                           target_class: int,
                           layer_name: str = 'fc2',
                           reg_lambda: float = 1e-4,
                           rank: int = 1,
                           include_bias: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Compute minimum-norm linear update under ridge regularization.

    Solves: Δ* = R K^T (K K^T + λI)^{-1}
    where K = [k_1, ..., k_n] are augmented activations from EDIT_MEM,
    R = [r_1, ..., r_n] are desired residuals (y_desired - W k_i).

    Then computes rank-k truncation via SVD.

    Args:
        model: Trained model
        edit_loader: DataLoader over EDIT_MEM examples (target class only)
        device: Compute device
        target_class: Class to restore (true label)
        layer_name: Layer to edit ('fc1' or 'fc2')
        reg_lambda: Ridge regularization
        rank: Target rank for SVD truncation (1, 2, 4, 8, full)
        include_bias: Whether to augment activations with bias term

    Returns:
        delta_weight: The edit to apply to the weight matrix
        metadata: Dict with diagnostics
    """
    model.eval()
    layer = getattr(model, layer_name)
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None

    # Collect keys and residuals from EDIT_MEM
    keys = []
    residuals = []

    with torch.no_grad():
        for data, target in edit_loader:
            data = data.to(device)
            if layer_name == 'fc2':
                # Hidden activations (post-ReLU for fc1)
                acts = model.forward_with_all_layers(data)['fc1_post_activation']
            elif layer_name == 'fc1':
                # Input activations (flattened)
                acts = data.view(data.size(0), -1)
            else:
                raise ValueError(f"Unknown layer: {layer_name}")

            # Filter to target class
            mask = target == target_class
            if mask.any():
                keys.append(acts[mask])

    if not keys:
        return None, {"error": "No examples of target class in edit_loader"}

    K = torch.cat(keys).T  # [hidden_dim, n_examples]

    # Augment with bias term
    if include_bias and b is not None:
        K = torch.cat([K, torch.ones(1, K.shape[1], device=K.device)], dim=0)
        W_aug = torch.cat([W, b.unsqueeze(1)], dim=1)
    else:
        W_aug = W

    # Desired output: one-hot for target class
    y_desired = torch.zeros(W.shape[0], device=device)
    y_desired[target_class] = 1.0

    # Current outputs: W K (or W_aug K)
    current_outputs = W_aug @ K  # [output_dim, n_examples]

    # Residuals: y_desired - current_outputs
    R = y_desired.unsqueeze(1) - current_outputs  # [output_dim, n_examples]

    # Minimum-norm solution: Δ = R K^T (K K^T + λI)^{-1}
    KKt = K @ K.T
    reg = reg_lambda * torch.eye(KKt.shape[0], device=KKt.device)
    try:
        inv = torch.linalg.inv(KKt + reg)
    except torch.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        inv = torch.linalg.pinv(KKt + reg)

    delta_full = R @ K.T @ inv  # [output_dim, hidden_dim (+1 for bias)]

    # Rank-k truncation via SVD
    if rank > 0 and rank < min(delta_full.shape):
        U, S, Vh = torch.linalg.svd(delta_full, full_matrices=False)
        delta = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    else:
        delta = delta_full

    # Separate weight and bias deltas
    if include_bias and b is not None:
        delta_weight = delta[:, :-1]
        delta_bias = delta[:, -1]
    else:
        delta_weight = delta
        delta_bias = None

    metadata = {
        'layer': layer_name,
        'rank': rank,
        'reg_lambda': reg_lambda,
        'delta_norm_fro': delta.norm(p='fro').item(),
        'delta_norm_spectral': delta.norm(p=2).item(),
        'n_edit_examples': K.shape[1],
        'target_class': target_class,
        'include_bias': include_bias,
    }

    return delta_weight, delta_bias, metadata


def apply_edit(model: nn.Module,
               delta_weight: torch.Tensor,
               delta_bias: Optional[torch.Tensor],
               layer_name: str) -> None:
    """Apply weight (and bias) edit to model in-place."""
    layer = getattr(model, layer_name)
    with torch.no_grad():
        layer.weight.data.add_(delta_weight)
        if delta_bias is not None and layer.bias is not None:
            layer.bias.data.add_(delta_bias)


def revert_edit(model: nn.Module,
                delta_weight: torch.Tensor,
                delta_bias: Optional[torch.Tensor],
                layer_name: str) -> None:
    """Revert weight (and bias) edit from model in-place."""
    layer = getattr(model, layer_name)
    with torch.no_grad():
        layer.weight.data.sub_(delta_weight)
        if delta_bias is not None and layer.bias is not None:
            layer.bias.data.sub_(delta_bias)


def evaluate_edit(model: nn.Module,
                  eval_loader: DataLoader,
                  device: torch.device,
                  target_class: int,
                  metric_fn: Optional[Callable] = None) -> Dict:
    """Evaluate edit on a DataLoader, returning per-class accuracies and metrics."""
    model.eval()

    if metric_fn is None:
        def metric_fn(logits, targets):
            preds = logits.argmax(1)
            accs = {}
            for c in range(10):
                mask = targets == c
                if mask.any():
                    accs[c] = (preds[mask] == c).float().mean().item()
                else:
                    accs[c] = 0.0
            return accs

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            all_logits.append(logits)
            all_targets.append(target)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    accs = metric_fn(all_logits, all_targets)

    return {
        'per_class_accuracy': accs,
        'target_class_accuracy': accs.get(target_class, 0.0),
        'mean_other_class_accuracy': np.mean([v for k, v in accs.items() if k != target_class]),
    }


def run_rank_k_experiment(model: nn.Module,
                          edit_loader: DataLoader,
                          eval_loader: DataLoader,
                          test_loader: DataLoader,
                          device: torch.device,
                          target_class: int,
                          layer_name: str = 'fc2',
                          ranks: List[int] = [1, 2, 4, 8, 16],
                          reg_lambda: float = 1e-4) -> Dict:
    """
    Run rank-k edit experiment across multiple ranks.

    Evaluates on:
      - EVAL_MEM: held-out memorized examples (generalization)
      - FINAL_TEST: clean test set (collateral damage)
    """
    model.eval()

    # Baseline on EVAL_MEM and FINAL_TEST
    baseline_eval = evaluate_edit(model, eval_loader, device, target_class)
    baseline_test = evaluate_edit(model, test_loader, device, target_class)

    results = {
        'baseline': {
            'eval_mem': baseline_eval,
            'final_test': baseline_test,
        },
        'edits': {},
    }

    for rank in ranks:
        delta_w, delta_b, meta = compute_min_norm_edit(
            model, edit_loader, device, target_class, layer_name,
            reg_lambda=reg_lambda, rank=rank
        )

        if delta_w is None:
            results['edits'][f'rank_{rank}'] = {'error': meta.get('error', 'Edit failed')}
            continue

        # Apply edit
        apply_edit(model, delta_w, delta_b, layer_name)

        # Evaluate
        eval_results = evaluate_edit(model, eval_loader, device, target_class)
        test_results = evaluate_edit(model, test_loader, device, target_class)

        # Recovery on EVAL_MEM
        recovery = eval_results['target_class_accuracy'] - baseline_eval['target_class_accuracy']

        # Collateral on FINAL_TEST
        collateral = baseline_test['mean_other_class_accuracy'] - test_results['mean_other_class_accuracy']

        # Target class accuracy on FINAL_TEST (should ideally stay same)
        target_preservation = test_results['target_class_accuracy'] - baseline_test['target_class_accuracy']

        results['edits'][f'rank_{rank}'] = {
            'meta': meta,
            'eval_mem': eval_results,
            'final_test': test_results,
            'recovery': recovery,
            'collateral_damage': collateral,
            'target_preservation': target_preservation,
        }

        # Revert for next rank
        revert_edit(model, delta_w, delta_b, layer_name)

    return results


def run_random_baseline(model: nn.Module,
                         eval_loader: DataLoader,
                         test_loader: DataLoader,
                         device: torch.device,
                         target_class: int,
                         layer_name: str = 'fc2',
                         target_norm: float = 1.0,
                         n_trials: int = 20,
                         ranks: List[int] = [1, 2, 4, 8]) -> Dict:
    """Random norm-matched rank-k edits as null baseline."""
    model.eval()
    layer = getattr(model, layer_name)

    # Baseline
    baseline_eval = evaluate_edit(model, eval_loader, device, target_class)
    baseline_test = evaluate_edit(model, test_loader, device, target_class)

    random_results = {f'rank_{r}': {'recoveries': [], 'collaterals': []} for r in ranks}

    for trial in range(n_trials):
        torch.manual_seed(trial)
        for rank in ranks:
            # Random low-rank matrix with target Frobenius norm
            out_dim, in_dim = layer.weight.shape
            U = torch.randn(out_dim, rank, device=layer.weight.device)
            V = torch.randn(rank, in_dim, device=layer.weight.device)
            U = U / U.norm()
            V = V / V.norm()

            delta = target_norm * U @ V
            delta_b = None  # No bias in random null for simplicity

            # Apply
            apply_edit(model, delta, delta_b, layer_name)

            # Evaluate
            eval_results = evaluate_edit(model, eval_loader, device, target_class)
            test_results = evaluate_edit(model, test_loader, device, target_class)

            recovery = eval_results['target_class_accuracy'] - baseline_eval['target_class_accuracy']
            collateral = baseline_test['mean_other_class_accuracy'] - test_results['mean_other_class_accuracy']

            random_results[f'rank_{rank}']['recoveries'].append(recovery)
            random_results[f'rank_{rank}']['collaterals'].append(collateral)

            # Revert
            revert_edit(model, delta, delta_b, layer_name)

    # Summarize
    summary = {}
    for rank in ranks:
        r = random_results[f'rank_{rank}']
        summary[f'rank_{rank}'] = {
            'recovery_mean': float(np.mean(r['recoveries'])),
            'recovery_std': float(np.std(r['recoveries'])),
            'collateral_mean': float(np.mean(r['collaterals'])),
            'collateral_std': float(np.std(r['collaterals'])),
        }

    return summary


def run_lora_baseline(model: nn.Module,
                       edit_loader: DataLoader,
                       eval_loader: DataLoader,
                       test_loader: DataLoader,
                       device: torch.device,
                       target_class: int,
                       layer_name: str = 'fc2',
                       rank: int = 4,
                       epochs: int = 20,
                       lr: float = 1e-3) -> Dict:
    """
    LoRA-style fine-tuning baseline for comparison.

    Adds low-rank adapter matrices A, B and trains them on EDIT_MEM.
    """
    model.train()
    layer = getattr(model, layer_name)
    out_dim, in_dim = layer.weight.shape

    # LoRA parameters
    A = torch.randn(rank, in_dim, device=device, requires_grad=True) * 0.01
    B = torch.randn(out_dim, rank, device=device, requires_grad=True) * 0.01

    optimizer = torch.optim.Adam([A, B], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in edit_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward with LoRA
            if layer_name == 'fc2':
                h = model.forward_with_all_layers(data)['fc1_post_activation']
            elif layer_name == 'fc1':
                h = data.view(data.size(0), -1)
            else:
                raise ValueError(f"Unknown layer: {layer_name}")

            logits = h @ layer.weight.T + layer.bias
            logits = logits + h @ (B @ A)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

    # Merge LoRA into weight
    with torch.no_grad():
        layer.weight.data += B @ A

    # Evaluate
    eval_results = evaluate_edit(model, eval_loader, device, target_class)
    test_results = evaluate_edit(model, test_loader, device, target_class)

    # Un-merge
    with torch.no_grad():
        layer.weight.data -= B @ A

    return {
        'eval_mem': eval_results,
        'final_test': test_results,
        'lora_rank': rank,
        'lora_epochs': epochs,
    }


def save_edit_results(results: Dict, output_path: Path):
    """Save edit experiment results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


if __name__ == '__main__':
    print("Low-rank edit module loaded. Requires trained models for testing.")