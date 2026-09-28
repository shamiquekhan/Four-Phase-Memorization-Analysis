"""
Early-warning prediction of memorization onset (Stage C).

Compares behavioral baselines (CSL, forgetting, loss) against internal signals
(gradient conflict, representation drift, margins) for predicting future
memorization events.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


def prepare_prediction_features(checkpoint_results: List[Dict],
                                 aggregated: Dict,
                                 provenance,
                                 lookahead: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target for early-warning prediction.

    At epoch t, predict Y_i(t, Δ) = 1[T_i > t and T_i ≤ t+Δ]
    where T_i is the memorization onset epoch for example i.

    Returns:
        X: [n_examples, n_features] feature matrix at epoch t
        y: [n_examples] binary target
        feature_names: list of feature names
    """
    n_epochs = len(checkpoint_results)
    n_samples = len(checkpoint_results[0]['indices'])

    # Target: will memorize within lookahead epochs
    onset = aggregated['memorization_onset']  # -1 if never memorized
    t = n_epochs - lookahead - 1  # prediction time (second-to-last window)

    if t < 0:
        raise ValueError(f"Not enough epochs ({n_epochs}) for lookahead {lookahead}")

    # Binary target: memorized in (t, t+lookahead]
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if onset[i] > t and onset[i] <= t + lookahead:
            y[i] = 1

    # Features at epoch t
    features = {}
    feature_names = []

    # 1. Behavioral baselines
    # Current loss
    features['loss'] = checkpoint_results[t]['losses']
    feature_names.append('loss')

    # CSL up to epoch t
    csl_t = np.sum([checkpoint_results[e]['losses'] for e in range(t+1)], axis=0)
    features['csl'] = csl_t
    feature_names.append('csl')

    # Forgetting count up to epoch t
    forgetting_t = np.zeros(n_samples)
    for i in range(n_samples):
        true_label = checkpoint_results[0]['true_labels'][i]
        pred_traj = [checkpoint_results[e]['predictions'][i] for e in range(t+1)]
        from analysis.memorization import compute_forgetting_events
        forgetting_t[i] = compute_forgetting_events(pred_traj, true_label)
    features['forgetting'] = forgetting_t
    feature_names.append('forgetting')

    # Margin gap (noisy - true)
    margin_gap = checkpoint_results[t]['noisy_label_losses'] - checkpoint_results[t]['true_label_losses']
    features['margin_gap'] = margin_gap
    feature_names.append('margin_gap')

    # True label probability
    probs = torch.softmax(torch.from_numpy(checkpoint_results[t]['losses'].reshape(-1, 1)), dim=-1)
    # Actually need logits for probs - skip for now

    # 2. Internal signals
    # Gradient conflict (if available)
    if 'gradient_conflict' in checkpoint_results[t]:
        gc = checkpoint_results[t]['gradient_conflict']
        # Map back to full sample space
        gc_full = np.zeros(n_samples)
        gc_indices = gc['indices']
        gc_full[gc_indices] = gc['cosines']
        features['grad_conflict'] = gc_full
        feature_names.append('grad_conflict')

    # Activation norm (hidden layer)
    # Would need to be stored separately

    # Representation drift (CKA with clean) - would need clean model

    # Stack features
    X = np.column_stack([features[name] for name in feature_names])

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, y, feature_names


def train_early_warning_model(X: np.ndarray,
                               y: np.ndarray,
                               feature_names: List[str],
                               C: float = 1.0,
                               max_iter: int = 1000) -> Tuple[LogisticRegression, Dict]:
    """
    Train logistic regression for early-warning prediction.

    Returns:
        model: fitted LogisticRegression
        metrics: dict with coefficients, AUROC, AUPRC
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', random_state=42)
    model.fit(X_scaled, y)

    y_pred = model.predict_proba(X_scaled)[:, 1]
    auroc = roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else 0.5
    auprc = average_precision_score(y, y_pred) if len(np.unique(y)) > 1 else y.mean()

    coef_dict = {name: float(coef) for name, coef in zip(feature_names, model.coef_[0])}

    metrics = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'prevalence': float(y.mean()),
        'coefficients': coef_dict,
        'intercept': float(model.intercept_[0]),
    }

    return model, metrics


def evaluate_feature_ablation(X: np.ndarray,
                               y: np.ndarray,
                               feature_names: List[str],
                               ablation_sets: Dict[str, List[str]]) -> Dict:
    """
    Evaluate prediction performance with different feature subsets.

    ablation_sets: dict of set_name -> list of feature names to include
    """
    results = {}

    for set_name, feat_list in ablation_sets.items():
        # Get column indices
        idx = [feature_names.index(f) for f in feat_list if f in feature_names]
        if not idx:
            results[set_name] = {'auroc': 0.5, 'auprc': 0.0, 'error': 'no valid features'}
            continue

        X_sub = X[:, idx]
        model, metrics = train_early_warning_model(X_sub, y, feat_list)
        results[set_name] = metrics

    return results


def run_early_warning_analysis(temporal_dir: Path,
                                provenance,
                                lookahead_horizons: List[int] = [5, 10, 20],
                                held_out_seeds: bool = True) -> Dict:
    """
    Run full early-warning analysis across seeds and horizons.

    temporal_dir: directory containing per-seed temporal artifacts
    """
    from analysis.memorization import aggregate_temporal_metrics

    seed_dirs = sorted(temporal_dir.glob('init_*_corr_*_load_*'))
    all_results = {}

    for seed_dir in seed_dirs:
        # Load checkpoint results
        epoch_files = sorted(seed_dir.glob('seed_*_epoch_*.npz'))
        if not epoch_files:
            continue

        # Group by seed
        seed = int(seed_dir.name.split('_')[-1])
        checkpoint_results = []
        for f in epoch_files:
            data = np.load(f)
            checkpoint_results.append({
                'indices': data['indices'],
                'losses': data['losses'],
                'true_label_losses': data['true_label_losses'],
                'noisy_label_losses': data['noisy_label_losses'],
                'margins': data['margins'],
                'predictions': data['predictions'],
                'true_labels': data['true_labels'],
                'noisy_labels': data['noisy_labels'],
                'gradient_conflict': data['gradient_conflict'].item() if 'gradient_conflict' in data else None,
            })

        # Aggregate
        aggregated = aggregate_temporal_metrics(checkpoint_results, provenance)

        seed_results = {}
        for lookahead in lookahead_horizons:
            try:
                X, y, feat_names = prepare_prediction_features(
                    checkpoint_results, aggregated, provenance, lookahead)

                # Define feature ablation sets
                ablation_sets = {
                    'loss_only': ['loss'],
                    'csl_only': ['csl'],
                    'forgetting_only': ['forgetting'],
                    'loss_csl': ['loss', 'csl'],
                    'csl_forgetting': ['csl', 'forgetting'],
                    'behavioral': ['loss', 'csl', 'forgetting', 'margin_gap'],
                    'internal': ['grad_conflict'],  # add more when available
                    'all': feat_names,
                }

                ablation_results = evaluate_feature_ablation(X, y, feat_names, ablation_sets)
                seed_results[f'lookahead_{lookahead}'] = ablation_results

            except Exception as e:
                seed_results[f'lookahead_{lookahead}'] = {'error': str(e)}

        all_results[str(seed)] = seed_results

    # Aggregate across seeds
    aggregated_results = {}
    for lookahead in lookahead_horizons:
        key = f'lookahead_{lookahead}'
        method_names = set()
        for seed_res in all_results.values():
            if key in seed_res and 'error' not in seed_res[key]:
                method_names.update(seed_res[key].keys())

        agg = {}
        for method in method_names:
            aurocs = []
            auprcs = []
            for seed_res in all_results.values():
                if key in seed_res and method in seed_res[key]:
                    m = seed_res[key][method]
                    if 'auroc' in m:
                        aurocs.append(m['auroc'])
                        auprcs.append(m['auprc'])
            if aurocs:
                agg[method] = {
                    'auroc_mean': float(np.mean(aurocs)),
                    'auroc_std': float(np.std(aurocs)),
                    'auprc_mean': float(np.mean(auprcs)),
                    'auprc_std': float(np.std(auprcs)),
                    'n_seeds': len(aurocs),
                }

        aggregated_results[key] = agg

    return {
        'per_seed': all_results,
        'aggregated': aggregated_results,
        'lookahead_horizons': lookahead_horizons,
    }


def save_prediction_results(results: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    print("Early-warning prediction module loaded.")