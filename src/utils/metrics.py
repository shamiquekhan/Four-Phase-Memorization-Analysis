"""
Metrics for mechanistic interpretability analysis:
σ (separability), monosemanticity, circuit sparsity, edit magnitude.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from typing import Tuple, List, Dict


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Canonical linear CKA (Centered Kernel Alignment).
    Zero-centers inputs, then computes HSIC-based similarity.

    References: Kornblith et al. (2019) "Similarity of Neural Network
    Representations Revisited"
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum()
    hsic_yy = (L * L).sum()
    return (hsic_xy / torch.sqrt(hsic_xx * hsic_yy)).item()


def compute_fdr(activations: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fisher discriminant ratio: tr(S_B) / tr(S_W).
    Dimension-invariant class separability measure.

    Args:
        activations: (N, H) hidden-layer activations
        labels:      (N,)   integer class labels (0..C-1)

    Returns:
        fdr: scalar float
    """
    activations = activations.float()
    classes = labels.unique()
    C = len(classes)
    H = activations.shape[1]

    global_mean = activations.mean(0)          # (H,)

    S_W = torch.zeros(H, H, device=activations.device)
    S_B = torch.zeros(H, H, device=activations.device)

    for c in classes:
        mask = labels == c
        X_c = activations[mask]
        mu_c = X_c.mean(0)
        n_c = X_c.shape[0]

        diff_W = X_c - mu_c
        S_W += diff_W.T @ diff_W

        diff_B = (mu_c - global_mean).unsqueeze(1)
        S_B += n_c * (diff_B @ diff_B.T)

    tr_SW = torch.trace(S_W).item()
    tr_SB = torch.trace(S_B).item()

    if tr_SW < 1e-10:
        return float('inf')

    return tr_SB / tr_SW


def compute_sigma_and_fdr(activations: np.ndarray, labels: np.ndarray) -> dict:
    """Return both σ (legacy) and FDR (new primary metric)."""
    classes = np.unique(labels)
    within_dists, between_dists = [], []
    class_means = {}

    for c in classes:
        X_c = activations[labels == c]
        mu_c = X_c.mean(0)
        class_means[c] = mu_c
        diffs = X_c - mu_c
        within_dists.append(np.mean(np.linalg.norm(diffs, axis=1)))

    class_list = list(class_means.values())
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            between_dists.append(np.linalg.norm(class_list[i] - class_list[j]))

    sigma = (sum(within_dists) / len(within_dists)) / (sum(between_dists) / len(between_dists))
    fdr = compute_fdr(torch.from_numpy(activations), torch.from_numpy(labels))

    return {"sigma": sigma, "fdr": fdr}


def compute_davies_bouldin(activations: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin index - dimension-invariant cluster separability metric.
    Lower is better (more separated). Replaces σ which is not dimension-invariant.

    DB = (1/k) * sum_{i=1}^{k} max_{j != i} (s_i + s_j) / d(c_i, c_j)
    where s_i = mean distance of points in cluster i to its centroid
          d(c_i, c_j) = distance between centroids i and j
    """
    classes = np.unique(labels)
    centroids = []
    within_dists = []
    for c in classes:
        class_acts = activations[labels == c]
        centroid = class_acts.mean(0)
        centroids.append(centroid)
        within_dists.append(np.mean(np.linalg.norm(class_acts - centroid, axis=1)))
    centroids = np.array(centroids)
    n_classes = len(classes)
    db_sum = 0
    for i in range(n_classes):
        max_ratio = 0
        for j in range(n_classes):
            if i == j:
                continue
            centroid_dist = np.linalg.norm(centroids[i] - centroids[j])
            ratio = (within_dists[i] + within_dists[j]) / (centroid_dist + 1e-8)
            max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio
    return db_sum / n_classes


def compute_calinski_harabasz(activations: np.ndarray, labels: np.ndarray) -> float:
    """
    Calinski-Harabasz index (variance ratio criterion).
    Higher is better (more separated). Dimension-invariant.

    CH = (trace(B) / (k-1)) / (trace(W) / (n-k))
    where B = between-cluster dispersion, W = within-cluster dispersion
    """
    classes = np.unique(labels)
    n = len(activations)
    k = len(classes)
    overall_mean = activations.mean(0)
    trace_b = 0
    trace_w = 0
    for c in classes:
        class_acts = activations[labels == c]
        n_c = len(class_acts)
        centroid = class_acts.mean(0)
        trace_b += n_c * np.sum((centroid - overall_mean) ** 2)
        trace_w += np.sum((class_acts - centroid) ** 2)
    if trace_w == 0 or n - k == 0:
        return 0.0
    return (trace_b / (k - 1)) / (trace_w / (n - k))


def compute_sigma(activations: np.ndarray, labels: np.ndarray,
                  n_samples_per_class: int = 200, seed: int = 42) -> float:
    """
    Compute within/between class distance ratio (σ).
    NOTE: This metric is not dimension-invariant. Use compute_davies_bouldin instead.

    σ = mean(within_class_dist) / mean(between_class_dist)
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    within_dists, between_dists = [], []

    for c in classes:
        class_acts = activations[labels == c]
        other_acts = activations[labels != c]

        n = min(len(class_acts), n_samples_per_class)
        idx = rng.choice(len(class_acts), n, replace=False)
        sample = class_acts[idx]
        diff = sample[:, None] - sample[None, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        within_dists.append(dists[np.triu_indices(n, k=1)].mean())

        m = min(len(other_acts), n_samples_per_class)
        other_idx = rng.choice(len(other_acts), m, replace=False)
        other_sample = other_acts[other_idx]
        bt_diff = sample[:, None] - other_sample[None, :]
        bt_dists = np.sqrt((bt_diff ** 2).sum(-1))
        between_dists.append(bt_dists.mean())

    mean_within = float(np.mean(within_dists))
    mean_between = float(np.mean(between_dists))
    sigma = mean_within / mean_between
    return sigma, mean_within, mean_between


def compute_monosemanticity(
    hidden_acts: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> Tuple[float, List[float]]:
    """
    Compute fraction of monosemantic neurons (max class correlation > threshold).

    Returns:
        mono_fraction: fraction of neurons with max class corr > threshold
        max_corrs: list of max correlations per neuron
    """
    n_neurons = hidden_acts.shape[1]
    n_classes = len(np.unique(labels))
    max_corrs = []

    for n in range(n_neurons):
        neuron_acts = hidden_acts[:, n]
        corrs = []
        for c in range(n_classes):
            class_indicator = (labels == c).astype(float)
            r, _ = pearsonr(neuron_acts, class_indicator)
            corrs.append(abs(r))
        max_corrs.append(max(corrs))

    mono_fraction = float(np.mean([r > threshold for r in max_corrs]))
    return mono_fraction, max_corrs


def compute_circuit_sparsity(
    model: nn.Module, test_loader: DataLoader, device: str = 'cpu',
    threshold: float = 0.02
) -> Tuple[float, float, Dict[int, List[int]]]:
    """
    Ablation study: for each hidden neuron, measure per-class accuracy drop.
    Returns mean circuit size, sparsity, and critical neuron map.
    """
    model.eval()
    hidden_size = model.hidden_dim if hasattr(model, 'hidden_dim') else 16
    n_classes = 10

    baseline = {c: 0 for c in range(n_classes)}
    counts = {c: 0 for c in range(n_classes)}

    relu_module = None
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu_module = m
            break

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            for c in range(n_classes):
                mask = y == c
                baseline[c] += (preds[mask] == y[mask]).sum().item()
                counts[c] += mask.sum().item()

    for c in range(n_classes):
        baseline[c] /= max(counts[c], 1)

    critical_map = {}
    for c in range(n_classes):
        critical_map[c] = []

    for n in range(hidden_size):
        def make_hook(neuron_idx):
            def hook_fn(module, input, output):
                output = output.clone()
                output[:, neuron_idx] = 0
                return output
            return hook_fn

        hook = relu_module.register_forward_hook(make_hook(n))

        ablated = {c: 0 for c in range(n_classes)}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                for c in range(n_classes):
                    mask = y == c
                    ablated[c] += (preds[mask] == y[mask]).sum().item()

        hook.remove()

        for c in range(n_classes):
            ablated[c] /= max(counts[c], 1)
            drop = baseline[c] - ablated[c]
            if drop > threshold:
                critical_map[c].append(n)

    circuit_sizes = [len(critical_map[c]) for c in range(n_classes)]
    mean_circuit = float(np.mean(circuit_sizes))
    total_possible = hidden_size * n_classes
    total_critical = sum(circuit_sizes)
    sparsity = float(1 - total_critical / max(total_possible, 1))

    return mean_circuit, sparsity, critical_map


def compute_edit_magnitude(model_before: nn.Module, model_after: nn.Module) -> float:
    """Compute Frobenius norm of weight changes between two models."""
    total_diff = 0.0
    for p1, p2 in zip(model_before.parameters(), model_after.parameters()):
        total_diff += (p1.data - p2.data).norm(p='fro').item() ** 2
    return float(np.sqrt(total_diff))


def extract_hidden_activations(
    model: nn.Module, loader: DataLoader, device: str = 'cpu', max_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract hidden layer activations and labels from a data loader."""
    model.eval()
    all_hidden, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.get_hidden(x).cpu().numpy()
            all_hidden.append(h)
            all_labels.append(y.numpy())
            if sum(len(h) for h in all_hidden) >= max_samples:
                break

    return np.concatenate(all_hidden)[:max_samples], np.concatenate(all_labels)[:max_samples]


def compute_spectral_metrics(weight: torch.Tensor) -> dict:
    """
    Spectrum-derived quantities for a weight matrix. Replaces the invalid
    inference "spectral norm down => effective rank down" with directly
    computed measures.

    Returns:
        stable_rank:        ||W||_F^2 / ||W||_2^2  (1 = rank-1, = full rank for isotropic)
        effective_rank:     exp(H(p)) with p_i = sigma_i / sum(sigma)  (Roy & Vetterli)
        spectral_entropy:    H(p) / log(rank_max)  (normalized to [0, 1])
        participation_ratio: (sum sigma)^2 / sum(sigma^2)
        top1_singular:      sigma_1 (spectral norm)
        cumulative_energy_top_k for k in {1, 2, 4, 8}: energy captured
        singular_values:    full spectrum (descending)
    """
    with torch.no_grad():
        s = torch.linalg.svdvals(weight.float())
        s = s[s > 1e-12]
        if len(s) == 0:
            return {'stable_rank': 0.0, 'effective_rank': 0.0,
                    'spectral_entropy': 0.0, 'participation_ratio': 0.0,
                    'top1_singular': 0.0, 'cumulative_energy': {},
                    'singular_values': []}

        fro2 = float((s ** 2).sum())
        top1 = float(s[0])
        stable_rank = fro2 / (top1 ** 2 + 1e-12)

        p = s / s.sum()
        ent = float(-(p * p.clamp_min(1e-12).log()).sum())
        effective_rank = float(np.exp(ent))
        max_rank = len(s)
        spectral_entropy = ent / np.log(max_rank)

        pr = float((s.sum() ** 2) / ((s ** 2).sum() + 1e-12))

        energy = (s ** 2) / fro2
        cum = {}
        for k in [1, 2, 4, 8]:
            if k <= max_rank:
                cum[k] = float(energy[:k].sum())

        return {
            'stable_rank': stable_rank,
            'effective_rank': effective_rank,
            'spectral_entropy': spectral_entropy,
            'participation_ratio': pr,
            'top1_singular': top1,
            'cumulative_energy': cum,
            'singular_values': s.tolist(),
        }


def compute_selectivity(hidden_acts: np.ndarray, labels: np.ndarray,
                        top_ratio_threshold: float = 0.5) -> dict:
    """
    Neuron class-selectivity via top-1/top-2 mean-activation dominance,
    replacing the Pearson-indicator 'monosemanticity' metric that confounds
    selectivity with firing frequency.

    For each neuron j:
        S_j = (mu_top1 - mu_top2) / (mu_top1 + eps)

    Returns:
        mean_selectivity, frac_selective (S_j > threshold),
        per-neuron top-1 class assignments, and selectivity values.
    """
    n_neurons = hidden_acts.shape[1]
    classes = np.unique(labels)
    class_means = {int(c): hidden_acts[labels == c].mean(0) for c in classes}

    mu = np.stack([class_means[int(c)] for c in classes])  # (C, H)
    sorted_mu = np.sort(mu, axis=0)[::-1]                   # per-neuron descending
    top1, top2 = sorted_mu[0], sorted_mu[1]
    sel = (top1 - top2) / (np.abs(top1) + 1e-9)

    top_classes = np.argmax(mu, axis=0)
    return {
        'mean_selectivity': float(sel.mean()),
        'std_selectivity': float(sel.std()),
        'frac_selective': float((sel > top_ratio_threshold).mean()),
        'threshold': top_ratio_threshold,
        'top_classes': top_classes.tolist(),
        'selectivity_values': sel.tolist(),
    }


def compute_weight_norms_spectral(model) -> dict:
    """Per-layer norms + spectral metrics for every Linear layer."""
    out = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight is not None:
                W = module.weight.data
                key = name if name else 'linear'
                out[f'{key}_frobenius'] = float(W.norm(p='fro'))
                out[f'{key}_spectral_norm'] = float(torch.linalg.svdvals(W.float())[0])
                out[f'{key}_stable_rank'] = compute_spectral_metrics(W)['stable_rank']
                out[f'{key}_effective_rank'] = compute_spectral_metrics(W)['effective_rank']
                out[f'{key}_spectral_entropy'] = compute_spectral_metrics(W)['spectral_entropy']
    return out


def evaluate_class_accuracy(
    model: nn.Module, loader: DataLoader, class_id: int, device: str = 'cpu'
) -> float:
    """Evaluate accuracy on a specific class."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            mask = y == class_id
            if mask.any():
                x, y = x[mask].to(device), y[mask].to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

    return correct / max(total, 1)
