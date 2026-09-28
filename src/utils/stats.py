"""Statistical utilities: CI computation, effect sizes, multiplicity control.

v2 STATISTICAL CONTRACT (one convention for every script):
  - Primary unit of replication: the random seed.
  - For each seed we compute a paired (clean, corrupted) difference when a
    comparison is being made; never compare unpaired across-seed means when
    pairing is available.
  - Reported interval: Student-t CI on per-seed values (labelled as such).
    bootstrap_ci() is provided as a robustness check; scripts must state
    which interval they print. Paper text must match (no "bootstrap" wording
    unless bootstrap_ci was used).
  - Effect sizes: paired dz = mean(delta) / std(delta).
  - Multiplicity: use holm_correction() for families of p-values instead of
    reporting dozens of raw p < 0.0001 values.
  - Seed sources: initialization, corruption, loader seeds are DECOUPLED.
    Load from config/experiment_config.yaml for the canonical lists.
"""

import numpy as np
import torch
import yaml
from scipy import stats
from typing import Callable, Dict, List, Any, Tuple
from pathlib import Path


# Load canonical seed lists from config (single source of truth)
_CONFIG_PATH = Path(__file__).parent.parent.parent / 'configs' / 'experiment_config.yaml'
with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

INITIALIZATION_SEEDS = _cfg.get('initialization_seeds', [42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999])
CORRUPTION_SEEDS = _cfg.get('corruption_seeds', [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010])
LOADER_SEEDS = _cfg.get('loader_seeds', [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010])

# Legacy alias for backward compatibility
SEEDS = INITIALIZATION_SEEDS


def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Student-t confidence interval for the mean across seeds.
    (Explicitly NOT bootstrap; label as "Student-t CI" in all outputs.)
    """
    values = list(values)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    if n < 2:
        return float(np.mean(values)), float(np.mean(values)), float(np.mean(values))

    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(mean), float(mean - ci), float(mean + ci)


def bootstrap_ci(values: List[float], confidence: float = 0.95,
                n_resamples: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean (robustness check companion to compute_ci)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        m = float(values.mean()) if len(values) else float('nan')
        return m, m, m
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_effect_size_dz(values_a: List[float], values_b: List[float]) -> float:
    """Paired standardized effect size: dz = mean(a-b) / sd(a-b)."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    if len(a) != len(b) or len(a) < 2:
        return float('nan')
    d = a - b
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return float('nan')
    return float(d.mean() / sd)


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (order-preserving output).

    Input p-values must be a family of pre-specified comparisons; do not
    feed every exploratory number through this.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p[idx]
        adj = min(1.0, max(adj, prev))
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def run_with_seeds(experiment_fn: Callable, seeds: List[int] = None, **kwargs) -> Dict[str, List[float]]:
    """
    Run any experiment function across multiple seeds.

    Args:
        experiment_fn: Callable that takes seed as first arg and returns dict of metrics
        seeds: List of integer seeds (defaults to SEEDS)
        **kwargs: Additional arguments passed to experiment_fn

    Returns:
        Dict of metric_name -> list of values across seeds
    """
    if seeds is None:
        seeds = SEEDS

    results: Dict[str, List[float]] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        metrics = experiment_fn(seed=seed, **kwargs)
        for k, v in metrics.items():
            results.setdefault(k, []).append(float(v))
    return results


def aggregate_results(raw_results: Dict[str, List[float]], confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Aggregate raw multi-seed results into mean ± CI format.

    Args:
        raw_results: Dict of metric_name -> list of values across seeds
        confidence: Confidence level for CI

    Returns:
        Dict of metric_name -> {'mean': float, 'ci_lower': float, 'ci_upper': float, 'std': float}
    """
    aggregated = {}
    for metric, values in raw_results.items():
        mean, ci_lower, ci_upper = compute_ci(values, confidence)
        aggregated[metric] = {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': float(np.std(values)),
            'values': values
        }
    return aggregated


def format_mean_ci(aggregated: Dict[str, Dict[str, float]], metric: str, precision: int = 3) -> str:
    """Format a metric as 'mean ± ci' string."""
    m = aggregated[metric]
    return f"{m['mean']:.{precision}f} ± {m['ci_upper'] - m['mean']:.{precision}f}"


def paired_t_test(values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of measurements.

    Returns:
        t-statistic, p-value
    """
    t_stat, p_val = stats.ttest_rel(values_a, values_b)
    return float(t_stat), float(p_val)


def one_sample_t_test(values: List[float], popmean: float = 0.0) -> Tuple[float, float]:
    """
    Perform one-sample t-test against a population mean.

    Returns:
        t-statistic, p-value
    """
    t_stat, p_val = stats.ttest_1samp(values, popmean)
    return float(t_stat), float(p_val)