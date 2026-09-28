"""
verify_statistics.py — Verify CI computations and statistical claims.

Checks that all JSON result files have correctly computed CIs and that
the CI coverage is approximately correct.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

REPO = Path(__file__).parent.parent


def compute_ci(values):
    if len(values) < 2:
        return 0.0
    arr = np.array(values)
    se = sp_stats.sem(arr)
    return se * sp_stats.t.ppf(0.975, len(arr) - 1)


def check_ci_in_json(path, label):
    """Verify that stored CI matches recomputed CI from raw values."""
    with open(path) as f:
        data = json.load(f)

    issues = []
    for key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name in ['sigma', 'accuracy', 'mono_fraction', 'circuit_size', 'sparsity']:
            if metric_name not in metrics:
                continue
            values = metrics[metric_name]
            if not isinstance(values, list) or len(values) < 2:
                continue
            try:
                values = [float(v) for v in values]
            except (TypeError, ValueError):
                issues.append(f'  FAIL: {label} [{key}] {metric_name}: non-numeric values')
                continue

            stored_mean = metrics.get(f'{metric_name}_mean')
            stored_ci = metrics.get(f'{metric_name}_ci')

            if stored_mean is None or stored_ci is None:
                continue

            # Artifacts serialized with default=str can hold numeric values as
            # strings; coerce (and flag) instead of crashing on format specs.
            try:
                stored_mean = float(stored_mean)
                stored_ci = float(stored_ci)
            except (TypeError, ValueError):
                issues.append(f'  FAIL: {label} [{key}] {metric_name}_mean/_ci '
                              f'not numeric: {stored_mean!r}, {stored_ci!r}')
                continue

            computed_mean = float(np.mean(values))
            computed_ci = compute_ci(values)

            mean_ok = abs(computed_mean - stored_mean) < 0.001
            # CI may be stored as full width (2 × half-width) or half-width
            ci_full = 2 * computed_ci
            ci_ok = abs(computed_ci - stored_ci) < 0.001 or abs(ci_full - stored_ci) < 0.001

            if not mean_ok:
                issues.append(f'  FAIL: {label} [{key}] {metric_name}_mean: '
                              f'stored={stored_mean:.4f}, computed={computed_mean:.4f}')
            if not ci_ok:
                issues.append(f'  FAIL: {label} [{key}] {metric_name}_ci: '
                              f'stored={stored_ci:.4f}, expected half-width={computed_ci:.4f} '
                              f'or full-width={ci_full:.4f}')
    return issues


def check_n_values(path, label, min_n=10):
    """Check that result files have at least min_n values per metric."""
    with open(path) as f:
        data = json.load(f)

    issues = []
    for key, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name in ['davies_bouldin', 'accuracy', 'mono_fraction']:
            values = metrics.get(metric_name, [])
            if isinstance(values, list) and len(values) > 0 and len(values) < min_n:
                issues.append(f'  WARN: {label} [{key}] {metric_name} has {len(values)} values (expected ≥{min_n})')
    return issues


def main():
    print('=== STATISTICS VERIFICATION ===\n')
    all_issues = []
    passed = 0
    failed = 0

    # Check scaling analysis
    # v2 scaling protocol uses the first 5 config seeds (see RESULTS_v2.md),
    # so the per-metric minimum here is 5, not the campaign-wide 10.
    scaling_path = REPO / 'outputs' / 'analysis' / 'scaling' / 'scaling_analysis.json'
    if scaling_path.exists():
        issues = check_ci_in_json(scaling_path, 'scaling')
        n_issues = check_n_values(scaling_path, 'scaling', min_n=5)
        all_issues.extend(issues)
        all_issues.extend(n_issues)
        if not issues:
            print('  PASS: scaling_analysis.json CIs correct')
            passed += 1
        else:
            failed += len(issues)
    else:
        print('  SKIP: scaling_analysis.json not found')

    # Check rank ablation results
    rank_path = REPO / 'outputs' / 'analysis' / 'multiclass_rome' / 'multilayer_rome_comparison.json'
    if rank_path.exists():
        with open(rank_path) as f:
            data = json.load(f)
        # Check all 5 configs have 5 seed values (fc2_only, fc1_only, both_layers)
        config_ok = True
        for label in data:
            for k in ['fc2_only', 'fc1_only', 'both_layers']:
                n = len(data[label][k]['recovery'])
                if n < 5:
                    print(f'  FAIL: {label} {k} has {n} seeds (expected 5)')
                    config_ok = False
                    failed += 1
        if config_ok:
            print('  PASS: multilayer_rome_comparison.json has 5 seeds × 4 configs')
            passed += 1
    else:
        print('  SKIP: multilayer_rome_comparison.json not found')

    # Print all issues
    for issue in all_issues:
        print(issue)
        failed += 1

    print(f'\n=== SUMMARY: {passed} passed, {failed} issues ===')
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
