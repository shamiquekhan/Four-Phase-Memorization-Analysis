"""
verify_consistency.py — Verify documentation values against result artifacts.

v2 behavior (hardened):
  - Missing REQUIRED result artifacts are a hard failure (FAIL, exit 1),
    not a silent skip. If the v2 analysis has not been run, the verifier
    must say so loudly instead of reporting "0 passed, 0 failed".
  - Every checked number is recomputed from the raw per-seed values in
    outputs/analysis/ and compared against RESULTS_v2.md / README.md /
    PAPER.md within tolerance.

Run after pipeline or documentation changes to catch stale values.
"""

import json
import re
import statistics
import sys
from pathlib import Path

TOLERANCE = 0.02          # absolute tolerance for summary-scale metrics
DOC_LINK_TOLERANCE = 0.015  # docs round to 2-3 decimals; allow rounding slack
REPO = Path(__file__).parent.parent

REQUIRED_ARTIFACTS = [
    'outputs/analysis/phase1_clean/phase1_results.json',
    'outputs/analysis/phase1_corrupted/phase1_results.json',
    'outputs/analysis/phase2_corrupted/phase2_results.json',
    'outputs/analysis/phase3_corrupted/phase3_results.json',
    'outputs/analysis/phase4_noise_0.2/phase4_results.json',
    'outputs/analysis/multiclass_rome/multiclass_rome_results.json',
    'outputs/analysis/multiclass_rome/multilayer_rome_comparison.json',
    'outputs/analysis/multiclass_rome/random_baseline_results.json',
]


def load_json(rel_path):
    return json.loads((REPO / rel_path).read_text())


def mean(xs):
    return sum(xs) / len(xs)


def check(name, reported, computed, tol, results, passed, failed):
    if reported is None:
        results.append(f'  WARN: {name}: value not found in docs (skipped)')
        return passed, failed
    if abs(reported - computed) <= tol:
        results.append(f'  PASS: {name}: doc {reported:.4g} ≈ artifact {computed:.4g}')
        return passed + 1, failed
    results.append(f'  FAIL: {name}: doc {reported:.4g} ≠ artifact {computed:.4g} (tol {tol})')
    return passed, failed + 1


def extract_doc_number(text, pattern, cast=float, group=1):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        # Docs use the Unicode minus (\u2212); normalize before casting.
        return cast(m.group(group).replace('\u2212', '-'))
    except (ValueError, TypeError):
        return None


def main():
    passed = 0
    failed = 0
    results = []

    print('=== CONSISTENCY VERIFICATION (v2, hard-fail mode) ===\n')

    # ---- 0. Required artifacts -------------------------------------------
    missing = [p for p in REQUIRED_ARTIFACTS if not (REPO / p).exists()]
    if missing:
        for p in missing:
            results.append(f'  FAIL: missing required artifact: {p}')
            failed += 1
        results.append('\n  The v2 analysis outputs are incomplete. Run:')
        results.append('    python reproduce_all.py --skip-training   (with checkpoints present)')
        results.append('  or retrain first: python reproduce_all.py')
        print('\n'.join(results))
        print(f'\n=== SUMMARY: {passed} passed, {failed} failed ===')
        return False

    results.append(f'  PASS: all {len(REQUIRED_ARTIFACTS)} required v2 artifacts present')
    passed += 1

    readme = (REPO / 'README.md').read_text()
    results_v2 = (REPO / 'RESULTS_v2.md').read_text()

    # ---- 1. Phase 1: training performance + weight geometry --------------
    p1 = {cond: load_json(f'outputs/analysis/phase1_{cond}/phase1_results.json')
          for cond in ('clean', 'corrupted')}

    def p1_means(cond):
        rows = [p1[cond][s][0] for s in p1[cond]]
        return {f: mean([r[f] for r in rows])
                for f in rows[0]
                if isinstance(rows[0][f], (int, float))}

    clean, corr = p1_means('clean'), p1_means('corrupted')

    # README headline numbers (v2 table)
    # | **Train Accuracy** | 96.41% | 75.45% (noisy labels) | 2e-17 |
    m = re.search(r'\|\s*\*\*Train Accuracy\*\*\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%', readme)
    if m:
        p, failed = check('README train acc (clean)', float(m.group(1)), clean['train_acc'], TOLERANCE, results, passed, failed)
        passed = p
        p, failed = check('README train acc (corrupted)', float(m.group(2)), corr['train_acc'], TOLERANCE, results, passed, failed)
        passed = p
    else:
        results.append('  WARN: README Train Accuracy row not found')

    m = re.search(r'\|\s*\*\*Test Accuracy\*\*\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%', readme)
    if m:
        p, failed = check('README test acc (clean)', float(m.group(1)), clean['test_acc'], TOLERANCE, results, passed, failed)
        passed = p
        p, failed = check('README test acc (corrupted)', float(m.group(2)), corr['test_acc'], TOLERANCE, results, passed, failed)
        passed = p
    else:
        results.append('  WARN: README Test Accuracy row not found')

    # RESULTS_v2 phase-1 table
    checks_v2 = [
        ('RESULTS_v2 train acc (clean)', r'Train Acc \(noisy labels\)\s*\|\s*([\d.]+)%', 'clean', 'train_acc'),
        ('RESULTS_v2 train acc (corrupted)', r'Train Acc \(noisy labels\)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%', 'corrupted', 'train_acc', 2),
        ('RESULTS_v2 FC1 spectral (clean)', r'FC1 spectral norm\s*\|\s*([\d.]+)', 'clean', 'fc1_spectral'),
        ('RESULTS_v2 FC2 spectral (clean)', r'FC2 spectral norm\s*\|\s*([\d.]+)', 'clean', 'fc2_spectral'),
        ('RESULTS_v2 FC2 spectral (corrupted)', r'FC2 spectral norm\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', 'corrupted', 'fc2_spectral', 2),
        ('RESULTS_v2 FC2 stable rank (clean)', r'FC2 stable rank\s*\|\s*([\d.]+)', 'clean', 'fc2_stable_rank'),
        ('RESULTS_v2 FC2 stable rank (corrupted)', r'FC2 stable rank\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', 'corrupted', 'fc2_stable_rank', 2),
        ('RESULTS_v2 FDR (corrupted)', r'FDR \(h=16\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', 'corrupted', 'fdr', 2),
    ]
    for name, pattern, cond, field, *grp in checks_v2:
        reported = extract_doc_number(results_v2, pattern, group=grp[0] if grp else 1)
        if reported is None:
            results.append(f'  WARN: {name}: pattern not found in RESULTS_v2.md')
            continue
        p, failed = check(name, reported, p1_means(cond)[field] if cond in ('clean', 'corrupted') else reported,
                          TOLERANCE, results, passed, failed)
        passed = p

    # ---- 2. Phase 2: cross-model drift -----------------------------------
    p2 = load_json('outputs/analysis/phase2_corrupted/phase2_results.json')

    drift_rows = list(p2['cross_model_drift'].values())
    for layer in ('fc1_post', 'output'):
        computed = mean([d[layer] for d in drift_rows])
        reported = extract_doc_number(
            results_v2,
            rf'\|\s*{layer}\s*\|\s*([\d.]+)\s*\|')
        if layer == 'fc1_post':
            reported = extract_doc_number(results_v2, r'\|\s*fc1_post\s*\|\s*\*\*([\d.]+)\*\*')
        elif layer == 'output':
            reported = extract_doc_number(results_v2, r'\|\s*output\s*\|\s*\*\*?([\d.]+)\*\*?\s*\|')
        p, failed = check(f'RESULTS_v2 drift {layer}', reported, computed, DOC_LINK_TOLERANCE, results, passed, failed)
        passed = p

    # ---- 3. Phase 3: memorization dynamics -------------------------------
    p3 = load_json('outputs/analysis/phase3_corrupted/phase3_results.json')
    rows3 = list(p3.values())

    mem_frac = mean([r['behavioral']['memorized_fraction_of_changed'] for r in rows3])
    p, failed = check('README behavioral memorization (frac of changed)',
                      extract_doc_number(readme, r'memorization \(changed AND fits noisy label AND ≠ original\) is only\s*\*\*([\d.]+)%'),
                      mem_frac * 100, TOLERANCE, results, passed, failed)
    passed = p

    fit_orig = mean([r['behavioral']['changed_fit_original_label'] for r in rows3])
    p, failed = check('README/RESULTS_v2 changed→original-label fit',
                      extract_doc_number(results_v2, r'Changed examples fitting ORIGINAL label\s*\|\s*\*\*([\d.]+)%'),
                      fit_orig * 100, TOLERANCE, results, passed, failed)
    passed = p

    group_align = mean([r['gradient_alignment_group'] for r in rows3])
    p, failed = check('RESULTS_v2 group gradient alignment',
                      extract_doc_number(results_v2, r'Group gradient alignment \(clean vs changed, fc1\)\s*\|\s*\*+(−?-?[\d.]+)'),
                      group_align, DOC_LINK_TOLERANCE, results, passed, failed)
    passed = p

    tracin_cos = mean([r['tracin_self_influence']['mean_cosine_noisy_vs_orig_grad'] for r in rows3])
    anti_frac = mean([r['tracin_self_influence']['frac_anti_aligned'] for r in rows3])
    p, failed = check('README TracIn cos',
                      extract_doc_number(readme, r'TracIn\s*cos\(noisy-grad, orig-grad\) = (−?-?[\d.]+)'),
                      tracin_cos, DOC_LINK_TOLERANCE, results, passed, failed)
    passed = p
    p, failed = check('RESULTS_v2 fraction anti-aligned',
                      extract_doc_number(results_v2, r'Fraction of changed examples anti-aligned\s*\|\s*\*\*([\d.]+)%'),
                      anti_frac * 100, TOLERANCE, results, passed, failed)
    passed = p

    # ---- 4. Phase 4: delta-norm dose-response ----------------------------
    clean_p4 = load_json('outputs/analysis/phase4_clean/phase4_results.json')
    corr_p4 = load_json('outputs/analysis/phase4_noise_0.2/phase4_results.json')

    def fc2_delta_norms(data):
        return [data[s]['fc2'][str(cls)]['delta_norm']
                for s in data for cls in range(10)]

    clean_mean = mean(fc2_delta_norms(clean_p4))
    corr_mean = mean(fc2_delta_norms(corr_p4))
    ratio = clean_mean / corr_mean if corr_mean else float('nan')

    p, failed = check('RESULTS_v2 fc2 delta-norm (clean ref 0.801)',
                      extract_doc_number(results_v2, r'\(Clean reference: ([\d.]+)\.\)'),
                      clean_mean, DOC_LINK_TOLERANCE, results, passed, failed)
    passed = p
    p, failed = check('RESULTS_v2 fc2 delta-norm ratio @20% (2.16×)',
                      extract_doc_number(results_v2, r'\|\s*20%\s*\|\s*[\d.]+\s*\|\s*([\d.]+)×'),
                      ratio, 0.05, results, passed, failed)
    passed = p

    # Noise sweep monotonicity: corrupted delta-norm must decrease with noise rate
    sweep = []
    for rate in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        path = REPO / f'outputs/analysis/phase4_noise_{rate}/phase4_results.json'
        if path.exists():
            sweep.append((rate, mean(fc2_delta_norms(load_json(str(Path('outputs/analysis') / f'phase4_noise_{rate}/phase4_results.json'))))))
    if len(sweep) >= 3:
        values = [v for _, v in sweep]
        monotone = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        if monotone:
            results.append(f'  PASS: noise sweep monotone decreasing across {len(sweep)} rates')
            passed += 1
        else:
            results.append(f'  FAIL: noise sweep not monotone: {sweep}')
            failed += 1
    else:
        results.append(f'  WARN: noise sweep incomplete ({len(sweep)} rates on disk)')

    # ---- 5. Multi-class ROME: EDIT/EVAL firewall invariants --------------
    mc = load_json('outputs/analysis/multiclass_rome/multiclass_rome_results.json')
    ml = load_json('outputs/analysis/multiclass_rome/multilayer_rome_comparison.json')

    # fc1-only edits must recover exactly 0 on the held-out EVAL split
    fc1_recoveries = [v for cfg in ml.values() for v in cfg['fc1_only']['recovery']]
    if all(v == 0.0 for v in fc1_recoveries):
        results.append(f'  PASS: fc1-only recovery is exactly 0 in all {len(fc1_recoveries)} runs (EVAL firewall)')
        passed += 1
    else:
        nonzero = [v for v in fc1_recoveries if v != 0.0]
        results.append(f'  FAIL: fc1-only recovery nonzero for {len(nonzero)}/{len(fc1_recoveries)} runs: {nonzero[:5]}')
        failed += 1

    # fc2-only recovery must be positive for every config
    bad = [c for c, cfg in ml.items() if mean(cfg['fc2_only']['recovery']) <= 0]
    if not bad:
        results.append(f'  PASS: fc2-only recovery > 0 in all {len(ml)} configs')
        passed += 1
    else:
        results.append(f'  FAIL: fc2-only recovery ≤ 0 for configs {bad}')
        failed += 1

    # Random null must be degenerate at 0 (doc claims honest reporting)
    rb = load_json('outputs/analysis/multiclass_rome/random_baseline_results.json')
    random_vals = [v for cfg in rb.values() if isinstance(cfg, dict)
                   for v in cfg.get('random_mean', []) if isinstance(v, (int, float))]
    if random_vals and all(v == 0.0 for v in random_vals):
        results.append(f'  PASS: random-null recovery is degenerate 0.0 ({len(random_vals)} runs, as documented)')
        passed += 1
    else:
        results.append(f'  WARN: random-null structure unexpected ({len(random_vals)} numeric values)')

    # ---- 6. Scaling sweep (v2): required if the analysis has been run -----.
    # Once outputs/analysis/scaling*/scaling_analysis.json exist, they must be
    # complete (7 widths x 5 seeds) and consistent with the docs.
    for label, path in [('scaling (clean)', 'outputs/analysis/scaling/scaling_analysis.json'),
                        ('scaling (corrupted)', 'outputs/analysis/scaling_corrupted/scaling_analysis.json')]:
        spath = REPO / path
        if not spath.exists():
            results.append(f'  WARN: {label} not present (run analyze_scaling.py)')
            continue
        sc = load_json(path)
        widths = ['16', '32', '64', '128', '256', '512', '1024']
        incomplete = [w for w in widths
                      if w not in sc or len(sc[w].get('fdr', [])) < 5]
        if incomplete:
            results.append(f'  FAIL: {label} incomplete for widths {incomplete} (need 5 seeds)')
            failed += 1
            continue
        results.append(f'  PASS: {label} complete: 7 widths x 5 seeds')
        passed += 1
        # Cross-check the h=16 FDR values against the docs (RESULTS_v2 scaling table).
        fdr_pat = (r'\|\s*16\s+\|\s+[\d.]+ \u00b1 [\d.]+\s+\|\s+([\d.]+)') if 'corrupted' in label \
            else (r'\|\s*16\s+\|\s+([\d.]+)')
        reported = extract_doc_number(results_v2, fdr_pat)
        if reported is not None:
            p_, failed_ = check(f'{label} h=16 FDR vs RESULTS_v2', reported,
                                mean(sc['16']['fdr']), TOLERANCE, results, passed, failed)
            passed, failed = p_, failed_
        # Clean-condition FDR must fall with width (monotone non-increasing).
        if 'corrupted' not in label:
            fdr_by_w = [mean(sc[w]['fdr']) for w in widths]
            if all(fdr_by_w[i] >= fdr_by_w[i + 1] for i in range(len(fdr_by_w) - 1)):
                results.append('  PASS: clean FDR monotonically non-increasing with width')
                passed += 1
            else:
                results.append(f'  FAIL: clean FDR not monotone: {fdr_by_w}')
                failed += 1

    # ---- summary ----------------------------------------------------------
    print('\n'.join(results))
    print(f'\n=== SUMMARY: {passed} passed, {failed} failed ===')
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
