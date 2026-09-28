# Paper-to-Code Mapping

**Working title:** Structural Fingerprints of Label Memorization in Shallow Neural Networks

> **⚠️ Status: v2.** All empirical values below come from the **v2 campaign**
> (90 models retrained with corruption provenance and the EDIT/EVAL firewall;
> 10 paired seeds, 95% Student-t CIs). v0 values are **not** reproduced here —
> see the audit trail in [RESULTS_v2.md](RESULTS_v2.md) for what changed and why.
> This file is kept in lockstep with `README.md` and `RESULTS_v2.md` and is
> cross-checked by `scripts/verify_consistency.py` (hard-fail mode) and
> `scripts/verify_statistics.py`.

Maps claims and results to specific source files and functions.
Architecture: MNISTNet 784→16→10 unless stated.

## Abstract

| Claim | Empirical value (v2) | Code location |
|-------|---------------------|---------------|
| Behavioral memorization of 20% label noise | **1.1% of changed examples** (0.22% of all); 92.8% of changed examples still fit their ORIGINAL label | `src/analysis/phase3_influence.py:compute_behavioral_memorization()` |
| Class separability under corruption (FDR, h=16) | 0.84 (clean) → **1.50** (corrupted), p=3e-06 | `src/utils/metrics.py:compute_fdr()` |
| Corruption lowers operator scale, not rank | FC2 spectral norm −47% (p=3e-09); FC2 stable rank unchanged (p=0.32), effective rank +0.2 (p=3e-03) | `src/utils/metrics.py:compute_spectral_metrics()` |
| Cross-model representational drift (CKA) | Layer-graded: output 0.496 > fc1_post 0.360 > fc1_pre 0.233; fc1_post exceeds the seed noise floor (0.29), output drift ≈ corrupted-condition seed floor (0.48) | `src/analysis/phase2_representation.py:cross_model_cka()`, `cross_seed_cka_controls()` |
| Gradient anti-alignment (group, fc1) | **−0.88 ± 0.03** | `src/analysis/phase3_influence.py:compute_group_gradient_alignment()` |
| TracIn per-example anti-alignment | cos = **−0.33 ± 0.09**; 76% of changed examples anti-aligned | `src/analysis/phase3_influence.py:compute_tracin_scores()` |
| Rank-one delta-norm dose-response (EDIT split) | 1.78× (5%) → 2.66× (50%), monotone, all p≤1e-06 (clean ref 0.801) | `src/analysis/phase4_rome.py:compute_delta_norms()` |
| Rank-one intervention on held-out EVAL data | fc2-only recovery **+10.9 to +16.9pp** (all p<0.02); fc1-only exactly **0.0pp** in every config; sequential fc2→fc1 (3.5–8.1pp) worse than fc2-only | `src/analysis/multiclass_rome.py:run_rank1_experiment()`, `src/analysis/multilayer_rome.py:run_multilayer_rome_experiment()` |
| Random-norm-matched null | Degenerate: all nulls recover 0.0pp (zero variance); z undefined; empirical p at the permutation floor 1/21 | `src/analysis/multiclass_rome.py:run_rank1_with_random_baseline()` |

## Phase 1: Weight Geometry & Spectral Structure

| Result | File | Function |
|--------|------|----------|
| Spectral norms, Frobenius norms, stable/effective rank, spectral entropy, cumulative energy | `src/analysis/phase1_basic.py` | `compute_weight_norms()`, `analyze_checkpoint()` |
| Stable rank unchanged, effective rank slightly up under corruption | `src/utils/metrics.py` | `compute_spectral_metrics()` |
| FDR increase under corruption (0.84 → 1.50) | `src/utils/metrics.py` | `compute_fdr()` |

**Framing note:** the v0 inference "spectral norm drop ⇒ lower-rank memorization"
is withdrawn. Direct rank metrics show corruption reduces weight *scale* while
rank structure is preserved.

## Phase 2: Cross-Model Representational Drift

| Result | File | Function |
|--------|------|----------|
| Drift = 1 − CKA(clean layer, corrupted layer) on identical inputs | `src/analysis/phase2_representation.py` | `cross_model_cka()` |
| Seed noise-floor controls (clean↔clean, corr↔corr pairwise) | `src/analysis/phase2_representation.py` | `cross_seed_cka_controls()` |
| Within-model layer CKA (secondary) | `src/analysis/phase2_representation.py` | `analyze_representations()` |

**v2 primary analysis is cross-model drift with seed floors.** The v0 claim
"distortion localizes at the ReLU via within-model CKA" is superseded: the
clearest cross-model signature is at fc1_post (0.360 vs floor 0.294); output
drift is indistinguishable from the corrupted-condition seed floor, i.e. label
noise mainly inflates readout variance across seeds.

## Phase 3: Memorization Dynamics (non-circular)

| Result | File | Function |
|--------|------|----------|
| Behavioral memorization (changed AND fits noisy AND ≠ original) | `src/analysis/phase3_influence.py` | `compute_behavioral_memorization()` |
| Loss gap on ground-truth corrupted indices | `src/analysis/phase3_influence.py` | `compute_loss_gap()` |
| Group gradient alignment, clean vs changed examples | `src/analysis/phase3_influence.py` | `compute_group_gradient_alignment()` |
| TracIn-style per-example self-influence | `src/analysis/phase3_influence.py` | `compute_tracin_scores()` |
| Temporal (per-epoch) trajectory of the above | `src/analysis/temporal_dynamics.py` | `compute_tracin()`, `evaluate_group_gradient_alignment()` |

Influence functions (per-batch Hessian, CG) were **demoted to future work** in
v2; TracIn-style scoring plus the behavioral definition replace them.

## Phase 4: Rank-One Interventions (ROME-inspired, EDIT/EVAL firewall)

| Result | File | Function |
|--------|------|----------|
| Closed-form class-mean rank-one edit (NOT Meng et al. ROME: no causal tracing, no key/value covariance) | `src/analysis/multiclass_rome.py` | `compute_rank1_edit()`, `apply_rank1_edit()` |
| Edit built on EDIT half, evaluated on disjoint EVAL half | `src/data/splits.py` | `split_edit_eval()` (+ provenance saved per run) |
| Delta-norm dose-response vs noise rate (6 rates, monotone) | `src/analysis/phase4_rome.py` | `compute_delta_norms()` |
| Multi-config validation (7→1, 1→7, 5→6, 0→8), fc2-only / fc1-only / sequential | `src/analysis/multiclass_rome.py`, `src/analysis/multilayer_rome.py` | `run_rank1_experiment()`, `run_multi_layer_rank1()`, `sequential_multilayer_rome()` |
| Norm-matched random null | `src/analysis/multiclass_rome.py` | `apply_random_edit()`, `run_rank1_with_random_baseline()` |
| Rank ablation (SVD rank-k) | `src/analysis/rank_ablation.py` | `run_rank_ablation()` |

**Headline v2 numbers:** fc2-only recovery on untouched EVAL data +10.9 to
+16.9pp (p<0.02 across configs); fc1-only recovery is exactly 0.0pp in every
config; sequential fc2→fc1 underperforms fc2-only. The random null is
degenerate (0.0pp, zero variance) and is reported as such — no infinite
signal-ratio claims.

## Scaling & Phase 5 (pending v2 re-derivation)

| Component | Status | Code |
|-----------|--------|------|
| Width scaling (FDR, monosemanticity, circuit size, sparsity) | **v0 protocol; rerun pending (v2.1)** — v0 values must not be cited as established | `src/scaling/train_scaling.py`, `analyze_scaling.py`, `src/utils/metrics.py:compute_monosemanticity()`, `compute_circuit_sparsity()` |
| CIFAR-10 replication | **v2 protocol implemented; rerun pending** | `src/analysis/cifar_replication.py`, `rome_cifar10.py`, `analyze_cifar10.py` |
| Phase 5 LoRA vs rank-one comparison | Code ready with matched edit sets; rerun pending | `src/analysis/phase5_lora_comparison.py:run_lora_correction()`, `run_rank1_correction()`; `src/analysis/subspace_overlap.py` |
| Temporal dynamics instrumentation | Code ready; wired via `src/training/train_temporal.py` + `src/analysis/temporal_dynamics.py` | epoch checkpoints → drift / delta-norm / TracIn per epoch |

## Statistical Validation

| Item | Location |
|------|----------|
| 95% Student-t CI | `src/utils/stats.py:compute_ci()` |
| Paired t-tests, Holm correction, paired d_z | `src/utils/stats.py` |
| Seeds: single source of truth (config list, never `range(n)`) | `configs/experiment_config.yaml: seeds`, `src/utils/stats.py:SEEDS` |
| Corruption provenance (selected == changed, guaranteed) | `src/data/corruption.py:CorruptionProvenance` |
| EDIT/EVAL split provenance | `src/data/splits.py:SplitProvenance` |
| Consistency gate (hard-fail on missing artifacts; doc↔JSON reconciliation) | `scripts/verify_consistency.py` |
| CI / seed-count gate | `scripts/verify_statistics.py` |
| Scientific-invariant tests | `tests/test_invariants.py`, `tests/test_metrics.py` (54 tests) |

## Related Work

**Zhang et al. (2017) — "Understanding deep learning requires rethinking generalization"**
Networks can fit random labelings of training data. *Our delta*: Zhang et al.
establish that fitting is possible; we measure **when it actually happens**.
Under the v2 behavioral definition, a capacity-limited model at 20% noise fits
only 1.1% of changed examples while retaining 92.8% of their original labels —
fitting capacity does not imply fitting behavior, and memorization claims must
be conditioned on the regime where noisy-label fit occurs.

**Arpit et al. (2017) — "A Closer Look at Memorization in Deep Networks"**
Networks learn simple patterns first and memorize noise later. *Our delta*: we
replace loss-quantile (circular) memorization definitions with ground-truth
corruption provenance saved per run, and add causal interventions.

**Koh & Liang (2017) — "Understanding Black-box Predictions via Influence Functions"**
Per-sample attribution via inverse-Hessian products. *Our delta (revised in
v2)*: influence functions with per-batch Hessians proved unreliable at this
scale; we demote them to future work and use TracIn-style gradient
self-influence plus a behavioral definition. We additionally show that
attribution alone does not localize memorization to a layer — the rank-one
intervention does.

**Meng et al. (2022) — "Locating and Editing Factual Associations in GPT"**
ROME for causal localization via rank-one edits in LLMs. *Our delta*: we use a
**ROME-inspired closed-form class-mean rank-one edit** (no causal tracing, no
key/value covariance constraint) in shallow classifiers, with an EDIT/EVAL
firewall so recovery is measured on genuinely held-out data. Edits act only at
the output-adjacent layer; fc1-only edits recover exactly nothing.

**Kornblith et al. (2019) — "Similarity of Neural Network Representations Revisited"**
CKA as a representation-similarity measure. *Our delta (revised in v2)*: we
use cross-model CKA drift **with same-condition seed noise floors** rather
than within-model layer similarity. Without the floors, ordinary seed variance
is easily misread as a corruption effect.

**Elhage et al. (2022) — "Toy Models of Superposition"**
Monosemanticity vs polysemanticity across capacity. *Our delta*: the width
scaling of monosemanticity/circuit-size claims is from the v0 protocol and is
**pending re-derivation under v2** (see Scaling table above); it should be
presented as exploratory until then.

## Figures

| Figure | Description | Generator |
|--------|-------------|-----------|
| 1 | Four-phase framework diagram | `src/analysis/visualizations.py:figure1_framework()` |
| 2 | Rank-one edit pipeline diagram | `figure2_rome_pipeline()` |
| 3 | t-SNE manifold progression | `figure3_tsne_manifold()` |
| 4–6 | Neuron weights, correlation heatmap, circuit map | `figure4_neuron_weights()`, `figure5_correlation_heatmap()`, `figure6_circuit_map()` |
| 7 | Intervention recovery bar chart | `figure7_rome_barchart()` |
| 8 | Scaling analysis | `figure8_scaling_analysis()` |

All figures are regenerated from `outputs/` by
`python src/analysis/visualizations.py --results-dir outputs --output-dir outputs/figures --config configs/experiment_config.yaml`.
No manually typed experimental values in figures or tables.

## Reproducibility

| Item | Location |
|------|----------|
| Full pipeline | `reproduce_all.py` (loads `configs/experiment_config.yaml`; seeds/widths from config) |
| Configuration | `configs/experiment_config.yaml` |
| Seeds | config list (42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999) |
| Environment | `environment.yml`, `requirements.txt` |
| Tests | `python -m pytest tests/` (54 tests, incl. corruption/split invariants) |
| Consistency gate | `python scripts/verify_consistency.py` (hard-fail mode; doc↔artifact reconciliation) |
| Checkpoints + provenance | `outputs/{clean,corrupted,targeted_corrupted}/seed_*/` with `corruption_provenance.json`; v0 outputs archived in `outputs/v0/` |
| Authoritative results | `RESULTS_v2.md` + raw JSON in `outputs/analysis/` |
| Phase-5/LoRA | `reproduce_all.py --phase5` |
