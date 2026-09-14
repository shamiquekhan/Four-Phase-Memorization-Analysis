> **Note on version history:** The initial codebase was developed locally and uploaded in a single commit. All subsequent changes are committed incrementally. See [CHANGELOG.md](CHANGELOG.md) for a narrative of what changed and when.

# Structural Fingerprints of Label Memorization in Shallow Neural Networks

> ## ⚠️ v2 methodology revision in progress — v0 results below are EXPLORATORY
>
> An internal audit found methodological defects in the v0 pipeline. The fixes
> below are implemented in code; **all v0 headline numbers must be re-derived
> from retrained checkpoints before being trusted**:
>
> | Defect (v0) | Status | Fix |
> |---|---|---|
> | Label corruption could re-assign the ORIGINAL label (~10% of "corrupted" samples were silently unchanged) | ✅ fixed | `src/data/corruption.py`: selected == changed, guaranteed; provenance saved |
> | "Corrupted sample" equated with "memorized sample" | ✅ fixed | Behavioral definition: changed AND fits noisy label AND ≠ original |
> | Rank-one edit built AND evaluated on the same test set (leakage) | ✅ fixed | `src/data/splits.py`: stratified disjoint EDIT/EVAL split, provenance saved |
> | Method called "ROME" but is not Meng et al. (2022) ROME | ✅ renamed | Now "ROME-inspired closed-form rank-one edit" everywhere |
> | CKA measured within-model layer similarity, not clean↔corrupted drift | ✅ fixed | `cross_model_cka()` + cross-seed noise-floor controls |
> | Influence functions used per-batch Hessian, no damping; "CG sensitivity" flag computed nothing | ✅ replaced | Demoted to future work; TracIn-style + behavioral memorization instead |
> | Seeds via `range(n)` in runners vs config list | ✅ fixed | Config `seeds:` is the single source of truth everywhere |
> | "Spectral norm down ⇒ lower rank" inference | ✅ fixed | Direct stable rank / effective rank / entropy metrics added |
> | `∞` signal ratios; uncorrected multiple tests | ✅ fixed | z vs empirical null, Holm correction, paired d_z effect sizes |
> | CI wording (paper said bootstrap, code computed Student-t) | ✅ fixed | Paper now says Student-t; bootstrap_ci available as robustness check |
>
> Existing checkpoints are **v0** (no corruption provenance); the v2 pipeline
> detects and flags them as degraded. Retraining campaign is the next step.

A systematic 4-phase analysis of how label memorization leaves structural fingerprints in shallow ReLU networks — spanning cross-model CKA representation drift, spectral geometry, selectivity, behavioral memorization dynamics, and ROME-inspired closed-form rank-one interventions. Primary experiments on MNIST (784→16→10), validated on CIFAR-10 (3-layer MLP), with width scaling.

> **Terminology (v2):** "rank-one edit" below refers to a ROME-INSPIRED
> closed-form class-mean rank-one update, NOT the original ROME algorithm of
> Meng et al. (2022) (no causal tracing, no key/value covariance constraint).
> All v0 numbers labeled ROME below are pending v2 re-derivation.

## Key Results — v2 (re-derived, 10 seeds, 95% Student-t CI)

All numbers below are from the v2 campaign: 90 models retrained with
provenance, EDIT/EVAL firewall enforced, behavioral memorization definition.
v0 numbers are archived in `outputs/v0/` and superseded.

| Metric | Clean | Corrupted (20% noise) | p (paired) |
|--------|:-----:|:---------------------:|:----------:|
| **Train Accuracy** | 96.41% | 75.45% (noisy labels) | 2e-17 |
| **Test Accuracy** | 95.32% | 93.52% | 6e-06 |
| **FC1 Spectral Norm** | 4.43 | 3.71 | 4e-06 |
| **FC2 Spectral Norm** | 2.56 | 1.35 | 3e-09 |
| **FC2 Stable Rank** | 3.82 | 3.97 | 0.32 (n.s.) |
| **FC2 Effective Rank** | 8.95 | 9.16 | 3e-03 |
| **FDR (h=16)** | 0.84 | 1.50 | 3e-06 |
| **Cross-model CKA drift (output)** | — | 0.496 | — |
| **Rank-one delta-norm (fc2, EDIT split)** | 0.801 | 0.371 | 1e-06 |

**The honest v2 story (differs materially from v0):**

1. **The h=16 model does NOT memorize 20% label noise.** Behavioral
   memorization (changed AND fits noisy label AND ≠ original) is only
   **1.1% of changed examples** (0.22% of all). 92.8% of corrupted examples
   still fit their ORIGINAL label — this is underfitting, not memorization.
   v0's "Memorized Fraction = 0.200" was an artifact of equating
   corrupted == memorized.
2. **Corruption lowers weight scale, not rank.** fc2 spectral norm drops
   47% (p=3e-9) but stable rank is unchanged (p=0.32) and effective rank
   slightly rises. Confirms the audit's warning: spectral-norm drop ≠
   lower-rank memorization.
3. **Separability increases under corruption at h=16** (FDR 0.84 → 1.50).
4. **Cross-model drift is layer-graded** (output 0.50 > fc1_post 0.36 >
   fc1_pre 0.23) and exceeds seed noise floors at fc1_post; but output
   drift ≈ the corrupted-condition seed floor (0.48), i.e. noise mainly
   inflates readout variance across seeds.
5. **Rank-one delta-norm dose-response is real but smaller than v0:**
   ratios 1.78× (5% noise) → 2.66× (50%), monotone, all p<1e-5, n=10 paired
   seeds (v0 claimed 3.4×–6.0×).
6. **Rank-one intervention on untouched EVAL data:** recovery +10.9 to
   +16.9pp (all p<0.02); **fc1-only edits recover exactly 0.0pp** in every
   config; sequential fc2→fc1 (3.5–8.1pp) is worse than fc2-only.
7. **Random-null is degenerate** (all 20 norm-matched nulls recover 0.0pp,
   zero variance): z undefined, empirical p at the permutation floor 1/21.
   Reported honestly; no "signal ratio = ∞".
8. **Per-example gradient anti-alignment is measurable:** TracIn
   cos(noisy-grad, orig-grad) = −0.33 ± 0.09; 76% of changed examples
   anti-aligned. Group-level fc1 gradient alignment at convergence:
   −0.88 (v0 wrongly measured +0.99 with clean/corrupt batches mixed).

See [RESULTS.md](RESULTS.md) for full tables.

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── model.py                  # MNISTNet + CIFAR10MLP
│   │   └── cifarnet.py              # CIFARNet (3072→256→128→10)
│   ├── utils/
│   │   ├── metrics.py               # CKA, FDR, monosemanticity, ROME utilities
│   │   └── stats.py                 # CI, paired t-test, multi-seed runner
│   ├── training/
│   │   ├── train_clean.py           # MNIST clean
│   │   ├── train_corrupted.py       # MNIST corrupted (saves corrupt_indices.npy)
│   │   ├── train_targeted_corrupted.py
│   │   ├── train_cifar.py           # CIFARNet combined clean/corrupted trainer
│   │   ├── train_cifar10_clean.py
│   │   ├── train_cifar10_corrupted.py
│   │   └── train_cifar10_scaling.py
│   ├── analysis/
│   │   ├── phase1_basic.py          # Weight norms, gradients, FDR
│   │   ├── phase2_representation.py # CKA, PCA, activation statistics
│   │   ├── phase3_influence.py      # Memorization metrics, gradient alignment
│   │   ├── phase4_rome.py           # ROME (single-layer)
│   │   ├── multiclass_rome.py       # Multi-class ROME + random baseline
│   │   ├── multilayer_rome.py       # Sequential + joint multi-layer ROME
│   │   ├── rank_ablation.py         # SVD rank ablation
│   │   ├── cifar_replication.py     # CIFAR-10 CKA + ROME + rank ablation rep.
│   │   ├── analyze_cifar10.py
│   │   ├── rome_cifar10.py
│   │   ├── analyze_cifar10_scaling.py
│   │   └── visualizations.py        # All publication figures
│   └── scaling/
│       ├── train_scaling.py
│       └── analyze_scaling.py        # FDR, monosemanticity, sparsity vs width
├── docs/
│   └── theoretical_propositions.md   # 3 formal propositions
├── paper/
│   ├── main.tex                      # LaTeX paper
│   └── related_work.bib
├── scripts/
│   ├── verify_consistency.py         # Cross-document number verification
│   └── verify_statistics.py          # CI and seed count verification
├── configs/experiment_config.yaml
├── tests/test_metrics.py            # 24 unit tests
├── tests/test_invariants.py         # 30 scientific-invariant tests (v2)
├── reproduce_all.py                 # Single-command pipeline
├── RESULTS.md                       # Verified result tables
├── METHODOLOGY.md                   # 4-phase methodology
├── PAPER.md                         # Paper-to-code mapping
├── README.md
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate memorization-analysis

# Run full MNIST pipeline (using existing checkpoints)
python reproduce_all.py --skip-training

# Run with CIFAR-10 validation
python reproduce_all.py --skip-training --skip-cifar10

# From scratch (training + analysis + figures)
python reproduce_all.py

# Run tests
python -m pytest tests/ -v          # 54/54 pass
```

## Pipeline

1. **Phase 1: Weight Geometry** — Spectral norms, Frobenius norms, gradient norms, FDR
2. **Phase 2: Representation Analysis** — CKA similarity between layers, activation statistics
3. **Phase 3: Influence Functions** — Non-circular memorization scoring, gradient alignment
4. **Phase 4: ROME Analysis** — Rank-One Model Editing, multi-class validation, random baseline, multi-layer ROME

## Configuration

All hyperparameters in `configs/experiment_config.yaml` — model dimensions, training epochs, learning rates, seed lists, and CIFAR-10 config block.

## GitHub Topics

When publishing this repo on GitHub, add these topics in the repo settings (Settings → Topics):

`neural-networks` `memorization` `interpretability` `cka` `rome` `mnist` `representation-similarity` `model-editing`

## Citation

```bibtex
@software{memorization_analysis_2026,
  title = {Structural Fingerprints of Label Memorization in Shallow Neural Networks},
  year = {2026},
  url = {https://github.com/shamiquekhan/four-phase-memorization-analysis}
}
```
