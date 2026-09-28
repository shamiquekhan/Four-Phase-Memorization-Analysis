# Methodology: Memorization in Neural Networks (v2)

## Overview

Four-phase empirical analysis of how fully connected neural networks memorize training data. We compare models trained on clean MNIST against models trained with 20% label noise, measuring differences in weight geometry, cross-model representation drift, behavioral memorization dynamics, and causal intervention recoverability.

## Model Architecture

All MNIST experiments use **MNISTNet** — a two-layer fully connected network:

```
Input (784) → Linear(784→16) → ReLU → Linear(16→10) → Output (10)
```

- Hidden dimension: 16 (default), scaling experiments test 32–1024
- Parameter count: 12,874
- Training: Adam, lr=0.001, batch=128, 20 epochs, cross-entropy loss

The small hidden dimension (16) forces the model into a compressed representation regime where individual neuron interpretability is feasible.

CIFAR-10 experiments use **CIFARNet** — a three-layer MLP:
```
Input (3072) → Linear(3072→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10) → Output (10)
```

## Training Regimes

### Regime A: Random Label Noise (Phases 1–3, Scaling, Noise Sweep)
- 20% of training labels are randomly flipped to a **different** class before training (guaranteed change)
- Additional noise rates tested for ROME sweep: 5%, 10%, 20%, 30%, 40%, 50%
- ~6,000–30,000 of 60,000 training samples corrupted per seed, depending on noise rate
- Studies *general memorization*: the model must memorize a random subset of incorrect labels while learning genuine features from the remaining fraction
- Used in Phases 1–3 (weight geometry, cross-model CKA, behavioral memorization), scaling analysis, and noise rate sweep
- **Ground-truth corruption indices saved as `corrupt_indices.npy`** for non-circular Phase 3 analysis
- **Provenance object (`CorruptionProvenance`)** saved per run with selected indices, original labels, new labels, and change guarantee

### Regime B: Targeted Class Swap (Phase 4 / Multi-class ROME)
- All samples of a specific source class are relabeled as a specific target class
- ~5,900–6,000 samples corrupted per class pair
- Four configurations tested: 7→1, 1→7, 5→6, 0→8
- Creates a controlled *backdoor* that rank-one interventions can target and attempt to repair
- Used exclusively in Phase 4 (multi-class ROME validation)
- **EDIT/EVAL split**: Stratified disjoint split of each class's corrupted samples (50/50), provenance saved

### Clean Training (Baseline)
Standard supervised learning on MNIST (60k train, 10k test). 10 random seeds for statistical rigor. All metrics are compared against this baseline to isolate memorization effects from normal learning dynamics.

## Four-Phase Analysis Pipeline (v2)

### Phase 1: Weight Geometry & Spectral Structure
- **Purpose**: Measure low-level differences in weight matrices and optimization dynamics
- **Metrics**:
  - Frobenius norm of weight matrices (`||W||_F`)
  - Spectral norm of weight matrices (`||W||_2`)
  - **Stable rank** (`||W||_F^2 / ||W||_2^2`)
  - **Effective rank** (entropy of normalized singular values)
  - **Spectral entropy** and cumulative energy curves
  - Gradient norm at the end of training
  - Fisher discriminant ratio (FDR = tr(S_B)/tr(S_W), dimension-invariant class separability)
  - Training/Test accuracy and loss
- **Method**: Load final checkpoint, compute norms via PyTorch SVD, aggregate with 95% CI
- **Output**: `outputs/analysis/phase1_{clean,corrupted}/phase1_results.json`

### Phase 2: Cross-Model Representational Drift
- **Purpose**: Quantify how representations differ between clean and corrupted models using CKA with seed noise-floor controls
- **Metrics**:
  - **Linear CKA similarity between clean↔corrupted models** on identical test inputs (cross-model drift = 1 − CKA)
  - **Cross-seed noise-floor controls**: clean↔clean pairwise CKA (seed variance floor), corrupted↔corrupted pairwise CKA
  - Within-model layer CKA (secondary, for comparison)
  - Activation sparsity (fraction of zero activations at ReLU output)
  - PCA explained variance of hidden representations
- **Method**: Extract activations on 5000 test samples, compute CKA via HSIC. Zero-centering applied before CKA computation.
- **Key v2 change**: Drift is cross-model (clean vs corrupted) with seed floors, not within-model layer similarity.
- **Output**: `outputs/analysis/phase2_corrupted/phase2_results.json`

### Phase 3: Memorization Dynamics (non-circular)
- **Purpose**: Identify which training samples are behaviorally memorized using ground-truth corruption provenance
- **Metrics**:
  - **Behavioral memorization**: changed AND fits noisy label AND ≠ original label (ground-truth, non-circular)
  - Loss gap on ground-truth corrupted indices
  - **Group gradient alignment** (clean vs changed examples, fc1 weights)
  - **TracIn-style per-example self-influence** (gradient self-inner-product at convergence)
  - Temporal trajectory of the above across epochs
- **Method**: Uses saved `corrupt_indices.npy` (provenance) to define memorization without circularity. TracIn replaces influence functions (per-batch Hessian, CG) which were unreliable at this scale.
- **Output**: `outputs/analysis/phase3_corrupted/phase3_results.json`

### Phase 4: Rank-One Interventions (ROME-inspired, EDIT/EVAL firewall)
- **Purpose**: Apply and measure causal interventions to localize memorized associations
- **Approach**:
  - **Closed-form class-mean rank-one edit** (NOT Meng et al. 2022 ROME: no causal tracing, no key/value covariance constraint)
  - Edit formula: `Δ = outer(v - W·u, u) / (u·u)` where `u` = mean hidden activation of target class on EDIT split, `v` = desired output
  - **EDIT/EVAL firewall**: Edit built on EDIT half, evaluated on **disjoint EVAL half** (provenance saved)
  - Measure delta norm (magnitude of edit), effect on target class (EVAL), and side effects on other classes
  - Apply to both fc1 and fc2 layers independently
- **Multi-class ROME**: Apply targeted edits to the four corruption configs and measure recovery of source-class accuracy on EVAL split, side effects on other classes, and edit magnitude
- **Noise rate sweep**: Delta-norm computed at 6 noise rates (5%, 10%, 20%, 30%, 40%, 50%) to verify monotonic scaling with memorization load
- **Random baseline**: Norm-matched random rank-one perturbations (20 trials per config)
- **Rank Ablation**: Replace weight matrices with rank-k SVD approximations (k=1..10) and measure accuracy degradation at each rank
- **Output**: `outputs/analysis/phase4_noise_{rate}/`, `outputs/analysis/multiclass_rome/`

## Scaling Analysis (v2 protocol)

- Hidden dimensions tested: [16, 32, 64, 128, 256, 512, 1024]
- **Seeds**: First 5 seeds from config list (v2 protocol; 10 seeds used for Phases 1–4)
- Metrics tracked vs hidden dimension:
  - Fisher discriminant ratio (FDR = tr(S_B)/tr(S_W), dimension-invariant class separability)
  - Monosemanticity fraction (fraction of neurons with max correlation > threshold)
  - Circuit size (neurons needed per class decision, via critical neuron ablation)
  - Network sparsity (fraction of near-zero weights)
  - Test accuracy
  - CKA stability (clean pre→post ReLU across widths)
- Models trained for 100 epochs at each dimension

## Statistical Methodology (v2)

- **Seeds**: Config list (42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999) — single source of truth
- **95% confidence intervals**: Student's t-distribution: `mean ± t_{0.975, n-1} * σ / sqrt(n)`
- **Paired t-tests** for clean-vs-corrupted comparisons
- **Holm correction** for multiple comparisons
- **Effect sizes**: Paired Cohen's d_z
- **Permutation tests** for random-null baselines (empirical p-values)
- All metrics reported as `mean [CI_low, CI_high]` across seeds
- **Minimum n=5 seeds** for scaling analysis; n=10 for Phases 1–4

## Reproducibility

Full pipeline is orchestrated by `reproduce_all.py`:

```bash
# From scratch (training + analysis + figures)
python reproduce_all.py

# Using existing checkpoints (analysis + figures only)
python reproduce_all.py --skip-training

# With CIFAR-10 validation
python reproduce_all.py --skip-training --skip-cifar10

# Phase 5 (LoRA vs rank-one subspace comparison)
python reproduce_all.py --phase5
```

Configuration: `configs/experiment_config.yaml` (single source of truth for all hyperparameters, seeds, widths).

## Verification Gates

```bash
# Hard-fail consistency check (docs ↔ artifacts)
python scripts/verify_consistency.py

# CI and seed-count verification
python scripts/verify_statistics.py

# Scientific invariant tests (54 tests)
python -m pytest tests/ -v
```