# Preregistration: Memorization Dynamics and Causal Localization (v2.1)

**Status:** DRAFT — to be frozen before confirmatory 10-seed campaign  
**Repository:** `Four-Phase-Memorization-Analysis`  
**Date:** 2026-09-28  
**Git commit:** (to be filled at freeze)

---

## 1. Primary Hypotheses (Confirmatory)

### H1: Memorization Onset is Detectable at the Example Level
> **H1a:** In regimes where a non-trivial fraction of corrupted examples are memorized (M > 10%), individual corrupted examples exhibit a measurable transition epoch T_i where they begin persistently predicting the noisy label.
>
> **H1b:** This transition is preceded by a detectable increase in gradient conflict (cosine similarity between noisy-label and true-label gradients) and a divergence in per-example loss trajectories.

### H2: Internal Signals Improve Early-Warning Prediction Beyond Behavioral Baselines
> **H2:** A logistic regression model using internal signals (gradient conflict, representation drift, margin dynamics) at epoch t achieves significantly higher AUPRC for predicting "will be memorized by epoch t+Δ" than models using only behavioral baselines (CSL, current loss, forgetting count), evaluated on held-out seeds.

### H3: Memorization is Causally Mediated by Hidden-Layer Representations
> **H3a (Necessity):** For memorized examples, patching the corrupted model's hidden activation (post-ReLU) with the clean model's activation restores the true-label prediction at a rate significantly above chance.
>
> **H3b (Sufficiency):** Patching the clean model's hidden activation with the corrupted model's activation induces the noisy-label prediction at a rate significantly above chance.

### H4: Memorized Associations Occupy a Low-Dimensional Subspace
> **H4:** A rank-k weight edit (k ∈ {1,2,4,8}) constructed from EDIT_MEM generalizes to held-out EVAL_MEM with positive recovery and bounded collateral damage, outperforming random norm-matched null edits of the same rank.

---

## 2. Experimental Design

### 2.1 Stage A: Regime Sweep (Exploratory, Already Completed)
| Factor | Values |
|--------|--------|
| Hidden width | 16, 32, 64, 128, 256, 512 |
| Noise rate | 0%, 10%, 20%, 40%, 60% |
| Epochs | 20, 50, 100, 200 |
| Seeds per config | 3 (initialization × corruption × loader decoupled) |

**Decision rule:** Select 2–3 regimes where memorization fraction M ∈ [10%, 50%] for confirmatory study.

### 2.2 Stage B: Confirmatory Memorization Dynamics (Primary)
| Factor | Value |
|--------|-------|
| Architectures | MLP (h=64), MLP (h=256) |
| Noise rates | Selected from Stage A |
| Epochs | Selected from Stage A (sufficient to cross memorization transition) |
| Seeds | 10 (full canonical seed lists) |
| Optimizers | Adam (primary), SGD (robustness) |
| Gradient clipping | **Disabled** (to preserve gradient dynamics) |
| Checkpointing | Every epoch |
| Per-example logging | Every epoch: losses (true/noisy), margins, predictions, gradient conflict |

### 2.3 Stage C: Early-Warning Prediction (Primary)
| Component | Specification |
|-----------|---------------|
| Target | `Y_i(t, Δ) = 1[T_i > t ∧ T_i ≤ t+Δ]` (will memorize within Δ epochs) |
| Horizons Δ | 5, 10, 20 epochs |
| Baselines | Loss, CSL, Forgetting count, CSL+Loss, CSL+Forgetting |
| Internal features | Gradient conflict, representation drift (CKA), margin gap, activation norm |
| Model | Logistic regression (L2, C=1.0), standardized features |
| Evaluation | AUROC, AUPRC, held-out seeds (leave-one-seed-out) |
| Multiplicity | Holm correction across (horizon × feature-set) family |

### 2.4 Stage D: Causal Localization (Primary)
| Test | Method | Evaluation |
|------|--------|------------|
| Necessity | R→C patching (corrupt model + clean activation) | True-label recovery rate |
| Sufficiency | C→R patching (clean model + corrupt activation) | Noisy-label induction rate |
| Neuron ablation | Per-neuron causal effect on target margin | Effect size distribution |
| Subspace projection | Zero out top-k SVD components of D = h_R - h_C | Recovery vs. k |

**Example groups (matched):**
1. Memorized corrupted (M=1)
2. Non-memorized corrupted (M=0, changed)
3. Hard clean (matched on initial loss percentile)
4. Easy clean (matched on initial loss percentile)

### 2.5 Stage E: Low-Rank Editing (Primary)
| Component | Specification |
|-----------|---------------|
| EDIT set | 50% of memorized training examples per class (stratified) |
| EVAL set | Held-out 50% of memorized training examples |
| FINAL TEST | Official MNIST test set (10k, never used in development) |
| Ranks | 1, 2, 4, 8, 16, full |
| Regularization | Ridge λ = 1e-4 (fixed) |
| Bias | Included via augmented activations |
| Baselines | Random norm-matched (20 trials), LoRA (rank=4, 20 epochs), fine-tuning (SGD on EDIT) |
| Metrics | Recovery (EVAL_MEM target class), Collateral (FINAL_TEST other classes), Target preservation (FINAL_TEST target class) |

### 2.6 Stage F: Replication (Secondary)
| Architecture | Dataset |
|--------------|---------|
| Small CNN (2 conv + 1 FC) | CIFAR-10 (synthetic noise) |
| *Optional* | CIFAR-10N (natural noise) |

---

## 3. Statistical Analysis Plan

### 3.1 Confirmatory Comparisons
| Comparison | Test | Effect Size | Multiplicity |
|------------|------|-------------|--------------|
| H2: Internal > Behavioral (AUPRC) | Paired permutation test (seed-level) | Δ AUPRC with 95% bootstrap CI | Holm across horizons × feature sets |
| H3a: R→C recovery > 0 | One-sample t-test vs 0 | Cohen's d_z | None (single primary) |
| H3b: C→R induction > 0 | One-sample t-test vs 0 | Cohen's d_z | None (single primary) |
| H4: Rank-k recovery > Random null | Paired t-test vs random mean | Cohen's d_z | Holm across ranks |

### 3.2 Reporting Standards
- **Primary:** Effect size with 95% CI (Student-t over seeds) + exact p-value
- **Secondary:** Mean ± SEM, raw per-seed values in supplementary
- **No** "p < 0.0001" — report exact p to 3 decimal places
- **No** "∞ signal ratio" — report empirical p and effect size when null variance is zero
- All CI explicitly labelled "Student-t CI over 10 seeds"

### 3.3 Exclusion Criteria
- Seeds where training fails to converge (test_acc < 80% on clean)
- Seeds where corruption provenance is missing (v0 checkpoints)
- Examples where `true_label == noisy_label` (impossible by construction)

---

## 4. Computational Requirements

| Stage | Estimated GPU-hours |
|-------|---------------------|
| Stage A (sweep) | ~200 |
| Stage B (10 seeds × 2 archs × 2 noise × 2 opts) | ~160 |
| Stage C (prediction) | ~20 |
| Stage D (patching) | ~40 |
| Stage E (editing) | ~60 |
| Stage F (CNN) | ~80 |
| **Total** | **~560 GPU-hours** |

---

## 5. Data and Code Availability

- All code: `src/` (this repository)
- Config: `configs/experiment_config.yaml` (frozen at preregistration)
- Outputs: `outputs/` (versioned by git commit)
- Preregistration: `docs/PREREGISTRATION.md` (this document)
- Claim ledger: `docs/CLAIM_LEDGER.md`

---

## 6. Deviations from v2.0 Audit

| v2.0 Issue | v2.1 Fix |
|------------|----------|
| h=16 underfits noise (M ≈ 1%) | Regime sweep to find M > 10% |
| Single final checkpoint | Every-epoch checkpoints + per-example logs |
| Test-set EDIT/EVAL | Train-set EDIT_MEM / EVAL_MEM / FINAL_TEST firewall |
| "TracIn" misnomer | Renamed to `gradient_conflict` |
| Gradient clipping | Disabled for dynamics |
| Single optimizer | Adam + SGD |
| Correlated seeds | Decoupled init/corruption/loader seeds |
| CSL not implemented | CSL as primary baseline |
| No reverse patching | C→R sufficiency test added |
| One-hot ROME edit | Minimum-norm rank-k with bias |

---

## 7. Signatures (to be completed at freeze)

| Role | Name | Date | Git Commit |
|------|------|------|------------|
| PI | | | |
| Lead Experimenter | | | |
| Statistician | | | |

---

**To freeze this preregistration:**
1. Complete Stage A regime sweep
2. Select confirmatory regimes
3. Fill in git commit hash
4. All signatories sign
5. Commit to main branch with tag `v2.1-preregistered`