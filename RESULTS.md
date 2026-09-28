# Results: Memorization in Neural Networks

> ## ⚠️ STATUS: tables below are v0 EXPLORATORY
>
> The v2 campaign retrained 90 models with provenance and the EDIT/EVAL firewall and re-derived all headline numbers.
> **The v2 numbers differ materially** — see README "Key Results — v2" and `outputs/analysis/` for the authoritative results.
> **Authoritative source: [RESULTS_v2.md](RESULTS_v2.md)**
>
> Key v2 corrections:
> - Behavioral memorization is **1.1% of changed examples** (not 20%): 92.8% of corrupted examples still fit their original label.
> - fc2 stable rank is **unchanged** under corruption (p=0.32); corruption lowers weight scale, not rank.
> - Rank-one delta-norm ratios: **1.78×–2.66×** (not 3.4×–6.0×).
> - Group gradient alignment at convergence is **−0.88** (not +0.99).
> - Cross-model CKA drift with seed floors replaces within-model layer CKA.

All values reported as `mean [95% CI]` over seeds unless otherwise noted.
CI computed via Student's t-distribution: `mean ± t_{0.975, n-1} * SEM`. Paired t-tests used for clean-vs-corrupted comparisons.

---

## v0 Training Performance (EXPLORATORY)

| Metric | Clean (10 seeds) | Corrupted 20% (10 seeds) |
|--------|:-:|:-:|
| Train Accuracy | **96.68%** [96.60%, 96.77%] | **94.21%** [94.02%, 94.40%] |
| Test Accuracy | **95.31%** [95.16%, 95.46%] | **93.71%** [93.42%, 94.00%] |
| Final Train Loss | 0.124 | 1.015 |
| Final Test Loss | 0.165 | 0.410 |

---

## v0 Phase 1: Weight Geometry (EXPLORATORY)

### MNIST (2-layer MLP, 10 seeds)

| Metric | Clean | Corrupted |
|--------|:-:|:-:|
| FC1 Spectral Norm | 4.37 [4.24, 4.50] | **3.66** [3.53, 3.79] |
| FC2 Spectral Norm | 2.58 [2.42, 2.75] | **1.39** [1.30, 1.47] |
| FC1 Frobenius Norm | 12.56 [12.38, 12.74] | **10.33** [10.19, 10.47] |
| FC2 Frobenius Norm | 4.99 [4.83, 5.16] | **2.79** [2.70, 2.87] |
| Gradient Norm (final) | 0.94 [0.89, 0.98] | 1.00 [0.96, 1.05] |

---

## v0 Phase 2: Representation Similarity — Within-Model CKA (EXPLORATORY)

### MNIST (2-layer MLP, 10 seeds)

| Layer Pair | Clean | Corrupted | Δ (Clean−Corrupted) | p-value |
|------------|:-:|:-:|:-:|:-:|
| input→fc1_pre | 0.712 [0.697, 0.727] | 0.693 [0.676, 0.710] | +0.019 | 0.08 |
| **fc1_pre→fc1_post** | **0.850 [0.828, 0.873]** | **0.690 [0.669, 0.712]** | **+0.160** | **<0.001** |
| fc1_post→output | 0.706 [0.666, 0.745] | 0.715 [0.685, 0.745] | −0.009 | 0.65 |

**v0 Finding (superseded by v2 cross-model drift):** The ReLU nonlinearity (fc1_pre→fc1_post) shows a dramatic CKA drop under corruption.

---

## v0 Phase 3: Influence & Memorization (EXPLORATORY)

### Clean Model (loss-quantile definition, 10 seeds)

| Metric | Value |
|--------|:-:|
| Mean Loss | 0.112 [0.108, 0.117] |
| Accuracy | 96.68% [96.58%, 96.79%] |
| Memorized Fraction (top 25% loss) | 0.250 (by construction) |
| Loss Gap (memorized − forgotten) | **+0.442** [+0.426, +0.458] |

### Corrupted Model (ground-truth definition, 10 seeds)

| Metric | Value |
|--------|:-:|
| Mean Loss | 0.408 [0.396, 0.420] |
| Accuracy | 94.21% [93.98%, 94.45%] |
| Memorized Fraction (exact 20% corrupted) | 0.200 (matches noise rate) |
| Loss Gap (memorized − forgotten) | **−0.051** [−0.057, −0.045] |

---

## v0 Phase 4: ROME (Rank-One Model Editing) (EXPLORATORY)

### Single-Layer ROME — MNIST (fc2, all 10 classes, 10 seeds)

| Class | Clean Δ-norm [95% CI] | Corrupted Δ-norm [95% CI] | Ratio | p-value |
|------:|:-:|:-:|:-:|:-:|
| 0 | 22.46 [21.02, 23.90] | 4.54 [3.69, 5.40] | 4.94× | <0.0001 |
| 1 | 18.11 [16.97, 19.26] | 4.40 [4.02, 4.78] | 4.12× | <0.0001 |
| 2 | 21.01 [19.58, 22.45] | 4.23 [3.98, 4.48] | 4.97× | <0.0001 |
| ... | ... | ... | ... | ... |

**MNIST fc2 average**: Clean=19.08, Corrupted=4.40, Ratio=**4.34×**

### Noise Rate Sweep (v0)

| Noise Rate | FC2 Δ-norm | Ratio vs Clean (19.08) |
|:----------:|:----------:|:----------------------:|
| 0% (Clean) | 19.08 | 1.00× |
| 10% | ~5.67 | 3.37× |
| 20% | ~4.40 | 4.34× |
| 40% | ~3.19 | 5.98× |

### Baseline Comparison (v0)

| Method | Discriminability |
|--------|:----------------:|
| ROME delta-norm fc2 ratio | **4.34×** |
| Spectral norm fc2 ratio | 1.86× |
| Linear probe AUC | 0.514 |

### Random Baseline for ROME (v0)

| Config | ROME Recovery (mean±std) | Random Baseline (mean±std) | Signal Ratio |
|--------|:-:|:-:|:-:|
| 7→1 | +14.12% ± 8.28pp | 0.00% ± 0.00pp | ∞ |
| 1→7 | +21.74% ± 6.90pp | 0.00% ± 0.00pp | ∞ |
| 5→6 | +12.04% ± 3.38pp | 0.00% ± 0.00pp | ∞ |
| 0→8 | +9.71% ± 6.04pp | 0.00% ± 0.00pp | ∞ |

---

## v0 Scaling Analysis (EXPLORATORY)

| Hidden Dim | FDR | σ (legacy) | Monosemanticity | Circuit Size | Sparsity | Accuracy |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 16 | 0.862 | 0.687 | 0.269 | 6.6 | 0.588 | 95.31% |
| 32 | 0.654 | 0.742 | 0.222 | 8.3 | 0.740 | 96.85% |
| 64 | 0.560 | 0.771 | 0.212 | 3.5 | 0.945 | 97.50% |
| 128 | 0.470 | 0.797 | 0.184 | 0.4 | 0.997 | 97.85% |
| 256 | 0.456 | 0.801 | 0.175 | 0.1 | 1.000 | 97.93% |
| 512 | 0.423 | 0.812 | 0.154 | 0.0 | 1.000 | 98.17% |
| 1024 | 0.387 | 0.823 | 0.075 | 0.0 | 1.000 | 98.11% |

---

> **For authoritative v2 results, see [RESULTS_v2.md](RESULTS_v2.md) and the JSON artifacts in `outputs/analysis/`.**