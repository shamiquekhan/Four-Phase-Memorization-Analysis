# v2 Results (authoritative)

Derived from the v2 campaign: 90 models retrained from the config seed list
(42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999), corruption
provenance saved per run (selected == changed guaranteed), all edits built
from the EDIT half of a fixed stratified test split and evaluated only on
the disjoint EVAL half. Raw JSON: `outputs/analysis/`.

CI: Student-t, 95%. p-values: paired t-tests across 10 seeds unless noted.
These tables supersede the v0 tables in RESULTS.md.

---

## Training performance (MNIST 784-16-10, 20 epochs, Adam lr=1e-3)

| Metric | Clean (10 seeds) | Corrupted 20% (10 seeds) | p |
|--------|:-:|:-:|:-:|
| Train Acc (noisy labels) | 96.41% | 75.45% | 2e-17 |
| Test Acc | 95.32% | 93.52% | 6e-06 |

Under the behavioral definition (changed AND fits noisy AND ≠ original):

| Behavioral memorization (corrupted) | Rate |
|---|---:|
| Memorized fraction of changed examples | **1.10% ± 0.06** |
| Memorized fraction of all examples | 0.22% |
| Changed examples fitting noisy label | 1.10% |
| Changed examples fitting ORIGINAL label | **92.76% ± 0.25** |
| Clean-example train accuracy | 94.22% |

**Interpretation:** at this scale/regime the model does not memorize the
noise — it underfits it. Any memorization-structure claims must be
conditioned on a regime where noisy-label fit actually occurs.

---

## Phase 1: weight geometry + spectral structure

| Metric | Clean | Corrupted 20% | Δ | p (paired) |
|--------|:-:|:-:|:-:|:-:|
| FC1 spectral norm | 4.429 | 3.711 | −0.72 | 4e-06 |
| FC2 spectral norm | 2.555 | 1.348 | −1.21 | 3e-09 |
| FC2 stable rank | 3.823 | 3.969 | +0.15 | **0.32 (n.s.)** |
| FC2 effective rank | 8.950 | 9.158 | +0.21 | 3e-03 |
| FC2 spectral entropy | 0.952 | 0.962 | +0.01 | 3e-03 |
| FC2 top-1 cumulative energy | 0.263 | 0.254 | −0.01 | 0.34 (n.s.) |
| FDR (h=16) | 0.837 | 1.498 | +0.66 | 3e-06 |

**Interpretation:** corruption reduces operator scale (spectral norms) while
leaving rank structure intact — stable rank unchanged, effective rank
marginally up. The v0 claim "corruption lowers effective rank" is NOT
supported. Separability (FDR) increases at fixed width h=16.

---

## Phase 2: cross-model representational drift (v2 primary CKA analysis)

Drift = 1 − CKA(clean_layer, corrupted_layer) on identical test inputs.
Seed noise floors are pairwise CKA drifts between two same-condition models.

| Layer | Clean↔Corrupted drift | Seed floor (clean↔clean) | Seed floor (corr↔corr) |
|-------|:-:|:-:|:-:|
| input | 0.000 | 0.000 | 0.000 |
| fc1_pre | 0.233 | 0.228 | 0.228 |
| fc1_post | **0.360** | 0.294 | 0.261 |
| output | **0.496** | 0.253 | **0.483** |

**Interpretation:** drift is layer-graded and exceeds the seed noise floor
at fc1_post (0.360 vs 0.29) — the clearest cross-model signature. Output
drift (0.496) is comparable to the corrupted-condition seed floor (0.483):
label noise mainly inflates seed-to-seed readout variance rather than
producing a stable output-geometry change.

---

## Phase 3: memorization dynamics (v2, non-circular)

| Signal (corrupted, 10 seeds) | Value |
|---|---:|
| Loss gap (ground-truth corrupted indices) | −3.237 ± 0.039 |
| Group gradient alignment (clean vs changed, fc1) | **−0.883 ± 0.034** |
| TracIn cos(noisy-label grad, original-label grad) per example | **−0.335 ± 0.093** |
| Fraction of changed examples anti-aligned | **75.7%** |

**Interpretation:** per-example and group gradient anti-alignment between
noisy and original objectives is measurable and strong. v0's +0.9944
"alignment" was an artifact of measuring at convergence with mixed batches.

---

## Phase 4: rank-one delta-norm dose-response (EDIT split, 10 paired seeds)

fc2 closed-form rank-one edit norm, clean vs corrupted at each noise rate:

| Noise rate | Corrupted delta-norm | Clean/corrupted ratio | p (paired) |
|:-:|:-:|:-:|:-:|
| 5% | 0.451 | 1.78× | 3e-06 |
| 10% | 0.416 | 1.93× | 3e-06 |
| 20% | 0.371 | 2.16× | 1e-06 |
| 30% | 0.333 | 2.41× | 1e-06 |
| 40% | 0.312 | 2.57× | 8e-07 |
| 50% | 0.301 | 2.66× | 1e-06 |

(Clean reference: 0.801.) Monotone dose-response confirmed; magnitudes are
smaller than v0's 3.4×–6.0× claims. This statistic is a dose-sensitive
descriptor of the corrupted training signal, not a memorization detector.

---

## Multi-class rank-one intervention (EDIT/EVAL firewall, 5 seeds)

Edit directions built from EDIT half only; recovery measured on untouched
EVAL half only. Targeted corruption configs:

| Config | fc2-only recovery | fc1-only recovery | fc2→fc1 both | Side effects (fc2) | p (recovery ≠ 0) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 7→1 | +12.5pp | 0.0pp | +5.0pp | 19.2pp | 6e-04 |
| 1→7 | +15.1pp | 0.0pp | +5.1pp | 17.3pp | 1e-02 |
| 5→6 | +10.9pp | 0.0pp | +3.5pp | 23.7pp | 2e-03 |
| 0→8 | +16.9pp | 0.0pp | +8.1pp | 15.5pp | 2e-03 |

Random-null control: all 20 norm-matched random rank-one edits per config
recover 0.0pp on the EVAL split (zero-variance degenerate null); empirical
permutation p at floor 1/21 ≈ 0.048; z undefined. Reported without
infinite-ratio claims.

**Interpretation:** interventions act only at the output layer (fc1-only =
exactly 0 recovery in all configs); recovery on genuinely held-out data is
+11 to +17pp with substantial side effects (15–24pp mean absolute
per-class accuracy change).

---

## What changed vs v0 (audit trail)

| v0 claim | v2 finding | Why it changed |
|---|---|---|
| Memorized fraction = 0.200 | 0.011 (of changed) | corrupted ≠ memorized; behavioral definition |
| Corrupted train acc 94.2% | 75.5% (noisy-label) | v0 accuracy read from mixed/noisy labels; ~1/9 of corrupted got original label back via randint bug inflating fit |
| Lower effective rank under corruption | stable rank n.s., eff. rank +0.2 | direct spectral metrics instead of norm inference |
| ROME ratios 3.37×–5.98× | 1.78×–2.66× | EDIT/EVAL firewall + guaranteed-change corruption |
| Gradient alignment +0.994 | group −0.883, per-example −0.335 | separated clean/changed example sets; no mixed batches |
| Random baseline "signal ratio ∞" | degenerate null (0.0pp, 0 variance), p=1/21 floor | honest reporting of zero-variance nulls |
| "Distortion localizes at ReLU" (within-model CKA) | cross-model drift largest at fc1_post beyond seed floor; output drift ≈ corr seed floor | proper cross-model CKA + controls |

## Remaining known limitations

1. Memorization regime: at 20 epochs / h=16, noisy-label memorization is
   ~1%; memorization-structure claims need a regime where it's substantial
   (more epochs, wider models, or both — planned v2.1).
2. Temporal (per-epoch) dynamics not yet instrumented — anti-alignment
   trajectory across training is the next experiment.
3. CIFAR-10 replication not yet rerun under v2 protocol.
4. Scaling sweep not yet rerun under v2 protocol.
5. Phase 5 LoRA comparison not yet rerun with matched edit sets (code ready).
