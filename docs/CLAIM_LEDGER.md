# Claim Ledger: Memorization Dynamics and Causal Localization

**Version:** v2.1 (aligned with PREREGISTRATION.md)  
**Last updated:** 2026-09-28  
**Status:** Living document — updated as evidence accumulates

---

## Claim Status Definitions

| Status | Meaning |
|--------|---------|
| 🟢 **Supported** | Confirmatory evidence meets preregistered criteria |
| 🟡 **Preliminary** | Exploratory evidence; directionally consistent but not yet confirmatory |
| 🔴 **Refuted** | Confirmatory evidence contradicts claim |
| ⚪ **Pending** | Awaiting experiment |
| 🟠 **Qualified** | Supported with important caveats |

---

## Core Claims

| # | Claim | Hypothesis | Status | Evidence | Notes |
|---|-------|------------|--------|----------|-------|
| C1 | Individual corrupted examples exhibit a measurable memorization onset epoch T_i | H1a | ⚪ Pending | — | Requires Stage B regime with M > 10% |
| C2 | Gradient conflict increases before memorization onset | H1b | ⚪ Pending | — | Stage B temporal dynamics |
| C3 | Internal signals (grad conflict, representation drift) improve early-warning prediction beyond CSL/forgetting | H2 | ⚪ Pending | — | Stage C prediction |
| C4 | Hidden-layer representations causally mediate memorized behavior (necessity) | H3a | ⚪ Pending | — | Stage D R→C patching |
| C5 | Hidden-layer representations are sufficient to induce memorized behavior | H3b | ⚪ Pending | — | Stage D C→R patching |
| C6 | Memorized associations occupy a low-dimensional editable subspace | H4 | ⚪ Pending | — | Stage E rank-k editing |

---

## Supporting Claims (from v2.0 Audit)

| # | Claim | Status | Evidence | Notes |
|---|-------|--------|----------|-------|
| S1 | h=16 model underfits 20% label noise (M ≈ 1.1%) | 🟢 Supported | RESULTS_v2.md | Directly measured with behavioral definition |
| S2 | Corruption reduces weight scale, not rank (stable rank unchanged p=0.32) | 🟢 Supported | RESULTS_v2.md | Direct spectral metrics |
| S3 | Behavioral memorization definition is non-circular | 🟢 Supported | Code audit | `compute_behavioral_memorization()` |
| S4 | EDIT/EVAL firewall prevents leakage in rank-one edits | 🟢 Supported | Code audit | `split_edit_eval()` provenance |
| S5 | fc1-only edits recover exactly 0pp on held-out EVAL | 🟢 Supported | RESULTS_v2.md | Multi-class ROME results |
| S6 | Random null is degenerate (0pp, zero variance) | 🟢 Supported | RESULTS_v2.md | 20 trials, empirical p = 1/21 |
| S7 | Cross-model CKA drift with seed floors replaces within-model CKA | 🟢 Supported | Code audit | `cross_model_cka()` + `cross_seed_cka_controls()` |
| S8 | Gradient conflict (cos = -0.33) is measurable at convergence | 🟢 Supported | RESULTS_v2.md | `compute_label_objective_gradient_conflict()` |

---

## Deprecated v0 Claims (Explicitly Withdrawn)

| # | v0 Claim | Withdrawn Because | Replacement (v2) |
|---|----------|-------------------|------------------|
| D1 | "Spectral norm drop ⇒ lower-rank memorization" | Direct rank metrics show stable rank unchanged | "Corruption reduces weight scale, not rank" (S2) |
| D2 | "Memorized fraction = 20% (matches noise rate)" | Corrupted ≠ memorized; behavioral definition gives 1.1% | Behavioral definition (S1) |
| D3 | "ROME signal ratio = ∞" | Random null has zero variance; z undefined | "Random null degenerate, reported honestly" (S6) |
| D4 | "Within-model CKA shows ReLU localization" | Seed variance confounds; cross-model with floors needed | Cross-model drift with floors (S7) |
| D4 | "Influence functions localize memorization" | Per-batch Hessian CG invalid; demoted to future work | Gradient conflict + behavioral definition |
| D5 | "TracIn cosine measures influence" | Misnamed; actually gradient conflict | Renamed to `compute_label_objective_gradient_conflict()` |
| D6 | "Rank-one ROME edit on test set" | EDIT/EVAL overlap (leakage) | EDIT_MEM/EVAL_MEM from train set |
| D7 | "Model called ROME" | Not Meng et al. 2022 ROME | "ROME-inspired closed-form rank-one edit" |
| D8 | "Gradient alignment = +0.99" | Mixed clean/corrupt batches | Gradient conflict = -0.33 (S8) |

---

## Claims Requiring Re-derivation (v0 Protocol)

| # | Claim | Status | Required Action |
|---|-------|--------|-----------------|
| R1 | Width scaling: FDR decreases monotonically (0.86→0.39) | ⚪ Pending | Stage B with new seeds |
| R2 | Monosemanticity decreases with width (0.27→0.08) | ⚪ Pending | Stage B with new seeds |
| R3 | Circuit size decreases with width (6.6→0.0) | ⚪ Pending | Stage B with new seeds |
| R4 | Sparsity → 1.0 for h > 128 | ⚪ Pending | Stage B with new seeds |
| R5 | CIFAR-10 3-layer MLP spectral norms increase under noise | ⚪ Pending | Stage F replication |
| R6 | CIFAR-10 CKA distortion at output layer (Δ=0.10) | ⚪ Pending | Stage F replication |
| R7 | LoRA achieves 100% recovery with catastrophic side effects | ⚪ Pending | Stage E with new protocol |

---

## Evidence Tracking

| Claim | Experiment | Artifact Location | Verification |
|-------|------------|-------------------|--------------|
| C1 | Stage B temporal | `outputs/temporal/*/seed_*_aggregated.npz` | `verify_consistency.py` |
| C2 | Stage B temporal | `outputs/temporal/*/seed_*_epoch_*.npz` | `verify_consistency.py` |
| C3 | Stage C prediction | `outputs/prediction/early_warning.json` | `scripts/verify_statistics.py` |
| C4 | Stage D patching | `outputs/causal/patching_results.json` | `verify_consistency.py` |
| C5 | Stage D patching | `outputs/causal/patching_results.json` | `verify_consistency.py` |
| C6 | Stage E editing | `outputs/editing/rank_k_results.json` | `verify_consistency.py` |
| S1–S8 | v2.0 audit | `RESULTS_v2.md`, `outputs/analysis/` | `verify_consistency.py` (hard-fail) |

---

## Decision Rules for Paper

| Claim | Inclusion Rule |
|-------|----------------|
| C1–C6 (primary) | Include iff 🟢 Supported at confirmatory level |
| S1–S8 (supporting) | Include as established background/methodology |
| R1–R7 (re-derivation) | Include only if 🟢 Supported in v2.1; otherwise cite as "v0 exploratory" |
| D1–D8 (deprecated) | Explicitly mention as "withdrawn after internal audit" in Limitations |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | 2026-09-XX | Post-audit baseline (this repo's current state) |
| v2.1 | 2026-09-28 | Added C1–C6 (preregistered), deprecated D1–D8, qualified R1–R7 |

---

## Usage

Update this ledger **immediately** after each experiment completes. Before paper submission, every claim in the manuscript must have a corresponding row here with status 🟢 or explicit justification for inclusion.