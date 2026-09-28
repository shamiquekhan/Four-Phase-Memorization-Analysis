# Changelog

## v2.0.0 — Methodology revision (current)
- **Corruption provenance**: `selected == changed` guaranteed; `CorruptionProvenance` saved per run (`src/data/corruption.py`)
- **Behavioral memorization definition**: changed AND fits noisy label AND ≠ original (replaces corrupted == memorized)
- **EDIT/EVAL firewall**: Stratified disjoint split (`SplitProvenance` saved) — edits built on EDIT, evaluated on held-out EVAL
- **Terminology**: "ROME-inspired closed-form rank-one edit" everywhere (not Meng et al. 2022 ROME)
- **Cross-model CKA drift** + cross-seed noise-floor controls replace within-model layer CKA
- **Influence functions demoted**: Per-batch Hessian + CG replaced by TracIn-style gradient self-influence + behavioral definition
- **Seeds**: Config list is single source of truth everywhere (no more `range(n)`)
- **Rank metrics**: Stable rank, effective rank, spectral entropy added; "spectral norm down ⇒ lower rank" inference withdrawn
- **Statistics**: z vs empirical null, Holm correction, paired d_z effect sizes; no ∞ signal ratios
- **CI wording**: Paper says Student-t (code was already Student-t); bootstrap available as robustness
- **Hardened verification**: `verify_consistency.py` hard-fails on missing artifacts; recomputes from raw JSON
- **Pipeline**: `reproduce_all.py` loads seeds/widths from config; conditional analysis steps

## v1.1.0 — Reviewer round 2 fixes
- AUC 0.514 framing: reframed as "existence evidence" rather than "above-chance" detection
- Regime A/B gap: acknowledged that Phases 1-3 (random noise) differ from Phase 4 (targeted swaps)
- ROME side effects: mechanistically explained via circuit size (6.6) and monosemanticity (26.9%)
- Scaling/ROME tension: caveated that pipeline is diagnostic for compressed architectures; ROME expected to degrade at wider widths
- Training accuracy anomaly: explained as 16-unit bottleneck limiting capacity to 94.21% (not 100%)
- Future directions: added width-extended interventions as direction (ii)
- CHANGELOG: created this file for transparent version history

## v1.0.1 — Reviewer round 1 fixes
- DOCX: removed empty rows from Table 3, replaced pipeline text diagram with Figure 1 image
- Multi-layer ROME narrative corrected (sequential editing yields 5-18%, not 22-40% improvement)
- Added width-scaling CKA analysis
- Fixed README BibTeX URL

## v1.0.0 — Initial upload
- Full four-phase pipeline: weight geometry, CKA, influence scoring, ROME
- Ten-seed reproducibility with 95% CI throughout
- Width-scaling analysis (16-1024 units)
- CIFAR-10 cross-validation