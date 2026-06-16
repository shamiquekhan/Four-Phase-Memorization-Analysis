# Changelog

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
