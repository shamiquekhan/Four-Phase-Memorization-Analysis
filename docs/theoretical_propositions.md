# Theoretical Account: Three Propositions (v2)

> **Note:** Empirical support values below reference the **v0 protocol** and are **exploratory**.
> v2 re-derivation is pending for Propositions 1 and 2. Proposition 3's delta-norm
> dose-response (monotone 1.78×–2.66×) is confirmed in v2; the rank-one intervention
> results (+10.9 to +16.9pp recovery on EVAL split) are v2-validated.

## Proposition 1: Label Noise Localizes Distortion at the Deepest Pre-Output Interface

**Informal statement:** In a ReLU network trained with cross-entropy loss, random label noise forces the activation pattern at the deepest nonlinearity before the output to absorb class boundary inconsistencies that cannot be resolved by the surrounding linear transformations. The depth of this interface scales with network depth.

**Argument sketch:** Linear layers can only rotate and scale the representation — they cannot change the topology of the decision boundary. Each ReLU nonlinearity partitions the input space into polyhedral regions and determines which neurons activate. When labels are randomly flipped, the model must accommodate inputs from the same polyhedral region being assigned to different classes. The deepest pre-output interface bears the brunt of this disambiguation because it is the final stage before the output linear layer's decision boundary. In a two-layer network this interface is the single ReLU; in a deeper network it shifts to later nonlinearities (and potentially the output linear layer itself) as earlier features remain relatively stable.

**v0 Empirical support (pending v2 re-derivation):** Phase 2 on MNIST (2-layer) showed Δ=+0.160 at fc1_pre→fc1_post (the single nonlinearity) vs near-zero Δ at linear pairs. On CIFAR-10 (3-layer), the largest distortion shifted to the deepest pre-output interface (fc2_post→output, Δ=+0.1027, p=0.0094), while the first nonlinearity showed a much smaller Δ=+0.0183. This cross-architecture pattern — distortion concentrating at the deepest pre-output bottleneck — is predicted by the proposition.

**v2 revision:** The primary analysis is now **cross-model CKA drift with seed noise floors**, not within-model layer similarity. The v2 finding is that drift is layer-graded (output 0.496 > fc1_post 0.360 > fc1_pre 0.233) with fc1_post exceeding the clean-seed floor (0.294), but output drift ≈ corrupted-seed floor (0.48), implying label noise mainly inflates readout variance across seeds. The "localization at ReLU" claim is superseded by this more nuanced cross-model picture.

---

## Proposition 2: Memorization Affects Spectral Geometry in a Depth-Dependent Manner

**Informal statement:** Label corruption reshapes the singular value spectrum of weight matrices — reducing effective rank in shallow networks — but the direction of spectral norm change depends on architectural depth, with deeper layers in multi-layer networks exhibiting larger spectral norms to absorb additional separability burden.

**Argument sketch:** In a shallow network, corrupted labels force the entire weight matrix to distribute its capacity more uniformly across singular directions (flattening the spectrum). In a deeper network, intermediate and output layers must also absorb additional class-boundary complexity, which can manifest as increased spectral norms — the network dedicates more representational capacity at the output to resolve the noisy label assignments. The effective rank (number of significant singular directions) may still be reduced even if the leading singular value increases, because the trailing singular values drop off more steeply. Low-rank ablation at intermediate ranks thus tests effective rank regardless of spectral norm direction.

**v0 Empirical support (pending v2 re-derivation):** On MNIST (2-layer), spectral norms decreased under corruption (FC1: 4.37→3.66; FC2: 2.58→1.39). On CIFAR-10 (3-layer), FC1 showed no significant difference (18.22 vs 18.31, p=0.71), while FC2 and FC3 increased (FC2: 8.99→9.98, p=0.0003; FC3: 1.72→1.88, p=0.017). The rank-5 ablation gap of 6.1pp on MNIST confirmed that corrupted models have lower effective rank despite the ambiguous spectral norm signal on deeper architectures.

**v2 revision:** Direct rank metrics show **corruption reduces weight scale, not rank**. On MNIST (h=16, v2): FC2 spectral norm −47% (p=3e-9), but **stable rank unchanged (p=0.32)**, effective rank slightly up (p=3e-3). The v0 inference "spectral norm drop ⇒ lower-rank memorization" is **withdrawn**. FDR increases under corruption (0.84 → 1.50, p=3e-6), indicating *increased* class separability at the representation level.

---

## Proposition 3: Rank-One Edit Magnitude Is Inversely Related to Class Boundary Overlap

**Informal statement:** The rank-one delta-norm measures the geometric separability of a class in hidden space — larger edits correspond to more orthogonal class boundaries, smaller edits to overlapping ones.

**Argument sketch:** The rank-one update to W2 is `Δ = outer(v - W·u, u) / (u·u)` where u is the mean hidden activation of the target class and v is the desired output. The norm of Δ scales with `||v - W·u||`, which is the distance from the current output for class u to the desired output. In a clean model where classes are orthogonally separated in hidden space, moving class k to a new prediction requires a large perturbation because its hidden representation is far from the decision boundary of any other class. In a corrupted model, class boundaries overlap — hidden representations of different classes occupy similar regions — so a smaller perturbation suffices to shift the prediction. Thus, Δ-norm serves as a geometric separability probe.

**v2 Empirical support (validated):**
- **Delta-norm dose-response**: 1.78× (5% noise) → 2.66× (50%), monotone, all p≤1e-06 (clean ref 0.801). Confirms delta-norm tracks memorization load continuously.
- **Rank-one intervention on held-out EVAL data**: fc2-only recovery +10.9 to +16.9pp (all p<0.02); fc1-only exactly 0.0pp in every config; sequential fc2→fc1 (3.5–8.1pp) worse than fc2-only. Edits act only at the output-adjacent layer.
- **Random-norm-matched null**: Degenerate (all 20 nulls recover 0.0pp, zero variance); z undefined; empirical p at permutation floor 1/21. Reported honestly — no "signal ratio = ∞" claims.

**Terminology:** "ROME" in v0 → **"ROME-inspired closed-form class-mean rank-one edit"** in v2 (no causal tracing, no key/value covariance constraint). EDIT/EVAL firewall enforced.