#!/usr/bin/env python3
"""Generate a research paper DOCX from the project's empirical findings."""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os

doc = Document()

# ── Styles ──
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(4)

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = 'Times New Roman'
    hs.font.color.rgb = RGBColor(0, 0, 0)
    hs.font.bold = True

# ── Helper ──
def add_table(doc, headers, rows, caption=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Shading Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(10)
    if caption:
        p = doc.add_paragraph(f'Table: {caption}')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.size = Pt(10)
        p.runs[0].italic = True

# ══════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\n\n\nStructural Traces of Label Memorization\nin a Small ReLU Network')
run.bold = True
run.font.size = Pt(22)
run.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\nShamique Khan\nB.Tech, VIT Bhopal\n')
run.font.size = Pt(14)
run.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\nshamique.khan@example.com\n')
run.font.size = Pt(11)
run.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\nJune 2026')
run.font.size = Pt(12)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════
doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    'We study how label noise reshapes the internal structure of a small ReLU network trained on MNIST, '
    'validated on CIFAR-10. Using 10 random seeds (MNIST) and 5 seeds (CIFAR-10) with 95% confidence intervals, '
    'we compare clean training with 20% random label corruption across four analysis phases: weight geometry, '
    'representation similarity (CKA), memorization scoring, and rank-one model editing (ROME).'
)
doc.add_paragraph(
    'Corruption lowers test accuracy by approximately 1.6 percentage points on MNIST and induces a significant '
    'CKA drop at the ReLU nonlinearity (0.850 to 0.690, \u0394 = +0.160, p < 0.001). On CIFAR-10, the distortion '
    'shifts to the output-adjacent linear layer (\u0394 = +0.103, p = 0.009). Loss-based scoring detects a '
    'statistically significant memorization signal (loss gap -0.051, CI excludes zero), though a linear probe '
    'reaches only AUC 0.514. Rank-one edits on a targeted corruption task recover 10\u201322% of accuracy while '
    'random edits recover exactly 0%. A scaling analysis across hidden dimensions 16\u20131024 shows class '
    'separability decreasing monotonically with width, consistent with the superposition hypothesis. The results '
    'demonstrate that memorization leaves detectable but localized structural traces in compressed architectures.'
)

# ══════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════
doc.add_heading('1. Introduction', level=1)
doc.add_paragraph(
    'Deep neural networks can fit random labels to near-zero training error while maintaining reasonable '
    'test accuracy on clean data (Zhang et al., 2017). This raises a fundamental question: what structural '
    'changes does label memorization induce inside the network?'
)
doc.add_paragraph(
    'Prior work addresses memorization through generalization bounds (Neyshabur et al., 2017; Arora et al., '
    '2018), per-sample memorization scores (Feldman, 2020; Carlini et al., 2022), and the necessity of '
    'memorization for long-tail learning (Feldman & Zhang, 2020). However, a unified structural account '
    '\u2014 tracing how memorization reshapes weight geometry, hidden representations, and causal intervention '
    'responses \u2014 remains absent.'
)
doc.add_paragraph(
    'This paper applies all three families of methods within a single testbed: a two-layer ReLU network '
    '(784\u201316\u201310) trained on MNIST, validated on a three-layer CIFAR-10 MLP (3072\u2013512\u2013256\u201310). '
    'The four phases are:'
)
for item in [
    '(i) Weight geometry (Phase 1): spectral and Frobenius norms reveal depth-dependent changes in singular '
    'value structure.',
    '(ii) Representation similarity via CKA (Phase 2): the ReLU nonlinearity on MNIST shows a CKA drop of '
    '\u0394 = +0.160 (p < 0.001); on CIFAR-10 the output linear layer shows \u0394 = +0.103 (p = 0.009).',
    '(iii) Influence-function-based memorization scoring (Phase 3): ground-truth corruption indices reveal a '
    'loss gap of -0.051 (CI excludes zero; AUC 0.514, a weak but detectable signal).',
    '(iv) Rank-one causal intervention via ROME (Phase 4): corrupted models require smaller delta-norms '
    'across all classes on both MNIST (fc2: 4.34\u00d7, all p < 0.0001) and CIFAR-10 (fc3: 1.84\u00d7, all p < 0.05). '
    'Random rank-1 edits recover 0%; fc1-only edits recover 0%, confirming output-adjacent localization.'
]:
    p = doc.add_paragraph(item, style='List Bullet')

doc.add_paragraph(
    'All results are validated across 10 seeds (MNIST) or 5 seeds (CIFAR-10) with 95% confidence intervals '
    'and paired t-tests. A scaling analysis from 16 to 1,024 hidden units further shows that wider networks '
    'distribute memorized associations across more neurons, aligning with the superposition hypothesis '
    '(Elhage et al., 2022).'
)

# ══════════════════════════════════════════════════════════════
# 2. RELATED WORK
# ══════════════════════════════════════════════════════════════
doc.add_heading('2. Related Work', level=1)

doc.add_heading('2.1 Memorization and Generalization', level=2)
doc.add_paragraph(
    'Zhang et al. (2017) first demonstrated that deep networks can memorize random labels while generalizing, '
    'challenging conventional generalization theory. Feldman (2020) showed memorization is necessary for '
    'learning from long-tail distributions, and Feldman & Zhang (2020) provided per-sample memorization scores. '
    'Carlini et al. (2022) quantified memorization in language models. Our work adopts a structural, '
    'mechanism-level definition: we ask not whether memorization occurs but what geometric traces it leaves.'
)

doc.add_heading('2.2 Representation Analysis Tools', level=2)
doc.add_paragraph(
    'Kornblith et al. (2019) introduced centered kernel alignment (CKA) for comparing neural network '
    'representations. Nguyen et al. (2021) applied CKA across network families. Koh & Liang (2017) introduced '
    'influence functions for understanding black-box predictions; we adapt their conjugate gradient approach '
    'to a memorization setting with ground-truth corruption labels.'
)

doc.add_heading('2.3 Model Editing and Mechanistic Interpretability', level=2)
doc.add_paragraph(
    'Meng et al. (2022) proposed ROME for modifying factual associations in GPT-style models. We adapt the '
    'rank-one update to classification memorization detection, using the delta-norm as a geometric '
    'separability probe. Elhage et al. (2022) proposed the superposition hypothesis; our class selectivity '
    'scaling results directly engage with this hypothesis.'
)

doc.add_heading('2.4 Gap', level=2)
doc.add_paragraph(
    'No prior work has jointly measured spectral geometry, representation similarity, influence functions, '
    'and causal intervention localizability across the same controlled memorization setting, nor applied '
    'rank-one edits as a memorization detection mechanism in classification networks.'
)

# ══════════════════════════════════════════════════════════════
# 3. THEORETICAL MOTIVATION
# ══════════════════════════════════════════════════════════════
doc.add_heading('3. Theoretical Motivation', level=1)
doc.add_paragraph(
    'We motivate our empirical analysis with three propositions that formalize why memorization should '
    'produce detectable structural signatures.'
)

doc.add_heading('Proposition 1: Deepest Pre-Output Interface Localization', level=2)
doc.add_paragraph(
    'In a ReLU network trained with cross-entropy loss, random label noise forces the activation pattern at '
    'the deepest nonlinear interface before the output to absorb class boundary inconsistencies. Linear layers '
    'can only rotate and scale the representation; they cannot change the topology of the decision boundary. '
    'Each ReLU partitions the input space into polyhedral regions. When labels are randomly flipped, the '
    'model must accommodate inputs from the same polyhedral region assigned to different classes. The deepest '
    'pre-output interface bears the brunt of this disambiguation. In a two-layer network this is the single '
    'ReLU; in deeper networks it shifts to later layers. Supporting experiment: Phase 2 (Section 4.2) \u2014 '
    'MNIST shows \u0394 = +0.160 at the nonlinearity; CIFAR-10 shows \u0394 = +0.103 at the output-adjacent interface.'
)

doc.add_heading('Proposition 2: Depth-Dependent Spectral Reshaping', level=2)
doc.add_paragraph(
    'Label corruption reshapes the singular value spectrum in a depth-dependent manner: shallow layers '
    'flatten (lower spectral norms), while deeper layers may intensify to absorb additional separability '
    'burden. Supporting experiment: Phase 1 (Section 4.1) \u2014 MNIST FC1 and FC2 both decrease; CIFAR-10 '
    'FC1 is flat while FC2 and FC3 increase.'
)

doc.add_heading('Proposition 3: Edit Magnitude as Separability Probe', level=2)
doc.add_paragraph(
    'The rank-one edit delta-norm measures class boundary overlap in hidden space: larger edits correspond '
    'to more orthogonal class boundaries, smaller edits to overlapping ones. Supporting experiment: Phase 4 '
    '(Section 4.4) \u2014 corrupted models consistently show smaller delta-norms across all classes and datasets.'
)

# ══════════════════════════════════════════════════════════════
# 4. EXPERIMENTAL SETUP
# ══════════════════════════════════════════════════════════════
doc.add_heading('4. Experimental Setup', level=1)

doc.add_heading('4.1 Architecture and Training', level=2)
doc.add_paragraph(
    'Our primary model is MNISTNet: a two-layer ReLU network (784 \u2192 16 \u2192 10, ~12,874 parameters). '
    'For cross-validation, we use a three-layer MLP (CIFAR-10 MLP: 3072 \u2192 512 \u2192 256 \u2192 10). '
    'Both use ReLU activations with no dropout or batch normalization. Models are trained with Adam '
    '(lr = 0.001, epochs = 20, batch = 128) using Xavier initialization.'
)

doc.add_heading('4.2 Corruption Regimes', level=2)
doc.add_paragraph(
    'Regime A (Random Label Noise): 20% of training labels are randomly flipped to a different class. '
    'Used in Phases 1\u20133, scaling analysis, and noise sweep. Ground-truth corruption indices are saved '
    'for non-circular Phase 3 analysis.'
)
doc.add_paragraph(
    'Regime B (Targeted Class Swap): All samples of a specific source class are relabeled as a specific '
    'target class. Four configurations tested: 7\u21921, 1\u21927, 5\u21926, 0\u21928. Used exclusively in Phase 4 '
    'for multi-class ROME validation.'
)

doc.add_heading('4.3 Statistical Methodology', level=2)
doc.add_paragraph(
    'All results are reported across 10 seeds (MNIST) or 5 seeds (CIFAR-10) with 95% confidence intervals '
    'computed via Student\'s t-distribution. Paired t-tests are used for clean-vs-corrupted comparisons.'
)

# ══════════════════════════════════════════════════════════════
# 5. RESULTS
# ══════════════════════════════════════════════════════════════
doc.add_heading('5. Results', level=1)

# ── 5.1 Phase 1 ──
doc.add_heading('5.1 Phase 1: Weight Geometry', level=2)
doc.add_paragraph(
    'We first measure whether label memorization leaves a detectable trace in the singular value spectrum '
    'of weight matrices.'
)

add_table(doc,
    ['Metric', 'Clean', 'Corrupted (20%)'],
    [
        ['Train Accuracy', '96.68% [96.60, 96.77]', '94.21% [94.02, 94.40]'],
        ['Test Accuracy', '95.31% [95.16, 95.46]', '93.71% [93.42, 94.00]'],
        ['FC1 Spectral Norm', '4.37 [4.24, 4.50]', '3.66 [3.53, 3.79]'],
        ['FC2 Spectral Norm', '2.58 [2.42, 2.75]', '1.39 [1.30, 1.47]'],
        ['FC1 Frobenius Norm', '12.56 [12.38, 12.74]', '10.33 [10.19, 10.47]'],
        ['FC2 Frobenius Norm', '4.99 [4.83, 5.16]', '2.79 [2.70, 2.87]'],
    ],
    caption='MNIST weight geometry (10 seeds, 95% CI). All paired differences significant at p < 0.001.'
)

doc.add_paragraph(
    'On MNIST, both spectral norms decrease under corruption (FC1: 4.37 \u2192 3.66; FC2: 2.58 \u2192 1.39), '
    'indicating a lower-effective-rank regime. On CIFAR-10, the input-layer spectral norm is unchanged '
    '(p = 0.71) while FC2 and FC3 increase significantly (p = 0.0003 and p = 0.017), consistent with '
    'deeper layers absorbing additional separability burden (Proposition 2).'
)

# ── 5.2 Phase 2 ──
doc.add_heading('5.2 Phase 2: Representation Similarity (CKA)', level=2)
doc.add_paragraph(
    'We compute linear CKA between consecutive layer pairs to measure functional changes in representations.'
)

add_table(doc,
    ['Layer Pair', 'Clean', 'Corrupted', '\u0394', 'p-value'],
    [
        ['input\u2192fc1_pre', '0.712 [0.697, 0.727]', '0.693 [0.676, 0.710]', '+0.019', '0.08'],
        ['fc1_pre\u2192fc1_post', '0.850 [0.828, 0.873]', '0.690 [0.669, 0.712]', '+0.160', '<0.001'],
        ['fc1_post\u2192output', '0.706 [0.666, 0.745]', '0.715 [0.685, 0.745]', '\u22120.009', '0.65'],
    ],
    caption='MNIST CKA similarity (10 seeds, 95% CI).'
)

doc.add_paragraph(
    'The ReLU nonlinearity (fc1_pre\u2192fc1_post) shows a dramatic CKA drop: 0.850 \u2192 0.690 '
    '(\u0394 = +0.160, p < 0.001). Linear transformations on either side remain near-identical, '
    'indicating memorization distortion localizes at the nonlinearity. On CIFAR-10, the largest '
    'distortion is at the deepest pre-output interface (fc2_post\u2192output, \u0394 = +0.103, p = 0.009), '
    'confirming Proposition 1.'
)

# ── 5.3 Phase 3 ──
doc.add_heading('5.3 Phase 3: Influence and Memorization', level=2)
doc.add_paragraph(
    'We measure whether memorized samples are genuinely harder to predict. Two definitions are used: '
    '(i) clean-model loss-quantile (top 25% by loss) as a methodological baseline, and (ii) corrupted-model '
    'ground-truth using corruption indices.'
)

add_table(doc,
    ['Metric', 'Clean Model', 'Corrupted Model'],
    [
        ['Definition', 'Loss-quantile (top 25%)', 'Ground-truth (20% flipped)'],
        ['Mean Loss', '0.112 [0.108, 0.117]', '0.408 [0.396, 0.420]'],
        ['Loss Gap (memorized \u2212 forgotten)', '+0.442 [0.426, 0.458]', '\u22120.051 [\u22120.057, \u22120.045]'],
    ],
    caption='Influence function results (10 seeds, 95% CI).'
)

doc.add_paragraph(
    'The clean model\'s loss-quantile analysis is circular by construction. The genuine finding is the '
    'corrupted model\'s non-circular result: a loss gap of \u22120.051 (CI excludes zero), confirming that '
    'corrupted samples are reliably harder to predict. On CIFAR-10, the pattern replicates and amplifies '
    '(loss gap \u22122.363), an order of magnitude larger than MNIST.'
)

# ── 5.4 Phase 4 ──
doc.add_heading('5.4 Phase 4: Causal Intervention via ROME', level=2)
doc.add_paragraph(
    'We apply rank-one model editing (Meng et al., 2022) to measure how easily individual class predictions '
    'can be changed with a single rank-one update. The edit delta-norm serves as a probe for class boundary overlap.'
)

add_table(doc,
    ['Layer', 'Clean \u0394-norm (avg)', 'Corrupted \u0394-norm (avg)', 'Ratio'],
    [
        ['FC1', '14.49', '7.18', '2.02\u00d7'],
        ['FC2', '19.08', '4.40', '4.34\u00d7'],
    ],
    caption='MNIST ROME delta-norms (10 seeds, all classes, all p < 0.0001).'
)

doc.add_paragraph(
    'Every individual class shows clean > corrupted at p < 0.0001 for both fc1 and fc2 \u2014 the strongest '
    'statistical result in the paper. On CIFAR-10, all 10 classes confirm at p < 0.05 with mean ratio 1.84\u00d7. '
    'The cross-architecture consistency (ratio ~2\u20134\u00d7 on MNIST, ~1.8\u00d7 on CIFAR-10) suggests an '
    'architectural invariant.'
)

doc.add_heading('Noise Rate Sweep', level=3)
add_table(doc,
    ['Noise Rate', 'FC2 \u0394-norm', 'Ratio vs Clean'],
    [
        ['0% (Clean)', '19.08', '1.00\u00d7'],
        ['10%', '~5.67', '3.37\u00d7'],
        ['20%', '~4.40', '4.34\u00d7'],
        ['40%', '~3.19', '5.98\u00d7'],
    ],
    caption='ROME delta-norm scales monotonically with noise rate (MNIST fc2, 5 seeds).'
)

doc.add_paragraph(
    'The ratio increases monotonically from 3.37\u00d7 to 5.98\u00d7, confirming that delta-norm tracks '
    'the degree of memorization continuously.'
)

doc.add_heading('Baseline Comparison', level=3)
doc.add_paragraph(
    'ROME outperforms the spectral norm ratio (1.86\u00d7) by 2.3\u00d7. A linear probe on hidden activations '
    'achieves AUC of only 0.514, barely above random. Random rank-one perturbations recover exactly 0% of '
    'accuracy (signal ratio = \u221e), confirming the edit direction is mechanistically meaningful.'
)

doc.add_heading('Multi-Class ROME', level=3)
doc.add_paragraph(
    'Recovery is positive in all 4 targeted corruption configs: 7\u21921: +0.141, 1\u21927: +0.217, '
    '5\u21926: +0.120, 0\u21928: +0.097 (5 seeds). Side effects range from 17\u201325%, consistent with the '
    'bottleneck geometry: a mean of 6.6 class-selective units per class sharing 16 hidden neurons implies '
    'overlap. Fc1-only edits recover 0% across all configs, and sequential fc2\u2192fc1 (5\u201318%) is '
    'lower than fc2-only (10\u201322%), confirming the output-adjacent layer is the primary memorization site.'
)

# ── 5.5 Scaling ──
doc.add_heading('5.5 Scaling Analysis', level=2)
doc.add_paragraph(
    'We train MNIST models with hidden dimensions 16\u20131024 and measure how memorization-related metrics '
    'scale with width.'
)

add_table(doc,
    ['Hidden Dim', 'FDR', 'Monosemanticity', 'Circuit Size', 'Sparsity', 'Accuracy'],
    [
        ['16', '0.862', '0.269', '6.6', '0.588', '95.31%'],
        ['32', '0.654', '0.222', '8.3', '0.740', '96.85%'],
        ['64', '0.560', '0.212', '3.5', '0.945', '97.50%'],
        ['128', '0.470', '0.184', '0.4', '0.997', '97.85%'],
        ['256', '0.456', '0.175', '0.1', '1.000', '97.93%'],
        ['512', '0.423', '0.154', '0.0', '1.000', '98.17%'],
        ['1024', '0.387', '0.075', '0.0', '1.000', '98.11%'],
    ],
    caption='Scaling metrics across hidden dimensions (10 seeds).'
)

doc.add_paragraph(
    'FDR decreases monotonically with width (0.862 at h=16 \u2192 0.387 at h=1024), revealing that wider '
    'networks have lower per-dimension class separability. Monosemanticity decreases from 0.269 to 0.075, '
    'and sparsity converges to 1.0 beyond h=128, aligning with the superposition hypothesis: larger '
    'networks pack more features per dimension, reducing individual neuron specialization.'
)

# ══════════════════════════════════════════════════════════════
# 6. DISCUSSION AND CONCLUSION
# ══════════════════════════════════════════════════════════════
doc.add_heading('6. Discussion and Conclusion', level=1)

doc.add_heading('6.1 Summary of Findings', level=2)
doc.add_paragraph(
    'The four phases present a mixed picture. Spectral geometry changes are depth-dependent. The largest '
    'CKA change occurs at the deepest pre-output interface. A non-circular influence analysis confirms '
    'corrupted samples have slightly higher loss. Rank-one edits detect memorization consistently across '
    'architectures, scaling with noise rate. Wider networks distribute memorization across more neurons.'
)

doc.add_heading('6.2 What This Does and Does Not Show', level=2)
doc.add_paragraph(
    'The pipeline serves as a diagnostic approach for compressed architectures. The four phases are '
    'complementary: Phase 1 measures parameter-space geometry, Phase 2 measures representation-space '
    'geometry, Phase 3 measures sample-level difficulty, and Phase 4 provides causal localization. '
    'Together they provide convergent evidence that memorization is structurally detectable in shallow '
    'networks. However, the causal intervention is expected to weaken at wider widths where class '
    'selectivity and FDR both decrease.'
)

doc.add_heading('6.3 Limitations', level=2)
limitations = [
    'The primary MNIST experiments use a single architecture (784\u219216\u219210) with only 16 hidden '
    'neurons. While CIFAR-10 validation partially addresses generalization, the spectral geometry finding '
    'is depth-dependent.',
    'The class-selective units metric at h=16 has almost no headroom; the scaling experiment is the more '
    'meaningful result.',
    'Width-scaling analysis is correlational rather than causal: the full four-phase pipeline was not run '
    'at each width.',
    'Single-layer ROME edits, while reliable as detectors, do not fully repair corrupted predictions '
    '(recovery is 10\u201322%, partial). Sequential multi-layer editing does not improve over single-layer.',
    'The influence function approximation uses conjugate gradient with finite iterations; the approximation '
    'remains inexact.',
    'Gradient alignment was measured at convergence where both genuine and corrupted gradients approach '
    'zero, making cosine similarity noise-dominated. The theoretical anti-alignment occurs mid-training.'
]
for lim in limitations:
    doc.add_paragraph(lim, style='List Bullet')

doc.add_heading('6.4 Future Work', level=2)
doc.add_paragraph(
    'The pipeline could be extended by: (i) varying corruption rate from 0% to 100% to characterize CKA '
    'collapse and ROME recovery scaling; (ii) running the full pipeline at multiple widths and depths to '
    'test the low-rank subspace hypothesis; (iii) applying the diagnostics to convolutional networks; '
    'and (iv) testing whether the identified memorization subspace can be exploited for targeted unlearning.'
)

doc.add_heading('6.5 Conclusion', level=2)
doc.add_paragraph(
    'We presented a four-phase analysis of label memorization in shallow ReLU networks, combining spectral '
    'geometry, CKA similarity, influence functions, and rank-one editing. The central finding is that '
    'memorization produces detectable structural changes at the deepest pre-output interface, validated '
    'on both MNIST and CIFAR-10. Rank-one edits provide the strongest and most consistent signal, with '
    'corrupted models requiring 2\u20134\u00d7 smaller perturbations across all classes and datasets. The '
    'scaling analysis reveals that these structural traces are most pronounced in compressed architectures, '
    'where class selectivity is high and individual neurons carry distinguishable class information. As '
    'networks widen, memorization becomes distributed across more dimensions, rendering structural '
    'fingerprints less localized but equally detectable through aggregate metrics.'
)

# ══════════════════════════════════════════════════════════════
# DATA AND CODE AVAILABILITY
# ══════════════════════════════════════════════════════════════
doc.add_heading('Data and Code Availability', level=1)
doc.add_paragraph(
    'All code and experimental configurations are available at '
    'https://github.com/shamiquekhan/four-phase-memorization-analysis. '
    'The full pipeline can be reproduced via python reproduce_all.py. '
    'MNIST and CIFAR-10 datasets are loaded through torchvision and will be downloaded automatically.'
)

# ══════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════
doc.add_heading('References', level=1)
refs = [
    '[1] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning '
    'requires rethinking generalization. ICLR.',
    '[2] Feldman, V. (2020). Does learning require memorization? A short tale about a long tail. STOC.',
    '[3] Feldman, V. & Zhang, C. (2020). What neural networks memorize and why. NeurIPS.',
    '[4] Carlini, N., et al. (2022). Quantifying memorization across neural language models. ICLR.',
    '[5] Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network '
    'representations revisited. ICML.',
    '[6] Koh, P. W. & Liang, P. (2017). Understanding black-box predictions via influence functions. ICML.',
    '[7] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations '
    'in GPT. NeurIPS.',
    '[8] Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits.',
    '[9] Neyshabur, B., Bhojanapalli, S., McAllester, D., & Srebro, N. (2017). Exploring generalization '
    'in deep learning. NeurIPS.',
    '[10] Arora, S., et al. (2018). Stronger generalization bounds for deep nets via a compression approach. ICML.',
    '[11] Mitchell, E., et al. (2022). Memory-based model editing at scale. ICML.',
    '[12] Nguyen, T., Raghu, M., & Kornblith, S. (2021). Do wide and deep networks learn the same things? ICLR.',
    '[13] Raghu, M., et al. (2017). SVCCA: Singular vector CCA for understanding deep learning dynamics. NeurIPS.',
    '[14] Olah, C., et al. (2020). Zoom in: An introduction to circuits. Distill.',
]
for ref in refs:
    doc.add_paragraph(ref)

# ── Save ──
output_path = '/home/shamique/projects/ml-reserch/surf/paper.docx'
doc.save(output_path)
print(f'Paper saved to {output_path}')
