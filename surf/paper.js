const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType } = require('docx');
const fs = require('fs');

const FONT = "Times New Roman";
const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function runs(parts) {
  return parts.map(pt => new TextRun({ text: pt.text, font: FONT, size: 22, bold: !!pt.bold, italics: !!pt.italics, superScript: !!pt.sup }));
}
function para(parts, opts = {}) {
  return new Paragraph({ spacing: { after: 160, line: 276, ...(opts.spacing||{}) }, alignment: opts.align || AlignmentType.JUSTIFIED, children: runs(parts) });
}
function heading(text, level) {
  return new Paragraph({ heading: level, children: [new TextRun({ text, font: FONT, bold: true })], spacing: { before: 260, after: 140 } });
}
function caption(text) {
  return new Paragraph({ spacing: { after: 100, before: 80 }, children: [new TextRun({ text, font: FONT, size: 20, bold: true, italics: true })] });
}
function cell(text, opts = {}) {
  return new TableCell({
    borders,
    width: { size: opts.width || 1500, type: WidthType.DXA },
    shading: opts.header ? { fill: "E7E7E7", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, font: FONT, size: 18, bold: !!opts.header })] })]
  });
}
function table(headerRow, rows, widths) {
  const totalWidth = widths.reduce((a,b)=>a+b,0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ children: headerRow.map((h,i)=>cell(h,{header:true, width:widths[i]})) }),
      ...rows.map(r => new TableRow({ children: r.map((c,i)=>cell(c,{width:widths[i]})) }))
    ]
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: FONT },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "refList", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
    ]
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [
      // Title (anonymized — no author block)
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Structural Fingerprints of Label Memorization in Shallow Neural Networks: A Four-Phase Empirical Study", bold: true, size: 28, font: FONT })] }),

      heading("Abstract", HeadingLevel.HEADING_1),
      para([{text:"Understanding what neural networks memorize versus what they genuinely learn is a central open question in interpretability research. We present a four-phase empirical study of label memorization in a two-layer fully connected network (784-16-10, approximately 12,874 parameters) trained on MNIST, validated on a three-layer network trained on CIFAR-10. Clean networks are compared against networks trained with 20% random label noise across ten seeds (five for CIFAR-10), with 95% confidence intervals reported throughout. Weight-geometry analysis shows that corruption reduces MNIST spectral norms in both layers (FC1: 4.37 to 3.66; FC2: 2.58 to 1.39; both p < 0.001) but produces a depth-dependent pattern on CIFAR-10, where the input layer is unaffected (p = 0.71) while deeper layers intensify (p < 0.05). Representation-similarity analysis (centered kernel alignment, CKA) localizes the largest corruption-induced distortion to the deepest pre-output interface in both architectures \u2014 the ReLU nonlinearity on MNIST (0.850 to 0.690, p < 0.001) and the final linear layer on CIFAR-10 (0.580 to 0.477, p = 0.009) \u2014 indicating the effect tracks architectural depth rather than a fixed layer. A causal probe based on rank-one model editing (ROME) shows that the edit magnitude needed to overwrite a class association is 2.0 to 4.3 times larger in clean than corrupted MNIST models (p < 0.0001 for all ten classes at both layers) and 1.8 times larger on CIFAR-10 (p < 0.05, all ten classes), scaling monotonically with corruption rate (3.4\u00d7 to 6.0\u00d7 across 10\u201340% noise); this causal probe substantially outperforms a spectral-norm ratio (1.9\u00d7) and a linear probe on hidden activations (AUC = 0.514, near chance). Targeted class-swap edits recover 9.7\u201321.7 percentage points of lost accuracy, and rank-eight weight approximations retain approximately 90% of full performance. These results converge on the view that memorization leaves a low-dimensional, editable structural signature that grows more diffuse as network width increases."}]),
      para([{text:"Keywords: ", bold:true},{text:"neural network memorization; representation similarity; centered kernel alignment; influence functions; rank-one model editing; interpretability; MNIST; CIFAR-10"}]),

      heading("1. Introduction", HeadingLevel.HEADING_1),
      para([{text:"Modern neural networks routinely achieve near-zero training error even on datasets with deliberately corrupted labels, a phenomenon that fundamentally challenges classical notions of the bias\u2013variance trade-off [1]. Zhang et al. famously demonstrated that deep neural networks can perfectly memorize randomly labeled training data while simultaneously achieving strong generalization on clean test sets, suggesting that the traditional dichotomy between memorization and learning is inadequate for understanding these models [1]. More recently, Feldman [5] showed that long-tailed data distributions actually require a degree of memorization for optimal generalization, adding nuance to the picture: not all memorization is harmful, and some may be necessary."}]),
      para([{text:"Despite this growing body of knowledge, a fundamental question remains open: how is memorization physically realized inside a trained network? Does it correspond to identifiable changes in weight geometry, in the structure of internal representations, or in specific, localizable computational circuits? Answering this question has practical implications for interpretability, robustness, and the ability to selectively correct undesired memorized associations without retraining from scratch."}]),
      para([{text:"Three distinct lines of interpretability research bear on this question but have largely been pursued in isolation. First, representation similarity metrics \u2014 most prominently centered kernel alignment (CKA) [2] \u2014 allow direct comparison of how networks process information across layers and training conditions, but have rarely been used to systematically contrast clean and corrupted training regimes, and recent work has raised concerns about how reliably CKA tracks genuine functional change rather than manipulable artifacts [8]. Second, influence functions [3] provide a principled, gradient-based estimate of how much each individual training example shapes a model's parameters and have been proposed as a tool for identifying memorized examples; however, validating these scores against known ground-truth corruption indices, and benchmarking them against alternative probes, is uncommon in the literature. Third, rank-one model editing (ROME) [4], originally developed to localize and surgically edit factual associations in large language models, offers a causal lever: if a targeted weight edit to a hypothesized memorization circuit measurably restores behavior on affected examples, this constitutes causal \u2014 rather than purely correlational \u2014 evidence for the location of memorized information."}]),
      para([{text:"This paper unifies all three approaches within a deliberately minimal primary testbed \u2014 a two-layer fully connected network (784-16-10 units, approximately 12,874 trainable parameters) trained on the MNIST handwritten digit dataset [6] \u2014 and then extends the central claims to a second, deeper architecture (a three-layer network trained on CIFAR-10) to test whether the findings are specific to one network or reflect a more general pattern. The small hidden dimension of the primary testbed forces the network into a compressed representation regime in which individual neurons are more likely to specialize, making circuit-level interpretation tractable; the CIFAR-10 replication then asks whether the same qualitative signatures persist, and at which layer, once depth is added."}]),
      para([{text:"This study makes six primary contributions. First, we demonstrate that label memorization measurably alters weight geometry on MNIST, and that this effect becomes depth-dependent on the deeper CIFAR-10 architecture, where shallow layers are unaffected while output-adjacent layers intensify. Second, we show that memorization disproportionately disrupts the representation transformation at the deepest pre-output interface of a network \u2014 the ReLU nonlinearity in the two-layer model and the final linear layer in the three-layer model \u2014 and we explicitly engage with recent critiques of CKA's reliability, arguing that this cross-architecture, depth-tracking replication is difficult to explain as a manipulable artifact. Third, we validate influence-function-based memorization scores against ground-truth corruption indices and directly benchmark this correlational signal against two alternatives \u2014 a spectral-norm ratio and a causal ROME-based probe \u2014 showing the causal probe is markedly more discriminative. Fourth, we demonstrate through targeted ROME edits and a full-spectrum rank-ablation study that memorized class associations occupy a compact, identifiable, and partially recoverable subspace of the weight matrix, and we report a noise-rate sweep confirming the ROME signal scales monotonically with corruption load. Fifth, a width-scaling analysis from 16 to 1,024 hidden units reveals that class separability, measured by the Fisher discriminant ratio (FDR), decreases monotonically with capacity. Sixth, we package the four-phase pipeline \u2014 now validated across two architectures \u2014 as a reusable diagnostic toolkit for probing memorization in other settings."}]),

      heading("2. Related Work", HeadingLevel.HEADING_1),
      heading("2.1 Memorization and Generalization", HeadingLevel.HEADING_2),
      para([{text:"The relationship between memorization and generalization in deep learning has been an active area of research since Zhang et al. [1] showed that state-of-the-art models can simultaneously achieve perfect memorization on random labels and strong generalization on structured data. Subsequent theoretical work has shown that many common regularization techniques \u2014 dropout, weight decay, batch normalization \u2014 reduce but do not eliminate memorization [1]. Feldman [5] further argued that memorization of rare patterns is not merely an artifact of overparameterization but may be necessary for achieving optimal generalization in long-tailed distributions, reframing memorization as a feature rather than a bug in many practical settings."}]),
      heading("2.2 Centered Kernel Alignment", HeadingLevel.HEADING_2),
      para([{text:"Representation similarity analysis provides tools for comparing the internal geometry of neural networks. CKA [2], as introduced by Kornblith et al., is invariant to orthogonal transformations and isotropic scaling, making it well-suited to comparing representations across networks trained with different random seeds or under different conditions. Prior work has used CKA to study representational convergence across architectures and training procedures, but systematic application to the question of how memorization alters layer-wise representations remains relatively unexplored."}]),
      heading("2.3 Influence Functions and Sample Attribution", HeadingLevel.HEADING_2),
      para([{text:"Influence functions, introduced to machine learning by Koh and Liang [3], estimate the counterfactual impact of removing or up-weighting individual training examples on model predictions. They have been applied to tasks including mislabeled data detection and data poisoning analysis. A known limitation is that the quadratic approximation underlying influence functions may be inaccurate for non-convex networks; our use of conjugate-gradient estimation of the inverse Hessian-vector product follows standard practice for mitigating this issue, and our validation against ground-truth corruption indices, alongside a direct comparison with a causal probe (Section 4.4), provides empirical evidence on how much this correlational approximation can and cannot detect in our setting."}]),
      heading("2.4 Model Editing", HeadingLevel.HEADING_2),
      para([{text:"ROME [4], as introduced by Meng et al. for factual editing in large language models, frames targeted weight modification as a rank-one update problem. The key insight is that a single factual association can often be localized to a small subspace of a single layer's weight matrix, enabling surgical edits with minimal collateral effects. Extending this approach to memorization in classification networks \u2014 where the 'fact' to be edited is a specific class-label association \u2014 is a natural but underexplored direction."}]),
      heading("2.5 Reliability of Representation-Similarity Metrics", HeadingLevel.HEADING_2),
      para([{text:"A growing body of work cautions against treating CKA values as a direct readout of functional similarity. Davari et al. show that CKA is sensitive to a class of simple transformations \u2014 including translations of outlier points and transformations that preserve linear separability \u2014 and demonstrate that CKA values can be substantially manipulated without corresponding changes to a network's output behavior [8]. This raises a legitimate concern for any study, including this one, that uses a CKA drop as evidence of a meaningful representational change. We address this concern in two ways rather than by assumption: first, by replicating the CKA-collapse finding on a second, structurally different architecture (CIFAR-10's three-layer network) and showing that the location of the largest distortion shifts with depth in a manner consistent with a genuine, architecture-tracking phenomenon rather than a fixed numerical artifact (Section 4.2); second, by independently corroborating the same localization claim with a causal probe (ROME edit magnitude, Section 4.4) that does not rely on CKA at all. Separately, Nguyen et al. provide a theoretical account connecting label-noise memorization to a degradation (\u2018dilation\u2019) of neural collapse \u2014 the tendency of within-class representations to converge to a single point late in training \u2014 offering a complementary theoretical lens for why corrupted training would be expected to leave a representational, not just a weight-level, signature [9]."}]),

      heading("3. Methodology", HeadingLevel.HEADING_1),
      heading("3.1 Primary Architecture and Training (MNIST)", HeadingLevel.HEADING_2),
      para([{text:"All primary experiments use a two-layer fully connected network (MNISTNet) with 784 input units (flattened 28\u00d728 MNIST images), a hidden layer of 16 units with ReLU activation, and an output layer of 10 units corresponding to digit classes, totaling approximately 12,874 trainable parameters. Models are trained with the Adam optimizer [7] at a learning rate of 0.001 with batch size 128 for 20 epochs using cross-entropy loss. The small hidden dimension is a deliberate design choice: it forces the network into a compressed, interpretable regime in which individual neurons are more likely to specialize. Ten independent random seeds (42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999) are used for all primary comparisons."}]),
      heading("3.2 Cross-Architecture Validation (CIFAR-10)", HeadingLevel.HEADING_2),
      para([{text:"To test whether the primary findings are specific to the minimal two-layer testbed or reflect a more general pattern, a second architecture (CIFARNet, 3072-256-128-10, a three-layer fully connected network) is trained on CIFAR-10 under the same 20% label-noise protocol (Regime A, below). Because per-run training cost is substantially higher than for MNIST, this validation uses five random seeds (42, 123, 456, 789, 1024) rather than ten. All other training hyperparameters follow the MNIST protocol (Adam optimizer, cross-entropy loss) unless otherwise noted in the accompanying experiment configuration file. The CIFAR-10 architecture is used exclusively for the cross-architecture comparisons in Phases 1\u20133 and the ROME baseline comparison in Phase 4 (Sections 4.1, 4.2, 4.3, 4.4); the targeted class-swap intervention (Regime B) and the width-scaling analysis are conducted on MNIST only."}]),
      heading("3.3 Corruption Regimes", HeadingLevel.HEADING_2),
      para([{text:"Two distinct corruption regimes are employed for different phases of the analysis."}]),
      para([{text:"Regime A (random label noise) is used in Phases 1 through 3, the CIFAR-10 validation, and the noise-rate sweep. Twenty percent of training labels are randomly reassigned to a different class prior to training, independently for each seed (a default condition; 10% and 40% are additionally tested for the noise-rate sweep in Section 4.4). For MNIST this corrupts approximately 12,000 of 60,000 training examples at the default rate. The indices of corrupted examples are saved (corrupt_indices.npy) and reserved for post-hoc validation of memorization scores, ensuring that the Phase 3 analysis is not circular."}]),
      para([{text:"Regime B (targeted class swap) is used exclusively in Phase 4 on MNIST. All training examples belonging to a source class are relabeled to a target class, affecting approximately 5,900 to 6,000 examples per configuration. Four source\u2013target pairs are tested: 7\u21921, 1\u21927, 5\u21926, and 0\u21928. This creates a concentrated, easily localizable backdoor association suited to causal intervention experiments, in contrast to the diffuse, sample-level corruption of Regime A."}]),
      heading("3.4 Four-Phase Analysis Pipeline", HeadingLevel.HEADING_2),
      para([{text:"Phase 1: Weight Geometry. For each trained checkpoint, we compute the Frobenius and spectral norms of all weight matrices, the final-epoch gradient norm, and epoch-wise training and test accuracy and loss curves. Spectral norms quantify the maximum singular value of each weight matrix, providing a measure of the network's amplification factor along its most responsive direction."}]),
      para([{text:"Phase 2: Representation Similarity. Activations are extracted on held-out test examples (5,000 for MNIST) at successive points in the forward pass between each pair of adjacent layers. Linear CKA, computed via the Hilbert\u2013Schmidt independence criterion after zero-centering, quantifies the similarity of these representations between clean and corrupted models. Auxiliary diagnostics include activation sparsity, mean and standard deviation of activations, and principal component analysis (PCA)."}]),
      para([{text:"Phase 3: Influence and Memorization Scoring. Per-sample training loss is used to classify examples as memorized (top loss quartile when evaluated by a clean-trained model) or non-memorized. Approximate influence scores are computed via conjugate-gradient estimation of the inverse Hessian-vector product, following Koh and Liang [3]. To contextualize how discriminative this correlational signal actually is, a baseline comparison contrasts three probes for distinguishing clean from corrupted models: the spectral-norm ratio from Phase 1, a linear probe (logistic regression on hidden activations trained to predict per-sample corruption status, evaluated via AUC), and the causal ROME-based probe described below (Section 4.4)."}]),
      para([{text:"Phase 4: Causal Intervention via ROME. For each targeted configuration, a rank-one edit \u0394W = (v \u2212 Wu)u\u2009\u22ba\u2009/\u2009u\u2009\u22ba\u2009u is applied to a weight matrix W, where u is a key vector for the source class and v is a target output vector. We measure accuracy recovery on the source class, side effects on non-targeted classes, and the Frobenius norm of the edit \u2016\u0394W\u2016. Beyond the four targeted class-swap configurations, we additionally compute \u0394W for every class under ordinary (non-targeted) clean and corrupted models, on both MNIST and CIFAR-10, as a class-by-class baseline-comparison probe for memorization (Section 4.4), and repeat this at three noise rates (10%, 20%, 40%) to test whether the signal scales with corruption load. A complementary rank-ablation study replaces W with its best rank-k approximation for every k from 1 to 10 and measures the resulting accuracy degradation."}]),
      heading("3.5 Separability Metrics", HeadingLevel.HEADING_2),
      para([{text:"Two class-separability metrics are reported. The Fisher discriminant ratio (FDR) is computed as FDR = tr(S_B) / tr(S_W), where S_B and S_W are the between-class and within-class scatter matrices of hidden-layer activations. FDR is the primary metric for scaling comparisons because it is dimension-invariant, unlike a naive within/between distance ratio (\u03c3), which conflates geometric spread with dimensionality and is reported alongside FDR in Section 4.5 to illustrate this divergence. The Davies-Bouldin index was considered as an auxiliary diagnostic for cluster disentanglement but is not reported in the results below, as it did not yield additional information beyond FDR in this setting."}]),

      heading("4. Results", HeadingLevel.HEADING_1),
      heading("4.1 Phase 1: Weight Geometry", HeadingLevel.HEADING_2),
      para([{text:"Table 1 reports the principal weight-geometry results for MNIST, comparing models trained on clean data against models trained under Regime A (20% random label noise), aggregated across ten seeds with 95% confidence intervals."}]),
      caption("Table 1. Phase 1 results (MNIST): weight geometry and accuracy, clean vs. corrupted training (10 seeds, 95% CI)."),
      table(
        ["Metric","Clean (mean [95% CI])","Corrupted (mean [95% CI])","p-value"],
        [
          ["Test Accuracy (%)","95.31 [95.16, 95.46]","93.71 [93.42, 94.00]","<0.001"],
          ["FC1 Spectral Norm","4.37 [4.24, 4.50]","3.66 [3.53, 3.79]","<0.001"],
          ["FC2 Spectral Norm","2.58 [2.42, 2.75]","1.39 [1.30, 1.47]","<0.001"],
          ["FC1 Frobenius Norm","12.56 [12.38, 12.74]","10.33 [10.19, 10.47]","<0.001"],
          ["FC2 Frobenius Norm","4.99 [4.83, 5.16]","2.79 [2.70, 2.87]","<0.001"],
          ["Final Gradient Norm","0.94 [0.89, 0.98]","1.00 [0.96, 1.05]","<0.001"],
          ["Train Accuracy (%)","96.68 [96.60, 96.77]","94.21 [94.02, 94.40]","<0.001"],
        ],
        [2700,2350,2350,1100]
      ),
      para([{text:"Corrupted training reduces test accuracy only modestly, from 95.31% to 93.71%, despite 20% of training labels being incorrect. Training accuracy on the corrupted set reaches 94.21%, not 100%, because the 16-unit bottleneck lacks the capacity to memorize all 12,000 corrupted samples within 20 epochs; the network continues to learn the majority of genuine structure while simultaneously fitting the noise. Both spectral and Frobenius norms decrease significantly under corruption in both layers (all p < 0.001), consistent with the second layer bearing the primary burden of discriminative classification. The elevated final gradient norm under corruption (0.94 to 1.00) reflects the model's difficulty reconciling conflicting label signals at the end of training."}]),
      para([{text:"Table 1b extends this analysis to the deeper CIFAR-10 architecture (three-layer MLP, five seeds), which was not part of the original two-layer testbed but tests whether the weight-geometry effect generalizes."}]),
      caption("Table 1b. Phase 1 results (CIFAR-10): spectral norms, clean vs. corrupted training (5 seeds, 95% CI)."),
      table(
        ["Metric","Clean (mean [95% CI])","Corrupted (mean [95% CI])","p-value"],
        [
          ["FC1 Spectral Norm","18.22 [17.93, 18.51]","18.31 [17.93, 18.70]","0.71"],
          ["FC2 Spectral Norm","8.99 [8.83, 9.15]","9.98 [9.85, 10.10]","0.0003"],
          ["FC3 Spectral Norm","1.72 [1.68, 1.77]","1.88 [1.78, 1.98]","0.017"],
        ],
        [2700,2350,2350,1100]
      ),
      para([{text:"The CIFAR-10 result diverges from the MNIST pattern in an informative way: the input-adjacent layer (FC1) is statistically unchanged under corruption (p = 0.71), while the two deeper layers (FC2, FC3) increase under corruption (p = 0.0003 and p = 0.017, respectively). This is the opposite direction from the MNIST norms, which decrease, but the deeper interpretation is consistent across both architectures: the layers nearest the output \u2014 not the input \u2014 absorb the burden of resolving conflicting label signals. In the two-layer MNIST network the entire weight budget is output-adjacent by construction, so this burden manifests as an overall norm reduction; in the three-layer CIFAR-10 network it manifests as an increase concentrated in FC2 and FC3 while FC1 is left untouched. We return to this depth-dependent pattern in Section 4.2 and Section 5.2."}]),

      heading("4.2 Phase 2: Representation Similarity", HeadingLevel.HEADING_2),
      caption("Table 2. Phase 2 results (MNIST): CKA representation similarity and activation statistics (10 seeds, 95% CI)."),
      table(
        ["CKA Pair / Statistic","Clean (mean [95% CI])","Corrupted (mean [95% CI])","p-value"],
        [
          ["Input \u2192 FC1_pre","0.712 [0.697, 0.727]","0.693 [0.676, 0.710]","0.08"],
          ["FC1_pre \u2192 FC1_post (ReLU)","0.850 [0.828, 0.873]","0.690 [0.669, 0.712]","<0.001"],
          ["FC1_post \u2192 Output","0.706 [0.666, 0.745]","0.715 [0.685, 0.745]","0.65"],
          ["Activation Sparsity (FC1_post)","0.357 [0.328, 0.386]","0.576 [0.539, 0.612]","<0.001"],
          ["Mean Activation (FC1_post)","4.83 [4.39, 5.28]","1.53 [1.35, 1.71]","<0.001"],
          ["PC1 Variance Explained","0.263 [0.239, 0.288]","0.250 [0.237, 0.263]","0.25"],
        ],
        [3100,2350,2350,700]
      ),
      para([{text:"CKA between the network input and the pre-activation hidden layer (FC1_pre) changes only slightly under corruption (0.712 to 0.693, p = 0.08), indicating that the linear transformation performed by FC1 is comparatively stable across training regimes. By contrast, CKA between FC1_pre and the post-ReLU representation (FC1_post) drops sharply and significantly, from 0.850 to 0.690 (p < 0.001). Because this quantity measures how similar the transformation performed by the ReLU nonlinearity is between clean and corrupted models on the same test inputs, its collapse indicates that memorization is primarily encoded in which units are active for which inputs, rather than in the raw linear projection weights. Corrupted models exhibit higher activation sparsity (0.357 to 0.576) and substantially lower mean activation (4.83 to 1.53), indicating that noisy labels reorganize the representation into a more fragmented geometry."}]),
      caption("Table 2b. Phase 2 results (CIFAR-10): CKA representation similarity by layer pair (5 seeds, 95% CI)."),
      table(
        ["Layer Pair","Clean (mean [95% CI])","Corrupted (mean [95% CI])","\u0394","p-value"],
        [
          ["Input \u2192 FC1_pre","0.767 [0.748, 0.786]","0.765 [0.754, 0.777]","+0.002","0.88"],
          ["FC1_pre \u2192 FC1_post","0.819 [0.808, 0.830]","0.801 [0.796, 0.807]","+0.018","0.017"],
          ["FC1_post \u2192 FC2_pre","0.721 [0.698, 0.745]","0.749 [0.737, 0.761]","\u22120.028","0.08"],
          ["FC2_pre \u2192 FC2_post","0.177 [0.142, 0.211]","0.140 [0.109, 0.171]","+0.037","0.09"],
          ["FC2_post \u2192 Output","0.580 [0.528, 0.632]","0.477 [0.457, 0.498]","+0.103","0.009"],
        ],
        [2400,2100,2100,800,1100]
      ),
      para([{text:"On CIFAR-10, the largest CKA distortion is at FC2_post \u2192 Output (\u0394 = 0.103, p = 0.009), the final linear layer before the logits, while the first nonlinearity (FC1_pre \u2192 FC1_post) shows a much smaller, though still significant, effect (\u0394 = 0.018, p = 0.017). This is the cross-architecture replication referenced in Section 2.5: the distortion does not sit at a fixed numerical layer but consistently localizes to the deepest pre-output interface of whichever architecture is tested \u2014 the single ReLU nonlinearity in the two-layer network and the final linear layer in the three-layer network. A CKA value that could be freely manipulated without tracking functional structure would not be expected to relocate so systematically with architectural depth, which is the strongest evidence this study can offer, short of an independent similarity metric, that the collapse reflects a genuine representational change rather than an artifact of the kind documented by Davari et al. [8]."}]),

      heading("4.3 Phase 3: Influence and Memorization Scoring", HeadingLevel.HEADING_2),
      caption("Table 3. Phase 3 results: influence-based memorization scoring (10 seeds for MNIST, 5 for CIFAR-10, 95% CI)."),
      table(
        ["Metric","MNIST Clean","MNIST Corrupted","CIFAR-10 Clean","CIFAR-10 Corrupted"],
        [
          ["Mean Loss","0.112 [0.108, 0.117]","0.408 [0.396, 0.420]","0.407 [0.397, 0.417]","1.093 [1.074, 1.111]"],
          ["Accuracy (%)","96.68 [96.58, 96.79]","94.21 [93.98, 94.45]","85.77 [85.38, 86.17]","69.41 [68.93, 69.89]"],
          ["Memorized-Sample Definition","Loss-quantile (top 25%)","Ground-truth (20% flipped)","Loss-quantile (top 25%)","Ground-truth (20% flipped)"],
          ["Loss Gap (memorized \u2212 forgotten)","+0.442 [+0.426, +0.458]","\u22120.051 [\u22120.057, \u22120.045]","+1.417 [+1.382, +1.452]","\u22122.363 [\u22122.436, \u22122.291]"],
          ["Gradient Alignment (cosine sim.)","\u2014","+0.9944 [+0.9926, +0.9961]","\u2014","+0.441 [+0.301, +0.581]"],
        ],
        [2150,1800,1800,1700,1900]
      ),
      para([{text:"The clean model's loss-quantile analysis is reported as a methodological baseline only: defining memorized samples as the top 25% by loss is circular by construction (high-loss samples are labeled memorized because they are high-loss), yielding a mechanical gap of +0.442 on MNIST and +1.417 on CIFAR-10. The genuine finding is the corrupted model's non-circular result: using ground-truth corruption indices breaks the circularity and reveals a loss gap of \u22120.051 [\u22120.057, \u22120.045] on MNIST and \u22122.363 [\u22122.436, \u22122.291] on CIFAR-10 \u2014 an order of magnitude larger \u2014 confirming that genuinely corrupted samples are harder for the model to fit than clean ones, on both architectures. Gradient alignment between memorized and non-memorized samples at convergence is near-perfect on MNIST (+0.9944) but substantially lower on CIFAR-10 (+0.441), reflecting that the deeper, harder CIFAR-10 model has not converged as completely on its noisy labels (see Section 5.2 for the relationship between this convergence effect and the CKA-based anti-alignment theory)."}]),
      para([{text:"This per-sample loss-gap result is statistically robust, but a separate question is how well a memorization signal derived from hidden activations can identify individual corrupted examples. A linear probe \u2014 logistic regression trained on hidden activations to predict per-sample corruption status \u2014 achieves an AUC of only 0.514 on held-out data, which is close to chance (0.5) and should not be read as strong evidence of detectability at the individual-sample level; conventional thresholds for AUC interpretation place a value this close to 0.5 in the negligible-discrimination range. We return to this result in Section 4.4, where it is placed alongside a causal probe that achieves substantially higher discriminability on the identical question of clean-versus-corrupted status."}]),

      heading("4.4 Phase 4: Causal Interventions via ROME", HeadingLevel.HEADING_2),
      para([{text:"4.4.1 Baseline comparison: correlational vs. causal probes.", bold:true}]),
      para([{text:"To directly compare how well different signals distinguish clean from corrupted models, Table 4 contrasts three probes evaluated on the identical clean-versus-corrupted question: the spectral-norm ratio (Section 4.1), the linear-probe AUC (Section 4.3), and a causal probe based on ROME edit magnitude. For the ROME probe, a rank-one edit \u0394W is computed for every class (not only the four targeted backdoor configurations below) under ordinary clean and corrupted training, and the Frobenius norm \u2016\u0394W\u2016 required to overwrite each class's association is compared between conditions."}]),
      caption("Table 4. Baseline comparison: discriminability of three clean-vs-corrupted probes (MNIST, FC2)."),
      table(
        ["Probe","Discriminability","Basis"],
        [
          ["ROME \u0394-norm ratio (FC2, causal)","4.34\u00d7","Mean \u2016\u0394W\u2016 clean / corrupted, 10 classes \u00d7 10 seeds, p < 0.0001 every class"],
          ["Spectral-norm ratio (FC2, correlational)","1.86\u00d7","Clean / corrupted spectral norm, Table 1"],
          ["Linear probe on hidden activations (correlational)","AUC = 0.514","Logistic regression predicting corruption status; near chance"],
        ],
        [3000,1600,4850]
      ),
      para([{text:"The ROME-based causal probe is substantially more discriminative than either correlational alternative \u2014 more than twice as sensitive as the spectral-norm ratio and far more sensitive than the near-chance linear probe. This ranking helps interpret the weak AUC result from Section 4.3: the hidden activations of an individual corrupted sample do not carry an easily linearly separable signature of corruption, but the weight-space direction associated with a class's memorized association is comparatively large and editable. Per-class results underlying the FC2 ratio are summarized in Table 4b, alongside the corresponding FC1 (MNIST) and FC3 (CIFAR-10) results."}]),
      caption("Table 4b. ROME edit magnitude (\u2016\u0394W\u2016), clean vs. corrupted, averaged across 10 classes."),
      table(
        ["Architecture / Layer","Clean (mean)","Corrupted (mean)","Ratio","Significance"],
        [
          ["MNIST FC1","14.49","7.18","2.02\u00d7","p < 0.0001, all 10 classes"],
          ["MNIST FC2","19.08","4.40","4.34\u00d7","p < 0.0001, all 10 classes"],
          ["CIFAR-10 FC3","0.369 (avg.)","0.202 (avg.)","1.84\u00d7","p < 0.05, all 10 classes"],
        ],
        [2400,1800,1800,1200,2250]
      ),
      para([{text:"Every class on MNIST shows clean > corrupted edit magnitude at p < 0.0001 for both FC1 and FC2, and every class on CIFAR-10 confirms the same direction at p < 0.05 for FC3. The cross-architecture consistency \u2014 ratios of roughly 2\u20134\u00d7 on MNIST and 1.8\u00d7 on CIFAR-10, across different depths, widths, and datasets \u2014 suggests this relationship between class-boundary overlap and edit magnitude is not specific to the minimal two-layer testbed."}]),
      para([{text:"4.4.2 Noise-rate sweep.", bold:true}]),
      para([{text:"To verify that the ROME \u0394-norm signal tracks the degree of memorization continuously, rather than reflecting a binary clean-vs-20%-corrupted artifact, the FC2 \u0394-norm ratio is additionally computed at 10% and 40% label-noise rates (5 seeds each)."}]),
      caption("Table 5. Noise-rate sweep: MNIST FC2 \u0394-norm ratio vs. clean baseline (5 seeds)."),
      table(
        ["Noise Rate","FC2 \u0394-norm","Ratio vs. Clean (19.08)"],
        [
          ["0% (clean)","19.08","1.00\u00d7"],
          ["10%","~5.67","3.37\u00d7"],
          ["20%","~4.40","4.34\u00d7"],
          ["40%","~3.19","5.98\u00d7"],
        ],
        [3000,3000,3250]
      ),
      para([{text:"The ratio increases monotonically with noise rate (3.37\u00d7 \u2192 4.34\u00d7 \u2192 5.98\u00d7 across 10%, 20%, and 40% corruption), confirming that ROME \u0394-norm tracks memorization load continuously rather than detecting a fixed, noise-rate-specific artifact."}]),
      para([{text:"4.4.3 Targeted class-swap recovery.", bold:true}]),
      para([{text:"Table 6 reports per-configuration accuracy recovery for the four Regime B targeted class-swap backdoors."}]),
      caption("Table 6. Phase 4 results: ROME accuracy recovery by class-swap configuration (MNIST, 5 seeds)."),
      table(
        ["Config","Recovery (mean \u00b1 std)","Side Effects","Edit \u2016\u0394W\u2016","Seeds"],
        [
          ["7 \u2192 1","+0.141 \u00b1 0.083","0.171","1.102","5"],
          ["1 \u2192 7","+0.217 \u00b1 0.069","0.240","1.017","5"],
          ["5 \u2192 6","+0.120 \u00b1 0.034","0.246","0.934","5"],
          ["0 \u2192 8","+0.097 \u00b1 0.060","0.216","0.881","5"],
          ["Average","+0.144","0.218","0.984","\u2014"],
        ],
        [1900,2400,1600,1600,1750]
      ),
      para([{text:"Across all four configurations, rank-one edits produce positive accuracy recovery in the source class, ranging from 9.7 percentage points (0\u21928) to 21.7 percentage points (1\u21927), with a mean of 14.4 percentage points. Side effects on non-targeted classes range from 17% to 25% (reported as mean \u00b1 standard deviation across seeds rather than as a confidence interval, since each configuration\u2019s side-effect rate is itself a single aggregate statistic per seed). The substantial variance across configurations \u2014 1\u21927 recovers roughly twice as much as 0\u21928 \u2014 is mechanistically informative: the digit pair 0 and 8 plausibly share more structural overlap with other digits than 1 and 7 do, which may distribute the corresponding association across a larger or less separable subspace of the weight matrix."}]),
      para([{text:"To verify that this recovery is not an artifact of any rank-one perturbation, random rank-one edits of equivalent Frobenius norm are applied as a null baseline. Across all four configurations and five seeds, random edits recover exactly 0% of lost accuracy in every trial (signal ratio = \u221e), since the source class has 0% accuracy before any edit and a random perturbation does not reliably restore a class the model never learned to predict. This confirms that ROME locates a semantically meaningful direction in weight space rather than exploiting an artifact of perturbation magnitude."}]),
      para([{text:"Multi-layer sequential editing (FC2\u2192FC1) does not improve over FC2-only editing: recovery decreases from 10\u201322% (FC2-only) to 5\u201318% (sequential), with FC1-only edits recovering 0% across all configurations (Table 7). This reinforces the CKA finding that memorization concentrates at the output-adjacent layer; editing FC1 after FC2 partially undoes the corrective effect rather than adding to it, indicating that joint (rather than sequential) optimization across layers would be needed for an additive multi-layer intervention."}]),
      caption("Table 7. Multi-layer ROME recovery by layer (5 seeds, mean recovery)."),
      table(
        ["Config","FC2-only","FC1-only","FC2\u2192FC1 (sequential)"],
        [
          ["7\u21921","+0.141","0.000","+0.048"],
          ["1\u21927","+0.217","0.000","+0.182"],
          ["5\u21926","+0.120","0.000","+0.110"],
          ["0\u21928","+0.097","0.000","+0.059"],
        ],
        [2000,2300,2300,2650]
      ),
      para([{text:"4.4.4 Rank-ablation study.", bold:true}]),
      para([{text:"The rank-ablation study complements the targeted interventions above by asking how much of each model's accuracy is concentrated in the top singular directions of its FC2 weight matrix. Table 8 reports test accuracy as the matrix is replaced by successively higher-rank SVD approximations, across the full rank spectrum (1 through 10, the maximum possible rank for a 16-unit hidden layer feeding 10 output classes)."}]),
      caption("Table 8. Rank-ablation study: test accuracy vs. rank of FC2 weight-matrix approximation (10 seeds, 95% CI)."),
      table(
        ["Rank k","Clean Acc. (%)","Corrupted Acc. (%)","Gap (pp)"],
        [
          ["1 / 10","19.0 [17.9, 20.1]","19.3 [16.6, 21.9]","\u22120.3"],
          ["2 / 10","33.9 [30.6, 37.2]","30.0 [26.4, 33.6]","3.9"],
          ["3 / 10","44.9 [40.2, 49.7]","40.5 [37.0, 44.0]","4.4"],
          ["4 / 10","58.3 [53.5, 63.2]","53.2 [49.0, 57.3]","5.1"],
          ["5 / 10","68.4 [62.2, 74.6]","62.3 [57.9, 66.7]","6.1"],
          ["6 / 10","75.4 [67.3, 83.5]","72.6 [68.0, 77.3]","2.8"],
          ["7 / 10","80.1 [70.8, 89.4]","80.7 [76.3, 85.0]","\u22120.6"],
          ["8 / 10","88.9 [85.7, 92.2]","86.4 [84.0, 88.8]","2.5"],
          ["9 / 10","92.8 [90.7, 94.9]","92.3 [91.8, 92.9]","0.5"],
          ["10 / 10 (full)","95.3 [95.2, 95.5]","93.7 [93.4, 94.0]","1.6"],
        ],
        [1700,2350,2350,1850]
      ),
      para([{text:"The full spectrum reveals a more nuanced picture than the endpoint ranks alone suggest. The largest clean-vs-corrupted gap in the entire sweep occurs at rank 5/10, where corrupted models perform 6.1 percentage points worse than clean models (62.3% vs. 68.4%) \u2014 a larger gap than at either rank 8/10 (2.5 pp) or the full rank 10/10 (1.6 pp). This indicates that corrupted models require a broader effective rank spectrum to reach a given accuracy: their memorized label-noise associations are less compressible into the top singular directions than the genuine class structure is, so the corrupted-vs-clean gap is intermediate-rank-specific rather than monotonically shrinking as rank increases. At rank 8/10, both models retain approximately 90% of their respective full-rank accuracy (88.9% of 95.3% for clean, and 86.4% of 93.7% for corrupted), confirming that the bulk of classification-relevant information \u2014 including the memorized associations exploited by the Phase 4 backdoor edits \u2014 is concentrated in the top eight of ten singular directions, even though the rank-5 result shows that the remaining two directions disproportionately carry the most corruption-discriminating signal."}]),

      heading("4.5 Width-Scaling Analysis", HeadingLevel.HEADING_2),
      para([{text:"Table 9 shows how key metrics change with network width from 16 to 1,024 hidden units, trained for 100 epochs across ten seeds (clean training only)."}]),
      caption("Table 9. Scaling analysis: Fisher discriminant ratio (FDR), monosemanticity fraction, circuit size, sparsity, and test accuracy vs. hidden width (10 seeds)."),
      table(
        ["Width","FDR","Monosem. (%)","Circuit Size","Sparsity","Test Acc. (%)"],
        [
          ["16","0.862","26.9","6.6","0.588","95.31"],
          ["32","0.654","22.2","8.3","0.740","96.85"],
          ["64","0.560","21.2","3.5","0.945","97.50"],
          ["128","0.470","18.4","0.4","0.997","97.85"],
          ["256","0.456","17.5","0.1","1.000","97.93"],
          ["512","0.423","15.4","0.0","1.000","98.17"],
          ["1024","0.387","7.5","0.0","1.000","98.11"],
        ],
        [1300,1300,1900,1700,1500,2000]
      ),
      para([{text:"FDR decreases monotonically with width, from 0.862 at 16 units to 0.387 at 1,024 units, while a naive within/between distance ratio (\u03c3) moves in the opposite direction (0.687 to 0.823 over the same range) \u2014 the divergence between the two metrics confirms that \u03c3 conflates geometric spread with dimensionality, and that FDR's dimension-invariance is necessary to see the genuine separability decrease. Clean-model CKA at the ReLU nonlinearity remains stable across all widths tested (0.83\u20130.90), confirming that the representational-geometry effect described in Section 4.2 is specific to the clean-versus-corrupted comparison rather than to width itself. The monosemanticity fraction decreases sharply with width (26.9% at 16 units to 7.5% at 1,024 units), and circuit size \u2014 the number of hidden units whose ablation substantially changes a given class's output, computed against a fixed ablation threshold specified in the accompanying configuration file \u2014 shrinks toward zero beyond approximately 256 units. A circuit size of exactly 0.0 at the largest widths should be read as falling below this threshold rather than as literally zero functional circuitry; readers interested in the precise threshold value should consult the released configuration. Together, these results suggest that as network capacity increases, memorized associations are spread over an increasingly larger set of directions, making them harder to isolate and surgically edit, consistent with theoretical intuitions about superposition in overparameterized networks."}]),

      heading("5. Discussion", HeadingLevel.HEADING_1),
      heading("5.1 Convergent Evidence for Low-Dimensional Memorization", HeadingLevel.HEADING_2),
      para([{text:"The four phases of this analysis converge on a consistent picture, now strengthened by cross-architecture replication. Correlational evidence (Phases 1\u20133) demonstrates that label memorization is associated with altered weight norms (in a depth-dependent direction), a substantially altered post-activation representation geometry that localizes to whichever interface sits deepest in a given architecture, and a statistically distinguishable population of high-loss training examples on both MNIST and CIFAR-10. Causal evidence (Phase 4) demonstrates both that targeted edits to a small number of weight directions can selectively repair behavior associated with a specific memorized class association, and \u2014 via the baseline comparison in Table 4 \u2014 that this causal signal is substantially more discriminative of clean-versus-corrupted status than either of the correlational alternatives tested. Taken together, these results support the view that, in the small networks studied here, memorized label associations are encoded in a compact, identifiable, and partially editable subspace of the weight matrices, distinct from, though overlapping with, the subspace responsible for genuine class structure."}]),
      heading("5.2 Mechanistic Interpretation of the Depth-Dependent Distortion", HeadingLevel.HEADING_2),
      para([{text:"The sharp drop in CKA at the deepest pre-output interface (Section 4.2) is the most mechanistically specific correlational finding of this study, but it is also the one most exposed to the reliability concerns raised by Davari et al., who show CKA values can be manipulated without corresponding functional change [8]. Three considerations support reading the present result as genuine rather than artifactual. First, the effect replicates across two structurally different architectures while shifting location with depth \u2014 the single ReLU on MNIST, the final linear layer on CIFAR-10 \u2014 a pattern that would be a striking coincidence for a manipulable numerical artifact to reproduce. Second, the same localization claim (memorization concentrates at the output-adjacent layer) is independently corroborated by a method that does not use CKA at all: the Phase 4 multi-layer ROME results (Table 7) show FC1-only edits recover 0% of lost accuracy across every configuration, while FC2-only and FC2\u2192FC1 edits recover substantially more, agreeing with the representational finding through an entirely separate causal mechanism. Third, the theoretical account of Nguyen et al. connects label-noise memorization to a degradation of neural collapse \u2014 the late-training convergence of within-class representations \u2014 providing an independent reason to expect a genuine representational signature of corruption, rather than only a weight-norm one [9]. None of this rules out residual sensitivity of the CKA measurement to the specific transformations Davari et al. study, and a direct robustness check using an alternative similarity metric (such as SVCCA or orthogonal Procrustes analysis) remains a natural next step, identified explicitly as future work in Section 5.4 rather than claimed here."}]),
      para([{text:"The higher activation sparsity in corrupted MNIST models (0.357 to 0.576) indicates that a larger fraction of hidden units are silent on each input, which may be a signature of more example-specific, less generalizable representations \u2014 consistent with a picture in which a subset of hidden units act as detectors for specific, possibly mislabeled, training examples. This hypothesis is broadly supported by the Phase 3 monosemanticity results, the Phase 3 circuit-size measurements, and the Phase 4 rank-ablation results, though it remains a hypothesis rather than a directly verified mechanism."}]),
      heading("5.3 Limitations", HeadingLevel.HEADING_2),
      para([{text:"Several limitations bear on the interpretation of these results. First, while the central claims are now validated on a second, deeper architecture (CIFAR-10), both architectures tested are simple feedforward networks; convolutional or attention-based architectures, more representative of contemporary practice, remain untested and may exhibit qualitatively different signatures. Second, the CIFAR-10 validation uses five seeds rather than the ten used for MNIST, owing to higher per-run training cost; confidence intervals for the CIFAR-10 results are correspondingly wider. Third, most analyses use a fixed 20% corruption rate; the noise-rate sweep in Section 4.4.2 partially addresses how the ROME signal specifically scales with corruption rate (10\u201340%), but the remaining phases (weight geometry, CKA, rank-ablation, scaling) have not been repeated across multiple noise rates. Fourth, ROME recovery, while consistently positive, is partial (9.7 to 21.7 percentage points) and accompanied by non-trivial side effects on non-targeted classes (17% to 25%), indicating that memorized associations are not perfectly disentangled from genuine class representations even within their low-rank subspace. Fifth, the CKA-based mechanistic claim, while now supported by cross-architecture replication and independent ROME corroboration (Section 5.2), has not been checked against an alternative representation-similarity metric; this is a concrete and tractable next step rather than a settled question. Finally, the corrupted model's training accuracy of 94.21% (not 100%) on MNIST reflects the 16-unit bottleneck rather than a general property of memorization: the model lacks the capacity to memorize all 12,000 corrupted samples, prioritizing the 80% clean labels instead."}]),
      heading("5.4 Future Directions", HeadingLevel.HEADING_2),
      para([{text:"Future work could extend this pipeline along several axes: (1) repeating the weight-geometry, CKA, and rank-ablation analyses across the full 10\u201340% noise-rate range already validated for the ROME probe, to test whether the depth-dependent and rank-5 findings also scale continuously with corruption load; (2) checking the CKA-collapse finding against an alternative similarity metric, such as SVCCA or orthogonal Procrustes analysis, as a direct robustness check against the manipulability concerns raised in Section 5.2; (3) applying the same diagnostics to convolutional or attention-based architectures, where memorization is known to be practically important but considerably harder to localize; and (4) exploring whether the identified low-rank subspace of memorization can be exploited for more targeted unlearning or privacy-preserving data removal without degrading overall performance."}]),

      heading("6. Conclusion", HeadingLevel.HEADING_1),
      para([{text:"This study presented a four-phase empirical analysis of label memorization in a shallow two-layer neural network trained on MNIST, validated on a deeper three-layer network trained on CIFAR-10, integrating weight-geometry analysis, representation similarity via CKA, influence-based memorization scoring, and causal intervention via rank-one model editing. Label memorization measurably alters weight geometry in both architectures, with the effect concentrating at output-adjacent layers; sharply alters the representation transformation at the deepest pre-output interface of each architecture (CKA: 0.850 to 0.690 on MNIST, p < 0.001; 0.580 to 0.477 on CIFAR-10, p = 0.009); produces a statistically distinguishable population of high-loss training examples on both datasets; and can be partially and selectively reversed through low-rank weight edits. A direct baseline comparison shows that this causal ROME-based signal (4.34\u00d7 discriminability on MNIST, 1.84\u00d7 on CIFAR-10) substantially outperforms both a spectral-norm ratio (1.86\u00d7) and a linear probe on hidden activations (AUC = 0.514, near chance), and scales monotonically with corruption rate. A rank-eight weight approximation recovers approximately 90% of full network performance, though the full rank-ablation spectrum shows the largest clean-vs-corrupted gap occurs at an intermediate rank (5 of 10), indicating that corrupted models distribute their associations less compactly than the endpoint ranks alone would suggest. A width-scaling analysis further shows that class separability decreases as networks grow wider, with FDR falling from 0.862 at 16 units to 0.387 at 1,024 units."}]),
      para([{text:"Together, these results provide converging correlational and causal evidence, now replicated across two architectures, that memorization leaves a consistent, low-dimensional, and editable structural fingerprint that grows more diffuse as network capacity increases. The four-phase pipeline introduced here \u2014 weight geometry, representation similarity, influence scoring, and causal editing \u2014 may serve as a reusable diagnostic toolkit for probing memorization in other architectures and data regimes, contributing to the broader goal of making neural network behavior interpretable, auditable, and correctable."}]),

      heading("Data and Code Availability", HeadingLevel.HEADING_1),
      para([{text:"All code, configuration files, trained checkpoints, and figure-generation scripts supporting this study will be made publicly available upon publication. In accordance with this journal's double-blind review policy, the repository link is withheld from this manuscript and will be provided to editors upon request, or released at the point of de-anonymization."}]),

      heading("References", HeadingLevel.HEADING_1),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Zhang C, Bengio S, Hardt M, et al. Understanding deep learning requires rethinking generalization. In: Proc Int Conf Learn Represent (ICLR); 2017."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Kornblith S, Norouzi M, Lee H, Hinton G. Similarity of neural network representations revisited. In: Proc 36th Int Conf Mach Learn (ICML); 2019. p. 3519\u20133529."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Koh PW, Liang P. Understanding black-box predictions via influence functions. In: Proc 34th Int Conf Mach Learn (ICML); 2017. p. 1885\u20131894."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Meng K, Bau D, Andonian A, Belinkov Y. Locating and editing factual associations in GPT. In: Adv Neural Inf Process Syst (NeurIPS); 2022. vol. 35, p. 17359\u201317372."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Feldman V. Does learning require memorization? A short tale about a long tail. In: Proc 52nd Annu ACM SIGACT Symp Theory Comput (STOC); 2020. p. 954\u2013959."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"LeCun Y, Cortes C, Burges CJC. MNIST handwritten digit database; 1998. Available from: http://yann.lecun.com/exdb/mnist/"}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Kingma DP, Ba J. Adam: A method for stochastic optimization. In: Proc Int Conf Learn Represent (ICLR); 2015."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Davari MR, Horoi S, Natik A, et al. Reliability of CKA as a similarity measure in deep learning. In: Proc Int Conf Learn Represent (ICLR); 2023."}]) }),
      new Paragraph({ numbering: { reference: "refList", level: 0 }, spacing: { after: 100 }, children: runs([{text:"Nguyen DA, Levie R, Lienen J, et al. Memorization-dilation: Modeling neural collapse under label noise. In: Proc Int Conf Learn Represent (ICLR); 2023."}]) }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/paper/Structural_Fingerprints_Memorization_SURJ_REVISED.docx", buffer);
  console.log("done");
});