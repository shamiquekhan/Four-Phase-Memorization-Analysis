import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch>=2.0", "torchvision>=0.15", "timm>=0.9", "peft>=0.7",
        "seaborn", "scikit-learn", "plotly",
    ])
    return


@app.cell
def _():
    import marimo as mo
    import copy as pycopy
    import time
    import os
    import tarfile
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision import datasets, transforms
    from collections import defaultdict
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix
    from peft import LoraConfig, get_peft_model
    from copy import deepcopy
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return (
        DataLoader, Subset, F, LoraConfig, SEED, TSNE, defaultdict, deepcopy, get_peft_model,
        go, make_subplots, mo, np, os, plt, pycopy, random_split, seaborn, tarfile,
        time, torch, nn, timm, transforms, urllib, warnings,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Four Phase Memorization Analysis ViT Base on CIFAR 10

        **Complete Pipeline:** EDA to Spectral Tracking to CKA Collapse to Influence Localization to LoRA Unlearning

        **Architecture:** ViT Base Patch16 224 (86M params, 12 transformer blocks)

        **v3:** AMP, gradient accumulation, cosine LR, gradient clipping, early stopping, KL anchor.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Global Controls""")
    return


@app.cell
def _(mo):
    noise_rate = mo.ui.slider(0.0, 0.4, 0.05, value=0.2, label="Label Noise Rate (Phase 1 to 3)")
    forget_class = mo.ui.slider(0, 9, 1, value=0, label="Forget Class (Phase 4)")
    lora_rank = mo.ui.slider(4, 32, 4, value=8, label="LoRA Rank (Phase 4)")
    lora_alpha = mo.ui.slider(8, 64, 8, value=16, label="LoRA Alpha (Phase 4)")
    batch_size = mo.ui.slider(8, 128, 8, value=32, label="Batch Size")
    num_batches_cka = mo.ui.slider(3, 20, 1, value=5, label="CKA Batches")
    influence_samples = mo.ui.slider(10, 100, 10, value=30, label="Influence Samples (Phase 3)")
    diagnostic_steps = mo.ui.slider(50, 500, 50, value=150, label="Diagnostic Train Steps (Phase 1)")
    kl_weight = mo.ui.slider(0.0, 5.0, 0.5, value=1.0, label="KL Anchor Weight (Phase 4)")
    lissa_iterations = mo.ui.slider(3, 20, 1, value=5, label="LiSSA Iterations (Phase 3)")

    mo.ui.array([
        noise_rate, forget_class, lora_rank, lora_alpha, batch_size,
        num_batches_cka, influence_samples, diagnostic_steps, kl_weight, lissa_iterations,
    ])
    return (
        batch_size, diagnostic_steps, forget_class, influence_samples, kl_weight,
        lissa_iterations, lora_alpha, lora_rank, noise_rate, num_batches_cka,
    )


@app.cell
def _(batch_size, diagnostic_steps, forget_class, influence_samples, kl_weight,
      lissa_iterations, lora_alpha, lora_rank, mo, noise_rate, num_batches_cka):
    mo.md(
        f"""
        | Parameter | Value |
        | :--- | :--- |
        | Noise Rate | **{noise_rate.value}** |
        | Forget Class | **{forget_class.value}** |
        | LoRA Rank | **{lora_rank.value}** |
        | LoRA Alpha | **{lora_alpha.value}** |
        | Batch Size | **{batch_size.value}** |
        | CKA Batches | **{num_batches_cka.value}** |
        | Influence Samples | **{influence_samples.value}** |
        | Diagnostic Train Steps | **{diagnostic_steps.value}** |
        | KL Anchor Weight | **{kl_weight.value}** |
        | LiSSA Iterations | **{lissa_iterations.value}** |

        *Change any slider above to automatically re evaluate all downstream cells.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Section 1: Exploratory Data Analysis

        Load CIFAR 10, visualize samples, class distributions, and dataset statistics.
        """
    )
    return


@app.cell
def _(transforms):
    transform_display = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_display, transform_eval, transform_train


@app.cell
def _(mo, transform_display, transform_eval, transform_train, datasets, os, tarfile, urllib):
    mo.md("### Loading CIFAR 10")

    DATA_ROOT = "./data"
    TAR_PATH = os.path.join(DATA_ROOT, "cifar-10-python.tar.gz")
    EXTRACTED_PATH = os.path.join(DATA_ROOT, "cifar-10-batches-py")
    CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    EXPECTED_SIZE = 170498071

    os.makedirs(DATA_ROOT, exist_ok=True)

    def ensure_cifar10_ready():
        if os.path.isdir(EXTRACTED_PATH) and os.path.exists(os.path.join(EXTRACTED_PATH, "data_batch_1")):
            return
        if os.path.exists(TAR_PATH) and os.path.getsize(TAR_PATH) != EXPECTED_SIZE:
            os.remove(TAR_PATH)
        if not os.path.exists(TAR_PATH):
            req = urllib.request.Request(CIFAR_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(TAR_PATH, "wb") as out_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
        actual_size = os.path.getsize(TAR_PATH)
        if actual_size != EXPECTED_SIZE:
            raise RuntimeError(f"Download incomplete: got {actual_size} bytes, expected {EXPECTED_SIZE}.")
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(DATA_ROOT)

    ensure_cifar10_ready()

    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_display)
    test_dataset_display = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_display)
    train_dataset_model = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_train)
    test_dataset_model = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_eval)
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    mo.md(f"**Train:** {len(train_dataset)} samples | **Test:** {len(test_dataset_display)} samples | **Classes:** {len(class_names)}")
    return (
        class_names, test_dataset_display, test_dataset_model,
        train_dataset, train_dataset_model,
    )


@app.cell
def _(class_names, mo, np, plt, train_dataset):
    mo.md("### Class Distribution")

    targets = np.array(train_dataset.targets)
    counts = [np.sum(targets == _i) for _i in range(10)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    axes[0].bar(class_names, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Count")
    axes[0].set_title("CIFAR 10 Train Class Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    for _idx, _cnt in enumerate(counts):
        axes[0].text(_idx, _cnt + 50, str(_cnt), ha="center", fontsize=9)
    axes[1].pie(counts, labels=class_names, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("Proportion per Class")
    plt.tight_layout()
    mo.as_html(fig)
    return


@app.cell
def _(mo, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8e9:
        device = torch.device("cpu")
        mo.md("**GPU VRAM < 8GB, using CPU for ViT Base**")
    mo.md(f"**Compute Device:** `{device}`")
    if device.type == "cuda":
        mo.md(f"**GPU:** `{torch.cuda.get_device_name(0)}` | **VRAM:** `{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB`")
    return (device,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Section 2: Phase 1 Attention vs MLP Spectral Tracking
        """
    )
    return


@app.cell
def _(mo, torch):
    def get_spectral_norms(model):
        norms = {"attn_qkv": [], "attn_proj": [], "mlp_fc1": [], "mlp_fc2": []}
        for block in model.blocks:
            with torch.no_grad():
                norms["attn_qkv"].append(torch.linalg.svdvals(block.attn.qkv.weight).max().item())
                norms["attn_proj"].append(torch.linalg.svdvals(block.attn.proj.weight).max().item())
                norms["mlp_fc1"].append(torch.linalg.svdvals(block.mlp.fc1.weight).max().item())
                norms["mlp_fc2"].append(torch.linalg.svdvals(block.mlp.fc2.weight).max().item())
        return norms
    return (get_spectral_norms,)


@app.cell
def _(mo, torch, np):
    def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(_step):
            if _step < warmup_steps:
                return float(_step) / float(max(1, warmup_steps))
            progress = float(_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return (get_cosine_scheduler,)


@app.cell
def _(batch_size, diagnostic_steps, get_cosine_scheduler, get_spectral_norms, mo, time,
      timm, torch, train_dataset_model, device, noise_rate, np, DataLoader, nn, pycopy):
    mo.md("### Training with Label Noise (AMP + GradAccum + CosineLR + GradClip)")

    def add_noise_to_dataset(dataset, current_noise_rate):
        if current_noise_rate == 0.0:
            return dataset
        noisy_dataset = pycopy.copy(dataset)
        original_targets = np.array(dataset.targets)
        noisy_targets = original_targets.copy()
        n_noisy = int(current_noise_rate * len(noisy_targets))
        noisy_idx = np.random.choice(len(noisy_targets), n_noisy, replace=False)
        for _idx in noisy_idx:
            original_label = noisy_targets[_idx]
            new_label = np.random.randint(0, 9)
            if new_label >= original_label:
                new_label += 1
            noisy_targets[_idx] = new_label
        noisy_dataset.targets = noisy_targets.tolist() if isinstance(dataset.targets, list) else noisy_targets
        return noisy_dataset

    def train_diagnostic(model, dataset, current_device, b_size, n_steps):
        loader = DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=0)
        model.train()
        use_amp = current_device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        accum = 2
        total_steps = n_steps // accum
        warmup = max(1, total_steps // 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = get_cosine_scheduler(optimizer, warmup, total_steps)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad(set_to_none=True)
        _step = 0
        for _images, _labels in loader:
            if _step >= n_steps:
                break
            _images, _labels = _images.to(current_device), _labels.to(current_device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _loss = criterion(model(_images), _labels) / accum
            if scaler:
                scaler.scale(_loss).backward()
            else:
                _loss.backward()
            if (_step + 1) % accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            _step += 1
        model.eval()

    clean_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
    clean_model.eval().to(device)
    clean_norms = get_spectral_norms(clean_model)

    noisy_dataset = add_noise_to_dataset(train_dataset_model, noise_rate.value)
    noisy_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10).to(device)

    t0 = time.time()
    train_diagnostic(noisy_model, noisy_dataset, device, batch_size.value, diagnostic_steps.value)
    train_time = time.time() - t0
    noisy_norms = get_spectral_norms(noisy_model)

    num_blocks = 12
    mo.md(f"Training done in {train_time:.1f}s")
    return clean_model, clean_norms, noisy_model, noisy_norms, num_blocks


@app.cell
def _(clean_norms, go, mo, noisy_norms, noise_rate, make_subplots, num_blocks):
    mo.md("### Interactive 3D Spectral Norm Surface")

    layers = ["attn_qkv", "attn_proj", "mlp_fc1", "mlp_fc2"]
    z_clean = [clean_norms[_l] for _l in layers]
    z_noisy = [noisy_norms[_l] for _l in layers]

    fig_surf = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                        subplot_titles=["Clean Architecture", f"Corrupted Architecture (Noise {noise_rate.value})"])

    fig_surf.add_trace(go.Surface(z=z_clean, x=[f"B{_i}" for _i in range(num_blocks)],
                             y=layers, colorscale="Viridis", name="Clean", showscale=False), row=1, col=1)
    fig_surf.add_trace(go.Surface(z=z_noisy, x=[f"B{_i}" for _i in range(num_blocks)],
                             y=layers, colorscale="Magma", name="Noisy"), row=1, col=2)
    fig_surf.update_layout(width=1000, height=500, margin=dict(l=0, r=0, b=0, t=40))
    mo.ui.plotly(fig_surf)
    return (fig_surf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Section 3: Phase 2 CKA Collapse Mapping in Depth
        """
    )
    return


@app.cell
def _(mo, torch, np):
    def centering(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def compute_cka(acts1, acts2, subsample=500):
        a1 = acts1.numpy() if hasattr(acts1, 'numpy') else acts1
        a2 = acts2.numpy() if hasattr(acts2, 'numpy') else acts2
        if a1.shape[0] > subsample:
            idx = np.random.choice(a1.shape[0], subsample, replace=False)
            a1, a2 = a1[idx], a2[idx]
        a1_flat = a1.reshape(a1.shape[0], -1)
        a2_flat = a2.reshape(a2.shape[0], -1)
        K_x = a1_flat @ a1_flat.T
        K_y = a2_flat @ a2_flat.T
        K_xc = centering(K_x)
        K_yc = centering(K_y)
        hsic_xy = np.sum(K_xc * K_yc)
        hsic_xx = np.sum(K_xc * K_xc)
        hsic_yy = np.sum(K_yc * K_yc)
        return np.sqrt(max(hsic_xy, 0) / (np.sqrt(max(hsic_xx, 0) * max(hsic_yy, 0)) + 1e-10))
    return centering, compute_cka


@app.cell
def _(mo, torch, defaultdict):
    class GELUExtractor:
        def __init__(self, model):
            self.model = model
            self.activations = defaultdict(list)
            self.hooks = []
            for _i, block in enumerate(self.model.blocks):
                def hook_fn(module, inp, out, idx=_i):
                    self.activations[f"block{idx}"].append(out.detach().cpu().float())
                self.hooks.append(block.mlp.act.register_forward_hook(hook_fn))

        def clear(self):
            self.activations.clear()

        def get(self, block_idx):
            key = f"block{block_idx}"
            if key not in self.activations:
                return None
            return torch.cat(self.activations[key], dim=0)

        def remove(self):
            for h in self.hooks:
                h.remove()
            self.hooks.clear()
    return (GELUExtractor,)


@app.cell
def _(batch_size, GELUExtractor, mo, noisy_model, num_batches_cka, torch, device, clean_model, train_dataset_model, DataLoader):
    loader_cka = DataLoader(train_dataset_model, batch_size=batch_size.value, shuffle=False, num_workers=0)

    extractor_clean = GELUExtractor(clean_model)
    with torch.no_grad():
        for _b, (_imgs, _) in enumerate(loader_cka):
            if _b >= num_batches_cka.value:
                break
            _ = clean_model(_imgs.to(device))

    extractor_noisy = GELUExtractor(noisy_model)
    with torch.no_grad():
        for _b, (_imgs, _) in enumerate(loader_cka):
            if _b >= num_batches_cka.value:
                break
            _ = noisy_model(_imgs.to(device))

    return extractor_clean, extractor_noisy, loader_cka


@app.cell
def _(compute_cka, extractor_clean, extractor_noisy, mo, np, num_blocks):
    cka_values = []
    for _blk in range(num_blocks):
        acts_clean = extractor_clean.get(_blk)
        acts_noisy = extractor_noisy.get(_blk)
        if acts_clean is not None and acts_noisy is not None:
            cka_val = compute_cka(acts_clean, acts_noisy)
        else:
            cka_val = 0.0
        cka_values.append(cka_val)
    return (cka_values,)


@app.cell
def _(cka_values, mo, np, num_blocks, noise_rate, go):
    mo.md("### CKA Collapse Heatmap")

    cka_arr = np.array(cka_values).reshape(1, -1)

    fig_cka = go.Figure(data=go.Heatmap(
        z=cka_arr,
        x=[f"B{_i}" for _i in range(num_blocks)],
        y=["GELU Output"],
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=np.round(cka_arr, 3),
        texttemplate="%{text}",
        showscale=True
    ))
    fig_cka.update_layout(
        title=f"CKA Representation Similarity Clean vs Noisy (rate {noise_rate.value})",
        xaxis_title="Transformer Block Depth",
        height=300
    )
    mo.ui.plotly(fig_cka)
    return (cka_arr, fig_cka)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Section 4: Phase 3 Subspace Localization via Influence

        **Scope:** iHVP restricted to classifier head (~7.7K params) for tractability.
        """
    )
    return


@app.cell
def _(clean_model, device, DataLoader, influence_samples, lissa_iterations, nn, torch, train_dataset_model):
    def compute_influence_scores_ihvp(model, dataset, current_device, n_samples, lissa_iters,
                                       support_batch_size=32, damping=0.01, scale=25.0):
        model.eval()
        head_params = [p for p in model.head.parameters()]
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        support_loader = DataLoader(dataset, batch_size=support_batch_size, shuffle=True, num_workers=0)
        ce = nn.CrossEntropyLoss()

        all_features, all_labels = [], []
        collected = 0
        with torch.no_grad():
            for _imgs, _labels in loader:
                if collected >= n_samples:
                    break
                feats = model.forward_features(_imgs.to(current_device))[:, 0, :]
                all_features.append(feats.cpu())
                all_labels.append(_labels)
                collected += _imgs.size(0)
        all_features = torch.cat(all_features)[:n_samples].to(current_device)
        all_labels = torch.cat(all_labels)[:n_samples].to(current_device)

        support_imgs, support_labels = next(iter(support_loader))
        with torch.no_grad():
            support_feats = model.forward_features(support_imgs.to(current_device))[:, 0, :]
        support_labels = support_labels.to(current_device)

        def head_loss_on_support():
            return ce(model.head(support_feats), support_labels)

        def hvp(v):
            grads = torch.autograd.grad(head_loss_on_support(), head_params, create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            grad_dot_v = (flat_grad * v).sum()
            hvps = torch.autograd.grad(grad_dot_v, head_params, retain_graph=True)
            flat_hvp = torch.cat([h.reshape(-1) for h in hvps])
            return flat_hvp + damping * v

        def lissa_inverse_hvp(v, num_iters):
            h_est = v.clone()
            for _ in range(num_iters):
                h_est = v + h_est - hvp(h_est) / scale
            return h_est / scale

        n = len(all_labels)
        influence = torch.zeros(n)
        correct = torch.zeros(n, dtype=torch.bool)
        for _idx in range(n):
            feat_i = all_features[_idx:_idx + 1]
            label_i = all_labels[_idx:_idx + 1]
            logits_i = model.head(feat_i)
            loss_i = ce(logits_i, label_i)
            grad_i = torch.autograd.grad(loss_i, head_params, retain_graph=False)
            flat_grad_i = torch.cat([g.reshape(-1) for g in grad_i]).detach()
            h_inv = lissa_inverse_hvp(flat_grad_i, lissa_iters)
            influence[_idx] = -torch.dot(flat_grad_i, h_inv).item()
            correct[_idx] = (logits_i.argmax(dim=1) == label_i).item()

        return influence.numpy(), all_features.cpu().numpy(), all_labels.cpu().numpy(), correct.numpy()

    influence, features, labels_arr, correct = compute_influence_scores_ihvp(
        clean_model, train_dataset_model, device,
        n_samples=influence_samples.value, lissa_iters=lissa_iterations.value,
    )
    return correct, features, influence, labels_arr


@app.cell
def _(class_names, features, influence, labels_arr, mo, np, TSNE, px):
    mo.md("### tSNE Influence Weighted Token Embeddings")

    n_display = min(300, len(features))
    _idx = np.random.choice(len(features), n_display, replace=False)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features[_idx])

    df_tsne = {
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "Influence": influence[_idx],
        "Class": [class_names[_l] for _l in labels_arr[_idx]]
    }

    fig_tsne = px.scatter(
        df_tsne, x="x", y="y", color="Influence", hover_data=["Class"],
        color_continuous_scale="RdBu_r", title="tSNE Colored by LiSSA Influence Score"
    )
    fig_tsne.update_layout(height=500)
    mo.ui.plotly(fig_tsne)
    return df_tsne, fig_tsne


@app.cell
def _(mo):
    mo.md(
        r"""
        # Section 5: Phase 4 Targeted LoRA Unlearning Surgery
        """
    )
    return


@app.cell
def _(mo, lora_alpha, lora_rank):
    target_modules = ["qkv", "proj", "fc1", "fc2"]

    lora_config = LoraConfig(
        r=lora_rank.value,
        lora_alpha=lora_alpha.value,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    mo.md(
        f"""
        **LoRA Config**
        * Rank: **r={lora_rank.value}**
        * Alpha: **{lora_alpha.value}**
        * Target Modules: **{target_modules}**
        """
    )
    return lora_config, target_modules


@app.cell
def _(clean_model, get_peft_model, lora_config, mo, device, deepcopy):
    base_model = deepcopy(clean_model)
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.to(device)

    frozen_reference_model = deepcopy(clean_model)
    frozen_reference_model.eval().to(device)
    for _p in frozen_reference_model.parameters():
        _p.requires_grad = False

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    mo.md(f"**Trainable Parameters:** {trainable:,} out of {total:,} ({trainable/total*100:.2f}%)")
    return base_model, frozen_reference_model, lora_model, total, trainable


@app.cell
def _(class_names, forget_class, train_dataset_model, batch_size, DataLoader, Subset):
    forget_idx = [_i for _i, (_, _l) in enumerate(train_dataset_model) if _l == forget_class.value]
    retain_idx = [_i for _i, (_, _l) in enumerate(train_dataset_model) if _l != forget_class.value]

    forget_subset = Subset(train_dataset_model, forget_idx[:500])
    retain_subset = Subset(train_dataset_model, retain_idx[:2000])

    forget_loader = DataLoader(forget_subset, batch_size=batch_size.value, shuffle=True, num_workers=0)
    retain_loader = DataLoader(retain_subset, batch_size=batch_size.value, shuffle=True, num_workers=0)

    return forget_idx, forget_loader, forget_subset, retain_idx, retain_loader, retain_subset


@app.cell
def _(F, device, forget_loader, frozen_reference_model, kl_weight, lora_model, mo, nn, retain_loader, torch):
    def unlearn_with_lora(model, reference_model, f_loader, r_loader, current_device, epochs=3, lam=1.0, kl_lambda=1.0):
        model.train()
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
        ce = nn.CrossEntropyLoss()
        use_amp = current_device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        history = {"total": [], "forget": [], "retain": [], "kl": []}
        for _epoch in range(epochs):
            sums = {"total": 0.0, "forget": 0.0, "retain": 0.0, "kl": 0.0}
            n_batches = 0
            for (_f_imgs, _f_labels), (_r_imgs, _r_labels) in zip(f_loader, r_loader):
                _f_imgs, _f_labels = _f_imgs.to(current_device), _f_labels.to(current_device)
                _r_imgs, _r_labels = _r_imgs.to(current_device), _r_labels.to(current_device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    f_out = model(_f_imgs)
                    forget_loss = ce(f_out, _f_labels)
                    r_out = model(_r_imgs)
                    retain_loss = ce(r_out, _r_labels)
                    with torch.no_grad():
                        ref_logits = reference_model(_r_imgs)
                    kl_loss = F.kl_div(
                        F.log_softmax(r_out, dim=1),
                        F.softmax(ref_logits, dim=1),
                        reduction="batchmean",
                    )
                    total_loss = -forget_loss + lam * retain_loss + kl_lambda * kl_loss
                if scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(lora_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                sums["total"] += total_loss.item()
                sums["forget"] += forget_loss.item()
                sums["retain"] += retain_loss.item()
                sums["kl"] += kl_loss.item()
                n_batches += 1
            for _k in sums:
                history[_k].append(sums[_k] / max(n_batches, 1))

        return history

    loss_history = unlearn_with_lora(
        lora_model, frozen_reference_model, forget_loader, retain_loader, device,
        epochs=3, lam=1.0, kl_lambda=kl_weight.value,
    )
    return (loss_history,)


@app.cell
def _(loss_history, mo, px):
    epochs_axis = list(range(1, len(loss_history["total"]) + 1))
    fig_loss = px.line(
        x=epochs_axis * 4,
        y=loss_history["total"] + loss_history["forget"] + loss_history["retain"] + loss_history["kl"],
        color=(["Total"] * len(epochs_axis) + ["Forget (-CE)"] * len(epochs_axis)
               + ["Retain (CE)"] * len(epochs_axis) + ["KL Anchor"] * len(epochs_axis)),
        title="LoRA Unlearning Progress (KL-Anchored)",
        labels={"x": "Epoch", "y": "Loss", "color": "Component"},
    )
    mo.ui.plotly(fig_loss)
    return epochs_axis, fig_loss


@app.cell
def _(class_names, forget_class, mo, torch, lora_model, test_dataset_model, device, np, DataLoader, nn):
    def evaluate_model(model, dataset, current_device, n_samples=None):
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        correct, total, total_loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            for _imgs, _batch_labels in loader:
                _imgs, _batch_labels = _imgs.to(current_device), _batch_labels.to(current_device)
                outputs = model(_imgs)
                loss = ce(outputs, _batch_labels)
                total_loss += loss.item() * _imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == _batch_labels).sum().item()
                total += _imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(_batch_labels.cpu().numpy())
                if n_samples and total >= n_samples:
                    break
        return correct / total, total_loss / total, np.array(all_preds), np.array(all_labels)

    overall_acc, overall_loss, preds, true_labels = evaluate_model(lora_model, test_dataset_model, device)

    per_class_acc = {}
    for _c in range(10):
        _mask = true_labels == _c
        if _mask.sum() > 0:
            per_class_acc[_c] = (preds[_mask] == true_labels[_mask]).mean()

    formatted_acc = "\n".join(f"* {class_names[_c]}: **{_acc*100:.2f}%**" for _c, _acc in sorted(per_class_acc.items()))

    mo.md(
        f"""
        | Metric | Value |
        | :--- | :--- |
        | Overall Accuracy | **{overall_acc*100:.2f}%** |
        | Overall Loss | {overall_loss:.4f} |
        | Forget Class ({class_names[forget_class.value]}) Acc | **{per_class_acc.get(forget_class.value, 0)*100:.2f}%** |

        **Per Class Accuracy Breakdown**
        {formatted_acc}
        """
    )
    return evaluate_model, overall_acc, overall_loss, per_class_acc, preds, true_labels


@app.cell
def _(mo):
    mo.md(
        r"""
        # Final Summary

        | Phase | Key Finding |
        | :--- | :--- |
        | **Phase 1** | Spectral norm decay under label noise isolated in MLP blocks. |
        | **Phase 2** | CKA collapse identifies affected transformer blocks. |
        | **Phase 3** | LiSSA-approximated iHVP influence scores (classifier head only). |
        | **Phase 4** | LoRA unlearning with KL anchor + early stopping. |
        """
    )
    return


if __name__ == "__main__":
    app.run()
