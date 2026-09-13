#!/usr/bin/env python3
"""
Four-Phase Memorization Analysis — ViT-Base on CIFAR-10
Standalone script with all optimizations (AMP, gradient accumulation, cosine LR, etc.)
"""
import copy as pycopy
import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from peft import LoraConfig, get_peft_model
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Global config
NOISE_RATE = 0.2
FORGET_CLASS = 0
LORA_RANK = 8
LORA_ALPHA = 16
BATCH_SIZE = 8
NUM_BATCHES_CKA = 5
INFLUENCE_SAMPLES = 50
DIAGNOSTIC_STEPS = 100
KL_WEIGHT = 1.0
LISSA_ITERATIONS = 5
NUM_BLOCKS = 12
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU for ViT-Base on small GPUs (3.9GB VRAM is insufficient for 86M params + AMP)
if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 8e9:
    device = torch.device("cpu")
    print("GPU VRAM < 8GB, switching to CPU mode for ViT-Base")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# Helper functions
# ============================================================

def get_spectral_norms(model):
    norms = {"attn_qkv": [], "attn_proj": [], "mlp_fc1": [], "mlp_fc2": []}
    for block in model.blocks:
        with torch.no_grad():
            norms["attn_qkv"].append(torch.linalg.svdvals(block.attn.qkv.weight).max().item())
            norms["attn_proj"].append(torch.linalg.svdvals(block.attn.proj.weight).max().item())
            norms["mlp_fc1"].append(torch.linalg.svdvals(block.mlp.fc1.weight).max().item())
            norms["mlp_fc2"].append(torch.linalg.svdvals(block.mlp.fc2.weight).max().item())
    return norms


def add_noise_to_dataset(dataset, noise_rate):
    if noise_rate == 0.0:
        return dataset
    noisy = pycopy.copy(dataset)
    original = np.array(dataset.targets)
    noisy_labels = original.copy()
    n_noisy = int(noise_rate * len(noisy_labels))
    idx = np.random.choice(len(noisy_labels), n_noisy, replace=False)
    for i in idx:
        orig = noisy_labels[i]
        new = np.random.randint(0, 9)
        if new >= orig:
            new += 1
        noisy_labels[i] = new
    noisy.targets = noisy_labels.tolist() if isinstance(dataset.targets, list) else noisy_labels
    return noisy


def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_diagnostic(model, dataset, device, batch_size, n_steps, use_amp=True, grad_clip=1.0, accum=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model.train()
    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    total_steps = n_steps // accum
    warmup = max(1, total_steps // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = get_cosine_scheduler(optimizer, warmup, total_steps)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for images, labels in loader:
        if step >= n_steps:
            break
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = criterion(model(images), labels) / accum
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        step += 1
    model.eval()


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


class GELUExtractor:
    def __init__(self, model):
        self.model = model
        self.activations = defaultdict(list)
        self.hooks = []
        for i, block in enumerate(model.blocks):
            def hook_fn(module, inp, out, idx=i):
                self.activations[f"block{idx}"].append(out.detach().cpu().float())
            self.hooks.append(block.mlp.act.register_forward_hook(hook_fn))

    def get(self, block_idx):
        key = f"block{block_idx}"
        return torch.cat(self.activations[key], dim=0) if key in self.activations else None

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_influence_scores_ihvp(model, dataset, device, n_samples, lissa_iters,
                                   support_batch_size=128, damping=0.01, scale=25.0):
    model.eval()
    head_params = list(model.head.parameters())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    support_loader = DataLoader(dataset, batch_size=support_batch_size, shuffle=True, num_workers=2)
    ce = nn.CrossEntropyLoss()

    all_features, all_labels = [], []
    collected = 0
    with torch.no_grad():
        for imgs, labels in loader:
            if collected >= n_samples:
                break
            feats = model.forward_features(imgs.to(device))[:, 0, :]
            all_features.append(feats.cpu())
            all_labels.append(labels)
            collected += imgs.size(0)
    all_features = torch.cat(all_features)[:n_samples].to(device)
    all_labels = torch.cat(all_labels)[:n_samples].to(device)

    support_imgs, support_labels = next(iter(support_loader))
    with torch.no_grad():
        support_feats = model.forward_features(support_imgs.to(device))[:, 0, :]
    support_labels = support_labels.to(device)

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
    for idx in range(n):
        feat_i = all_features[idx:idx + 1]
        label_i = all_labels[idx:idx + 1]
        logits_i = model.head(feat_i)
        loss_i = ce(logits_i, label_i)
        grad_i = torch.autograd.grad(loss_i, head_params, retain_graph=False)
        flat_grad_i = torch.cat([g.reshape(-1) for g in grad_i]).detach()
        h_inv = lissa_inverse_hvp(flat_grad_i, lissa_iters)
        influence[idx] = -torch.dot(flat_grad_i, h_inv).item()
        correct[idx] = (logits_i.argmax(dim=1) == label_i).item()

    return influence.numpy(), all_features.cpu().numpy(), all_labels.cpu().numpy(), correct.numpy()


def unlearn_with_lora(model, ref_model, f_loader, r_loader, device, epochs=3, lam=1.0, kl_lambda=1.0, grad_clip=1.0):
    model.train()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    history = {"total": [], "forget": [], "retain": [], "kl": []}

    for epoch in range(epochs):
        sums = {"total": 0.0, "forget": 0.0, "retain": 0.0, "kl": 0.0}
        n_batches = 0
        for (f_imgs, f_labels), (r_imgs, r_labels) in zip(f_loader, r_loader):
            f_imgs, f_labels = f_imgs.to(device), f_labels.to(device)
            r_imgs, r_labels = r_imgs.to(device), r_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                f_out = model(f_imgs)
                forget_loss = ce(f_out, f_labels)
                r_out = model(r_imgs)
                retain_loss = ce(r_out, r_labels)
                with torch.no_grad():
                    ref_logits = ref_model(r_imgs)
                kl_loss = F.kl_div(F.log_softmax(r_out, dim=1), F.softmax(ref_logits, dim=1), reduction="batchmean")
                total_loss = -forget_loss + lam * retain_loss + kl_lambda * kl_loss
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(lora_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(lora_params, grad_clip)
                optimizer.step()
            sums["total"] += total_loss.item()
            sums["forget"] += forget_loss.item()
            sums["retain"] += retain_loss.item()
            sums["kl"] += kl_loss.item()
            n_batches += 1
        for k in sums:
            history[k].append(sums[k] / max(n_batches, 1))
        print(f"  Epoch {epoch+1}: total={history['total'][-1]:.4f} forget={history['forget'][-1]:.4f} "
              f"retain={history['retain'][-1]:.4f} kl={history['kl'][-1]:.4f}")
    return history


def evaluate_model(model, dataset, device, n_samples=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    correct = total = total_loss = 0
    all_preds, all_labels = [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, batch_labels in loader:
            imgs, batch_labels = imgs.to(device), batch_labels.to(device)
            outputs = model(imgs)
            loss = ce(outputs, batch_labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            if n_samples and total >= n_samples:
                break
    return correct / total, total_loss / total, np.array(all_preds), np.array(all_labels)


# ============================================================
# PHASE 1: Spectral Tracking
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Spectral Tracking")
print("=" * 60)

transform_display = transforms.Compose([transforms.ToTensor()])
transform_train = transforms.Compose([
    transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_eval = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DATA_ROOT = "./data"
train_dataset_display = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform_display)
test_dataset_display = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform_display)
train_dataset_model = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_train)
test_dataset_model = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_eval)
print(f"Train: {len(train_dataset_display)} | Test: {len(test_dataset_display)} | Classes: {len(CLASS_NAMES)}")

# Load clean model (try local cache first, then download)
import os as _os
LOCAL_CACHE = "./vit_tiny_cache/pytorch_model.bin"

# Try to load timm model - if download fails, use local weights
try:
    print("Loading ViT-Base pretrained...")
    clean_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
except Exception as e:
    print(f"Download failed ({e}), trying local weights...")
    clean_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
    if _os.path.exists(LOCAL_CACHE):
        sd = torch.load(LOCAL_CACHE, map_location="cpu", weights_only=True)
        # Map HF keys to timm keys
        mapped = {}
        for k, v in sd.items():
            new_k = k.replace("vit.", "").replace("encoder.", "")
            new_k = new_k.replace("attention.attention.", "attn.")
            new_k = new_k.replace("intermediate.dense.", "mlp.fc1.")
            new_k = new_k.replace("output.dense.", "mlp.fc2.")
            new_k = new_k.replace("layernorm_before.", "norm1.")
            new_k = new_k.replace("layernorm_after.", "norm2.")
            new_k = new_k.replace("embeddings.", "")
            mapped[new_k] = v
        clean_model.load_state_dict(mapped, strict=False)
        print("Loaded local weights (partial match)")

clean_model.eval().to(device)
clean_norms = get_spectral_norms(clean_model)
print(f"Clean model loaded: {sum(p.numel() for p in clean_model.parameters())} params")

# Train noisy model
noisy_dataset = add_noise_to_dataset(train_dataset_model, NOISE_RATE)
noisy_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
noisy_model.load_state_dict(clean_model.state_dict())
noisy_model.to(device)
t0 = time.time()
train_diagnostic(noisy_model, noisy_dataset, device, BATCH_SIZE, DIAGNOSTIC_STEPS)
t1 = time.time()
noisy_norms = get_spectral_norms(noisy_model)
print(f"Noisy model trained in {t1-t0:.1f}s")

# Plot spectral norms
layers = ["attn_qkv", "attn_proj", "mlp_fc1", "mlp_fc2"]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax_idx, (key, title) in enumerate(zip(layers, ["Attention QKV", "Attention Proj", "MLP FC1", "MLP FC2"])):
    c = clean_norms[key]
    n = noisy_norms[key]
    x = np.arange(NUM_BLOCKS)
    w = 0.35
    axes[ax_idx].bar(x - w/2, c, w, label="Clean", alpha=0.85, color="steelblue")
    axes[ax_idx].bar(x + w/2, n, w, label=f"Noisy ({NOISE_RATE})", alpha=0.85, color="salmon")
    delta = np.mean(np.array(n) - np.array(c))
    axes[ax_idx].set_title(f"{title}\nΔ = {delta:.4f}")
    axes[ax_idx].set_xlabel("Block")
    axes[ax_idx].set_ylabel("Spectral Norm")
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels([f"B{i}" for i in range(NUM_BLOCKS)], fontsize=7)
    axes[ax_idx].legend(fontsize=8)
plt.suptitle(f"Phase 1: Spectral Norm Decay Under Label Noise (rate={NOISE_RATE})", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/phase1_spectral_norms.png", dpi=150, bbox_inches="tight")
plt.close()
print("Phase 1 plot saved: outputs/figures/phase1_spectral_norms.png")

# ============================================================
# PHASE 2: CKA Collapse
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: CKA Collapse Mapping")
print("=" * 60)

loader_cka = DataLoader(train_dataset_model, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

extractor_clean = GELUExtractor(clean_model)
with torch.no_grad():
    for i_batch, (imgs, _) in enumerate(loader_cka):
        if i_batch >= NUM_BATCHES_CKA:
            break
        _ = clean_model(imgs.to(device))

extractor_noisy = GELUExtractor(noisy_model)
with torch.no_grad():
    for i_batch, (imgs, _) in enumerate(loader_cka):
        if i_batch >= NUM_BATCHES_CKA:
            break
        _ = noisy_model(imgs.to(device))

cka_values = []
for blk in range(NUM_BLOCKS):
    acts_c = extractor_clean.get(blk)
    acts_n = extractor_noisy.get(blk)
    cka_val = compute_cka(acts_c, acts_n) if (acts_c is not None and acts_n is not None) else 0.0
    cka_values.append(cka_val)
    print(f"  Block B{blk}: CKA = {cka_val:.4f}")

extractor_clean.remove()
extractor_noisy.remove()

# Plot CKA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cka_arr = np.array(cka_values)
im = axes[0].imshow(cka_arr.reshape(1, -1), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
axes[0].set_yticks([0])
axes[0].set_yticklabels(["GELU Output"])
axes[0].set_xlabel("Transformer Block")
axes[0].set_title(f"CKA: Clean vs Noisy (rate={NOISE_RATE})")
axes[0].set_xticks(range(NUM_BLOCKS))
axes[0].set_xticklabels([f"B{i}" for i in range(NUM_BLOCKS)])
for j in range(NUM_BLOCKS):
    axes[0].text(j, 0, f"{cka_values[j]:.3f}", ha="center", va="center", fontsize=8, fontweight="bold")
plt.colorbar(im, ax=axes[0], fraction=0.046)

colors_bar = ["#2ecc71" if v > 0.8 else "#f39c12" if v > 0.5 else "#e74c3c" for v in cka_values]
axes[1].bar(range(NUM_BLOCKS), cka_values, color=colors_bar, edgecolor="black", linewidth=0.5)
axes[1].axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="High similarity")
axes[1].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate")
axes[1].set_xlabel("Transformer Block")
axes[1].set_ylabel("CKA Score")
axes[1].set_title("CKA Score by Block Depth")
axes[1].set_xticks(range(NUM_BLOCKS))
axes[1].set_xticklabels([f"B{i}" for i in range(NUM_BLOCKS)], fontsize=8)
axes[1].legend()
plt.suptitle("Phase 2: CKA Collapse Mapping", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/phase2_cka_collapse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Phase 2 plot saved: outputs/figures/phase2_cka_collapse.png")

critical_block = int(np.argmin(cka_arr))
print(f"Most affected block: B{critical_block} (CKA={cka_arr[critical_block]:.4f})")
print(f"Mean CKA: {np.mean(cka_arr):.4f} ± {np.std(cka_arr):.4f}")

# ============================================================
# PHASE 3: Influence Scores (LiSSA iHVP)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Influence Localization (LiSSA iHVP)")
print("=" * 60)

t0 = time.time()
influence, features, labels_arr, correct = compute_influence_scores_ihvp(
    clean_model, train_dataset_model, device,
    n_samples=INFLUENCE_SAMPLES, lissa_iters=LISSA_ITERATIONS,
)
t1 = time.time()
print(f"Influence scores computed for {len(influence)} samples in {t1-t0:.1f}s")
print(f"  Mean influence: {np.mean(influence):.4f} ± {np.std(influence):.4f}")
print(f"  Correct predictions: {correct.mean()*100:.1f}%")

# t-SNE
n_display = min(500, len(features))
idx = np.random.choice(len(features), n_display, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
features_2d = tsne.fit_transform(features[idx])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=influence[idx], cmap="RdBu_r", s=10, alpha=0.7)
axes[0].set_title("t-SNE Colored by LiSSA Influence Score")
plt.colorbar(scatter, ax=axes[0], label="Influence")

unique_labels = np.unique(labels_idx := labels_arr[idx])
cmap = plt.cm.tab10
for i, lab in enumerate(unique_labels):
    mask = labels_arr[idx] == lab
    axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1], c=[cmap(i)], label=CLASS_NAMES[lab], s=10, alpha=0.7)
axes[1].set_title("t-SNE Colored by Class")
axes[1].legend(fontsize=7, markerscale=3)
plt.suptitle("Phase 3: Influence-Weighted Token Embeddings", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/phase3_influence_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("Phase 3 plot saved: outputs/figures/phase3_influence_tsne.png")

# ============================================================
# PHASE 4: LoRA Unlearning
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: LoRA Unlearning Surgery")
print("=" * 60)

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.1,
    target_modules=["qkv", "proj", "fc1", "fc2"], bias="none",
)
base_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
base_model.load_state_dict(clean_model.state_dict())
lora_model = get_peft_model(base_model, lora_config)
lora_model.to(device)

frozen_ref = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
frozen_ref.load_state_dict(clean_model.state_dict())
frozen_ref.eval().to(device)
for p in frozen_ref.parameters():
    p.requires_grad = False

trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_model.parameters())
print(f"LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}")
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

forget_idx = [i for i, (_, l) in enumerate(train_dataset_model) if l == FORGET_CLASS]
retain_idx = [i for i, (_, l) in enumerate(train_dataset_model) if l != FORGET_CLASS]
forget_subset = Subset(train_dataset_model, forget_idx[:500])
retain_subset = Subset(train_dataset_model, retain_idx[:2000])
forget_loader = DataLoader(forget_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
retain_loader = DataLoader(retain_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"Forget class {FORGET_CLASS} ({CLASS_NAMES[FORGET_CLASS]}): {len(forget_subset)} samples")
print(f"Retain classes: {len(retain_subset)} samples")

print("\nRunning LoRA unlearning...")
t0 = time.time()
loss_history = unlearn_with_lora(
    lora_model, frozen_ref, forget_loader, retain_loader, device,
    epochs=3, lam=1.0, kl_lambda=KL_WEIGHT,
)
t1 = time.time()
print(f"Unlearning complete in {t1-t0:.1f}s")

# Evaluate
print("\nEvaluating on FULL test set...")
overall_acc, overall_loss, preds, true_labels = evaluate_model(lora_model, test_dataset_model, device)
print(f"Overall Accuracy: {overall_acc*100:.2f}% | Loss: {overall_loss:.4f}")

per_class_acc = {}
for c in range(10):
    mask = true_labels == c
    if mask.sum() > 0:
        per_class_acc[c] = (preds[mask] == true_labels[mask]).mean()

print(f"\nPer-Class Accuracy:")
for c, acc in sorted(per_class_acc.items()):
    marker = " <-- FORGET" if c == FORGET_CLASS else ""
    print(f"  {CLASS_NAMES[c]:<12} {acc*100:.2f}%{marker}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm = confusion_matrix(true_labels, preds)
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Post-Unlearning)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

classes_list = list(range(10))
clean_accs = [100.0] * 10  # placeholder
lora_accs = [per_class_acc[c] * 100 for c in classes_list]
x = np.arange(10)
w = 0.35
axes[1].bar(x + w/2, lora_accs, w, label="Unlearned", alpha=0.8, color="salmon")
axes[1].axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Random baseline")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Per-Class Accuracy After Unlearning")
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_NAMES, rotation=45)
axes[1].legend()
plt.suptitle("Phase 4: LoRA Unlearning Results", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/phase4_lora_unlearning.png", dpi=150, bbox_inches="tight")
plt.close()
print("Phase 4 plot saved: outputs/figures/phase4_lora_unlearning.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ALL PHASES COMPLETE")
print("=" * 60)
print(f"Phase 1: Spectral norms extracted for clean vs noisy models")
print(f"Phase 2: CKA computed across {NUM_BLOCKS} blocks (most affected: B{critical_block})")
print(f"Phase 3: Influence scores for {INFLUENCE_SAMPLES} samples")
print(f"Phase 4: LoRA unlearning | Forget class acc: {per_class_acc.get(FORGET_CLASS, 0)*100:.2f}%")
print(f"Overall accuracy: {overall_acc*100:.2f}%")
print(f"\nOutputs saved to: outputs/figures/")
