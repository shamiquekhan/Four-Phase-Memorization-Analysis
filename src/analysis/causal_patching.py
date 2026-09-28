"""
Causal activation patching and neuron ablation (v2).

Implements necessity/sufficiency tests via activation patching:
  - Corrupt model + clean activation → necessity test
  - Clean model + corrupt activation → sufficiency test

Also provides neuron-level causal ablation and subspace projection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json


def get_layer_activations(model: nn.Module,
                          data: torch.Tensor,
                          layer_names: List[str],
                          device: torch.device) -> Dict[str, torch.Tensor]:
    """Extract activations from specified layers."""
    model.eval()
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handles = []
    for name in layer_names:
        layer = getattr(model, name)
        handles.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(data.to(device))

    for h in handles:
        h.remove()

    return activations


def patch_activation(model: nn.Module,
                     data: torch.Tensor,
                     layer_name: str,
                     patch_values: torch.Tensor,
                     device: torch.Tensor) -> torch.Tensor:
    """
    Run model with patched activation at specified layer.

    Args:
        model: The model to run
        data: Input data
        layer_name: Name of layer to patch
        patch_values: Activation values to inject (same shape as layer output)
        device: Compute device

    Returns:
        Model output logits with patched activation
    """
    model.eval()
    original_activation = {}

    def make_patch_hook(name, patch):
        def hook(module, input, output):
            original_activation[name] = output
            return patch
        return hook

    handle = getattr(model, layer_name).register_forward_hook(
        make_patch_hook(layer_name, patch_values))

    with torch.no_grad():
        output = model(data.to(device))

    handle.remove()
    return output


def run_activation_patching(model_clean: nn.Module,
                             model_corrupt: nn.Module,
                             data: torch.Tensor,
                             layer_name: str,
                             device: torch.device,
                             true_label: int,
                             noisy_label: int) -> Dict:
    """
    Run the four-condition activation patching experiment:

    | Model   | Hidden State | Meaning            |
    | ------- | ------------ | ------------------ |
    | Clean   | Clean        | baseline (C→C)     |
    | Corrupt | Corrupt      | memorized (R→R)    |
    | Corrupt | Clean        | necessity (R→C)    |
    | Clean   | Corrupt      | sufficiency (C→R)  |

    Args:
        model_clean: Model trained on clean labels
        model_corrupt: Model trained with label noise
        data: Input examples (batch)
        layer_name: Layer to patch (e.g., 'fc1' for post-ReLU)
        device: Compute device
        true_label: Ground-truth class
        noisy_label: Corrupted class

    Returns:
        Dict with logits and margins for all four conditions
    """
    # Get baseline activations
    clean_acts = get_layer_activations(model_clean, data, [layer_name], device)
    corrupt_acts = get_layer_activations(model_corrupt, data, [layer_name], device)

    h_clean = clean_acts[layer_name]
    h_corrupt = corrupt_acts[layer_name]

    # Four conditions
    results = {}

    # C→C (clean baseline)
    logits_cc = model_clean(data.to(device))
    probs_cc = torch.softmax(logits_cc, dim=-1)
    results['clean_clean'] = {
        'logits': logits_cc.cpu(),
        'margin_true': probs_cc[:, true_label].cpu(),
        'margin_noisy': probs_cc[:, noisy_label].cpu(),
        'pred': logits_cc.argmax(1).cpu(),
    }

    # R→R (corrupted baseline)
    logits_rr = model_corrupt(data.to(device))
    probs_rr = torch.softmax(logits_rr, dim=-1)
    results['corrupt_corrupt'] = {
        'logits': logits_rr.cpu(),
        'margin_true': probs_rr[:, true_label].cpu(),
        'margin_noisy': probs_rr[:, noisy_label].cpu(),
        'pred': logits_rr.argmax(1).cpu(),
    }

    # R→C (necessity: does clean activation restore true label?)
    logits_rc = patch_activation(model_corrupt, data, layer_name, h_clean, device)
    probs_rc = torch.softmax(logits_rc, dim=-1)
    results['corrupt_clean'] = {
        'logits': logits_rc.cpu(),
        'margin_true': probs_rc[:, true_label].cpu(),
        'margin_noisy': probs_rc[:, noisy_label].cpu(),
        'pred': logits_rc.argmax(1).cpu(),
    }

    # C→R (sufficiency: does corrupt activation induce noisy label?)
    logits_cr = patch_activation(model_clean, data, layer_name, h_corrupt, device)
    probs_cr = torch.softmax(logits_cr, dim=-1)
    results['clean_corrupt'] = {
        'logits': logits_cr.cpu(),
        'margin_true': probs_cr[:, true_label].cpu(),
        'margin_noisy': probs_cr[:, noisy_label].cpu(),
        'pred': logits_cr.argmax(1).cpu(),
    }

    # Compute patch effects
    results['patch_effects'] = {
        'necessity_margin_delta': (
            results['corrupt_clean']['margin_true'] -
            results['corrupt_corrupt']['margin_true']
        ).tolist(),
        'sufficiency_margin_delta': (
            results['clean_corrupt']['margin_noisy'] -
            results['clean_clean']['margin_noisy']
        ).tolist(),
        'necessity_recovery': (
            (results['corrupt_clean']['pred'] == true_label).float() -
            (results['corrupt_corrupt']['pred'] == true_label).float()
        ).mean().item(),
        'sufficiency_induction': (
            (results['clean_corrupt']['pred'] == noisy_label).float() -
            (results['clean_clean']['pred'] == noisy_label).float()
        ).mean().item(),
    }

    return results


def neuron_ablation(model: nn.Module,
                    data: torch.Tensor,
                    neuron_idx: int,
                    layer_name: str,
                    device: torch.device) -> torch.Tensor:
    """
    Ablate a specific neuron (set its activation to zero) and return logits.
    """
    def make_ablation_hook(name, idx):
        def hook(module, input, output):
            out = output.clone()
            out[:, idx] = 0.0
            return out
        return hook

    handle = getattr(model, layer_name).register_forward_hook(
        make_ablation_hook(layer_name, neuron_idx))

    with torch.no_grad():
        logits = model(data.to(device))

    handle.remove()
    return logits


def compute_neuron_causal_effects(model: nn.Module,
                                   data: torch.Tensor,
                                   layer_name: str,
                                   device: torch.device,
                                   target_class: int,
                                   max_neurons: Optional[int] = None) -> Dict:
    """
    Compute causal effect of ablating each neuron on target-class margin.

    Effect = margin_with_neuron - margin_without_neuron
    Positive = neuron helps the target class
    Negative = neuron hurts the target class
    """
    model.eval()

    # Baseline
    with torch.no_grad():
        baseline_logits = model(data.to(device))
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_margin = baseline_probs[:, target_class].cpu()

    layer = getattr(model, layer_name)
    n_neurons = layer.weight.shape[0] if layer_name.startswith('fc') else layer.weight.shape[1]

    if max_neurons is not None:
        n_neurons = min(n_neurons, max_neurons)

    effects = np.zeros(n_neurons)
    margins = np.zeros((n_neurons, len(data)))

    for i in range(n_neurons):
        ablated_logits = neuron_ablation(model, data, i, layer_name, device)
        ablated_probs = torch.softmax(ablated_logits, dim=-1)
        ablated_margin = ablated_probs[:, target_class].cpu()

        effects[i] = (baseline_margin - ablated_margin).mean().item()
        margins[i] = (baseline_margin - ablated_margin).numpy()

    return {
        'neuron_effects': effects,
        'per_example_margins': margins,
        'most_positive': int(np.argmax(effects)),
        'most_negative': int(np.argmin(effects)),
    }


def build_memorization_subspace(model_clean: nn.Module,
                                 model_corrupt: nn.Module,
                                 dataloader,
                                 layer_name: str,
                                 device: torch.device,
                                 provenance,
                                 max_examples: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the memorization subspace from hidden activation differences.

    For memorized examples: D = h_corrupt - h_clean
    Returns U, S, Vh from SVD of D matrix.
    """
    model_clean.eval()
    model_corrupt.eval()

    # Find memorized examples
    from analysis.memorization import compute_behavioral_memorization

    # Get all data
    all_data = []
    all_targets = []
    all_indices = []
    for batch_idx, (data, target) in enumerate(dataloader):
        all_data.append(data)
        all_targets.append(target)
        start = batch_idx * dataloader.batch_size
        all_indices.extend(range(start, start + len(data)))

    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)

    # Use a subset for computational efficiency
    if len(all_data) > max_examples:
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(all_data), size=max_examples, replace=False)
        all_data = all_data[subset_idx]
        all_targets = all_targets[subset_idx]
        all_indices = [all_indices[i] for i in subset_idx]

    # Get activations
    clean_acts = get_layer_activations(model_clean, all_data, [layer_name], device)
    corrupt_acts = get_layer_activations(model_corrupt, all_data, [layer_name], device)

    h_clean = clean_acts[layer_name]
    h_corrupt = corrupt_acts[layer_name]

    # D = h_corrupt - h_clean
    D = h_corrupt - h_clean

    # SVD
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)

    return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()


def apply_subspace_patch(model: nn.Module,
                         data: torch.Tensor,
                         layer_name: str,
                         U: np.ndarray,
                         k: int,
                         device: torch.device,
                         mode: str = 'zero') -> torch.Tensor:
    """
    Project out (or scale) the top-k subspace components.

    mode:
      'zero'   - zero out top-k components (remove memorization direction)
      'scale'  - scale top-k components by 0.5
      'clean'  - replace with clean activations (if U was built from D)
    """
    def make_subspace_hook(name, U_k, mode):
        def hook(module, input, output):
            # output: [batch, hidden]
            coeffs = output @ torch.from_numpy(U_k).to(output.device).T  # [batch, k]
            if mode == 'zero':
                # Reconstruct without top-k
                recon = output - coeffs @ torch.from_numpy(U_k).to(output.device)
                return recon
            elif mode == 'scale':
                recon = output - 0.5 * coeffs @ torch.from_numpy(U_k).to(output.device)
                return recon
            return output
        return hook

    U_k = U[:, :k]
    handle = getattr(model, layer_name).register_forward_hook(
        make_subspace_hook(layer_name, U_k, mode))

    with torch.no_grad():
        logits = model(data.to(device))

    handle.remove()
    return logits


def save_patching_results(results: Dict, output_path: Path):
    """Save patching results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))


if __name__ == '__main__':
    print("Causal patching module loaded. Run with actual models for testing.")