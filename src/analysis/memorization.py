"""
Memorization tracking and temporal dynamics utilities (v2).

Provides per-example memorization onset detection, CSL computation,
forgetting events, and temporal feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class MemorizationEvent:
    """Record of a single example's memorization transition."""
    example_idx: int
    true_label: int
    noisy_label: int
    onset_epoch: Optional[int] = None
    onset_checkpoint: Optional[int] = None
    is_persistent: bool = False
    loss_trajectory: List[float] = field(default_factory=list)
    true_label_loss_trajectory: List[float] = field(default_factory=list)
    noisy_label_loss_trajectory: List[float] = field(default_factory=list)
    margin_trajectory: List[float] = field(default_factory=list)
    prediction_trajectory: List[int] = field(default_factory=list)


def compute_csl(loss_trajectory: List[float]) -> float:
    """Cumulative Sample Loss (CSL) = sum of losses across epochs."""
    return float(sum(loss_trajectory))


def compute_forgetting_events(prediction_trajectory: List[int],
                               true_label: int) -> int:
    """
    Count forgetting events (Toneva et al., 2018).
    An event occurs when prediction switches from true_label to something else.
    """
    events = 0
    was_correct = False
    for pred in prediction_trajectory:
        is_correct = (pred == true_label)
        if was_correct and not is_correct:
            events += 1
        was_correct = is_correct
    return events


def compute_memorization_onset(prediction_trajectory: List[int],
                                noisy_label: int,
                                true_label: int,
                                min_persistence: int = 3,
                                window: int = 5) -> Optional[int]:
    """
    Detect when an example becomes persistently memorized to the noisy label.

    Memorization criterion:
      - At epoch t: prediction == noisy_label AND prediction != true_label
      - Persistence: condition holds for at least min_persistence of next window epochs
    """
    n_epochs = len(prediction_trajectory)
    for t in range(n_epochs):
        pred = prediction_trajectory[t]
        if pred == noisy_label and pred != true_label:
            # Check persistence
            future = prediction_trajectory[t:min(t + window, n_epochs)]
            if sum(1 for p in future if p == noisy_label and p != true_label) >= min_persistence:
                return t
    return None


def extract_per_example_metrics(model: nn.Module,
                                 dataloader,
                                 provenance,
                                 device: torch.device,
                                 criterion: nn.Module = None) -> Dict:
    """
    Extract per-example metrics at a single checkpoint.

    Returns:
        Dict with keys: losses, true_label_losses, noisy_label_losses,
                       margins, predictions, true_labels, noisy_labels
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction='none')

    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)
    orig_by_idx = {int(i): int(o) for i, o in
                   zip(provenance.changed_indices, provenance.original_labels)}

    all_losses = []
    all_true_losses = []
    all_noisy_losses = []
    all_margins = []
    all_preds = []
    all_true_labels = []
    all_noisy_labels = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)

            probs = torch.softmax(logits, dim=-1)
            margins = probs.gather(1, target.unsqueeze(1)).squeeze()

            all_losses.extend(loss.cpu().numpy())
            all_margins.extend(margins.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())

            start_idx = batch_idx * dataloader.batch_size
            for i in range(len(data)):
                idx = start_idx + i
                all_indices.append(idx)
                if idx in orig_by_idx:
                    all_true_labels.append(orig_by_idx[idx])
                    all_noisy_labels.append(target[i].item())
                    # True-label loss
                    true_target = torch.tensor([orig_by_idx[idx]], device=device)
                    true_loss = criterion(logits[i:i+1], true_target)
                    all_true_losses.append(true_loss.item())
                    # Noisy-label loss (already in all_losses)
                    all_noisy_losses.append(loss[i].item())
                else:
                    all_true_labels.append(target[i].item())
                    all_noisy_labels.append(target[i].item())
                    all_true_losses.append(loss[i].item())
                    all_noisy_losses.append(loss[i].item())

    return {
        'indices': np.array(all_indices),
        'losses': np.array(all_losses),
        'true_label_losses': np.array(all_true_losses),
        'noisy_label_losses': np.array(all_noisy_losses),
        'margins': np.array(all_margins),
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_true_labels),
        'noisy_labels': np.array(all_noisy_labels),
    }


def compute_gradient_conflict(model: nn.Module,
                               dataloader,
                               provenance,
                               device: torch.device,
                               layer_name: str = 'fc1') -> Dict:
    """
    Compute per-example gradient cosine between noisy-label and original-label objectives.
    """
    model.eval()
    changed_idx = set(int(i) for i in provenance.changed_indices)
    orig_by_idx = {int(i): int(o) for i, o in
                   zip(provenance.changed_indices, provenance.original_labels)}

    layer = getattr(model, layer_name)
    cosines = []
    indices_scored = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        for i in range(len(data)):
            idx = batch_idx * dataloader.batch_size + i
            if idx in changed_idx:
                model.zero_grad()
                torch.nn.functional.cross_entropy(
                    model(data[i:i+1]), target[i:i+1]).backward()
                g_noisy = getattr(layer, 'weight').grad.detach().clone().flatten()

                model.zero_grad()
                y_orig = torch.tensor([orig_by_idx[idx]], device=device)
                torch.nn.functional.cross_entropy(
                    model(data[i:i+1]), y_orig).backward()
                g_orig = getattr(layer, 'weight').grad.detach().clone().flatten()

                cos = torch.nn.functional.cosine_similarity(
                    g_noisy.unsqueeze(0), g_orig.unsqueeze(0)).item()
                cosines.append(cos)
                indices_scored.append(idx)

    return {
        'indices': np.array(indices_scored),
        'cosines': np.array(cosines),
        'mean': float(np.mean(cosines)) if cosines else 0.0,
        'std': float(np.std(cosines)) if cosines else 0.0,
        'frac_anti_aligned': float(np.mean(np.array(cosines) < 0)) if cosines else 0.0,
    }


def aggregate_temporal_metrics(checkpoint_results: List[Dict],
                                provenance) -> Dict:
    """
    Aggregate per-checkpoint metrics into temporal trajectories.

    checkpoint_results: list of dicts from extract_per_example_metrics()
    """
    n_epochs = len(checkpoint_results)
    n_samples = len(checkpoint_results[0]['indices'])

    # Initialize trajectories
    trajectories = {
        'losses': np.zeros((n_epochs, n_samples)),
        'true_label_losses': np.zeros((n_epochs, n_samples)),
        'noisy_label_losses': np.zeros((n_epochs, n_samples)),
        'margins': np.zeros((n_epochs, n_samples)),
        'predictions': np.zeros((n_epochs, n_samples), dtype=int),
    }

    for t, res in enumerate(checkpoint_results):
        for key in ['losses', 'true_label_losses', 'noisy_label_losses',
                    'margins', 'predictions']:
            trajectories[key][t] = res[key]

    # Compute per-example CSL
    csl = trajectories['losses'].sum(axis=0)

    # Compute forgetting events
    forgetting = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        true_label = checkpoint_results[0]['true_labels'][i]
        forgetting[i] = compute_forgetting_events(
            trajectories['predictions'][:, i].tolist(), true_label)

    # Compute memorization onset
    onset = np.full(n_samples, -1, dtype=int)
    for i in range(n_samples):
        true_label = checkpoint_results[0]['true_labels'][i]
        noisy_label = checkpoint_results[0]['noisy_labels'][i]
        pred_traj = trajectories['predictions'][:, i].tolist()
        onset_epoch = compute_memorization_onset(pred_traj, noisy_label, true_label)
        if onset_epoch is not None:
            onset[i] = onset_epoch

    return {
        'csl': csl,
        'forgetting': forgetting,
        'memorization_onset': onset,
        'trajectories': trajectories,
    }


def save_temporal_artifacts(checkpoint_results: List[Dict],
                             aggregated: Dict,
                             output_dir: Path,
                             seed: int):
    """Save temporal analysis artifacts for a seed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw trajectories
    np.savez(output_dir / f'seed_{seed}_trajectories.npz',
             **aggregated['trajectories'])

    # Save aggregated metrics
    np.savez(output_dir / f'seed_{seed}_aggregated.npz',
             csl=aggregated['csl'],
             forgetting=aggregated['forgetting'],
             memorization_onset=aggregated['memorization_onset'])

    # Save per-checkpoint per-example data
    for t, res in enumerate(checkpoint_results):
        np.savez(output_dir / f'seed_{seed}_epoch_{t:03d}.npz',
                 **res)

    # Save summary JSON
    summary = {
        'seed': seed,
        'n_epochs': len(checkpoint_results),
        'n_samples': len(checkpoint_results[0]['indices']),
        'n_memorized': int((aggregated['memorization_onset'] >= 0).sum()),
        'n_forgotten': int((aggregated['forgetting'] > 0).sum()),
    }
    with open(output_dir / f'seed_{seed}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    # Quick test
    print("Testing memorization utilities...")
    # Test CSL
    assert compute_csl([1.0, 0.8, 0.5, 0.3]) == 2.6
    # Test forgetting
    assert compute_forgetting_events([3, 3, 5, 5, 3], 3) == 2
    # Test onset
    pred = [2, 2, 2, 7, 7, 7, 7, 7]  # noisy=7, true=2
    onset = compute_memorization_onset(pred, noisy_label=7, true_label=2)
    assert onset == 3
    print("All tests passed.")