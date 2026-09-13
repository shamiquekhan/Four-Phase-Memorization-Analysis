"""
Label-corruption utilities with guaranteed label change and full provenance.

v0 bug being fixed here: np.random.randint(0, 10, n) can re-assign the
original label (~10% of "corrupted" indices were silently unchanged), yet
those indices were treated as ground-truth memorized examples downstream.

Contract of this module:
  - Every selected index is GUARANTEED to receive a label different from
    its original label (selected == changed, always).
  - Every corruption returns a CorruptionProvenance record and every
    training run saves it next to the checkpoints as
    ``corruption_provenance.json`` so datasets are auditable.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import List, Optional

import numpy as np


@dataclass
class CorruptionProvenance:
    """Audit record for a corrupted dataset."""
    kind: str                        # 'random' | 'targeted' | 'clean'
    n_samples: int
    noise_rate: Optional[float] = None
    source_class: Optional[int] = None
    target_class: Optional[int] = None
    seed: Optional[int] = None
    selected_indices: List[int] = field(default_factory=list)
    changed_indices: List[int] = field(default_factory=list)
    original_labels: List[int] = field(default_factory=list)
    corrupted_labels: List[int] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path) -> None:
        Path(path).write_text(self.to_json())


def _validate_change(original_labels: np.ndarray, new_labels: np.ndarray) -> None:
    """Hard invariant: a selected index must actually change label."""
    if np.any(original_labels == new_labels):
        bad = int(np.sum(original_labels == new_labels))
        raise AssertionError(
            f"Corruption invariant violated: {bad} selected samples kept their "
            f"original label. This must never happen."
        )


def corrupt_labels_random(dataset, noise_rate: float = 0.2, seed: int = 42,
                          n_classes: int = 10) -> tuple:
    """
    Corrupt an exact fraction of labels, guaranteeing new != original.

    Returns (dataset, provenance) where provenance.changed_indices can be
    used directly as ground-truth corrupted examples.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n_samples = len(targets)
    n_corrupt = int(n_samples * noise_rate)

    selected = rng.choice(n_samples, size=n_corrupt, replace=False)
    originals = targets[selected].copy()

    # For each selected sample, draw uniformly from the 9 labels != original.
    # Vectorized rejection-free sampling:
    #   offset in [1, n_classes-1], new = (orig + offset) % n_classes
    offsets = rng.integers(1, n_classes, size=n_corrupt)
    new_labels = (originals + offsets) % n_classes

    _validate_change(originals, new_labels)

    targets[selected] = new_labels
    dataset.targets = targets.tolist()

    provenance = CorruptionProvenance(
        kind='random',
        n_samples=n_samples,
        noise_rate=noise_rate,
        seed=seed,
        selected_indices=selected.tolist(),
        changed_indices=selected.tolist(),  # guaranteed identical
        original_labels=originals.tolist(),
        corrupted_labels=new_labels.tolist(),
    )
    return dataset, provenance


def apply_targeted_corruption(dataset, source: int, target: int, seed: int = 42):
    """
    Relabel every source-class sample to target (deterministic; used for ROME).

    Returns (dataset, provenance). changed_indices == indices of all
    source-class samples (source != target is asserted).
    """
    if source == target:
        raise ValueError("source and target classes must differ")

    targets = np.array(dataset.targets)
    mask = targets == source
    changed = np.where(mask)[0]
    originals = targets[changed].copy()

    targets[changed] = target
    dataset.targets = targets.tolist()

    provenance = CorruptionProvenance(
        kind='targeted',
        n_samples=len(targets),
        source_class=int(source),
        target_class=int(target),
        seed=seed,
        selected_indices=changed.tolist(),
        changed_indices=changed.tolist(),
        original_labels=originals.tolist(),
        corrupted_labels=[int(target)] * len(changed),
    )
    return dataset, provenance


def clean_provenance(n_samples: int) -> CorruptionProvenance:
    """Provenance record for an uncorrupted dataset."""
    return CorruptionProvenance(kind='clean', n_samples=n_samples)
