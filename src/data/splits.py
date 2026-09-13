"""
Data-split utilities enforcing the edit/eval firewall.

v0 bug being fixed here: ROME edit directions were constructed from the
same test_loader used for final evaluation, so reported "recovery"
partially measured fit to the edit-construction examples.

Contract:
  - split_edit_eval() partitions a dataset's indices into two DISJOINT sets.
  - Both index sets are returned so they can be saved into every experiment
    artifact (auditable provenance).
  - load_edit_eval_loaders() builds DataLoaders from the disjoint subsets.
"""
from pathlib import Path
import json
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset


class SplitProvenance:
    """Audit record for an edit/eval split."""

    def __init__(self, split_name: str, edit_indices: Sequence[int],
                 eval_indices: Sequence[int], seed: Optional[int] = None):
        self.edit_indices = sorted(int(i) for i in edit_indices)
        self.eval_indices = sorted(int(i) for i in eval_indices)
        self._assert_disjoint()
        self.split_name = split_name
        self.seed = seed

    def _assert_disjoint(self) -> None:
        if set(self.edit_indices) & set(self.eval_indices):
            raise AssertionError(
                "Split invariant violated: edit and eval index sets overlap. "
                "Edits must never be constructed from evaluation examples."
            )

    def to_dict(self) -> dict:
        return {
            'split_name': self.split_name,
            'seed': self.seed,
            'n_edit': len(self.edit_indices),
            'n_eval': len(self.eval_indices),
            'edit_indices': self.edit_indices,
            'eval_indices': self.eval_indices,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def split_edit_eval(indices: Sequence[int], eval_fraction: float = 0.5,
                    seed: int = 42, split_name: str = 'edit_eval') -> SplitProvenance:
    """
    Partition indices into disjoint edit (construction) and eval sets.

    Args:
        indices: candidate indices (e.g. all test-set indices of a class).
        eval_fraction: fraction of candidates reserved for final evaluation.
        seed: RNG seed for the partition (fixed => auditable).
    """
    rng = np.random.default_rng(seed)
    idx = np.array(sorted(int(i) for i in indices))
    idx = rng.permutation(idx)
    n_eval = int(len(idx) * eval_fraction)
    eval_idx = idx[:n_eval]
    edit_idx = idx[n_eval:]
    return SplitProvenance(split_name, edit_idx, eval_idx, seed=seed)


def load_edit_eval_loaders(dataset, split: SplitProvenance,
                           edit_batch_size: int = 128,
                           eval_batch_size: int = 256,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Build disjoint edit/eval DataLoaders from a SplitProvenance."""
    edit_loader = DataLoader(
        Subset(dataset, split.edit_indices),
        batch_size=edit_batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(
        Subset(dataset, split.eval_indices),
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return edit_loader, eval_loader
