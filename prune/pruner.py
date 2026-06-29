"""Orchestrates progressive, criterion-driven unstructured pruning."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.module import Module
from prune.criterion import PruningCriterion
from prune.mask import apply_mask, ensure_mask
from prune.schedule import CubicSchedule


class Pruner:
    """Update parameter masks each batch according to a sparsity schedule.

    Scores are computed globally across all trainable parameters, and the
    lowest-scoring connections are pruned to meet the scheduled sparsity level.
    """

    def __init__(
        self,
        module: Module,
        schedule: CubicSchedule,
        criterion: PruningCriterion,
    ) -> None:
        self.module = module
        self.schedule = schedule
        self.criterion = criterion
        self._last_sparsity: float = -1.0

    def _collect_parameters(self) -> list[Tensor]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def _global_keep_mask(self, sparsity: float, scores: np.ndarray) -> np.ndarray:
        """Build a flat boolean keep-mask that meets the exact sparsity budget.

        ``np.percentile`` supplies the global ranking cutoff; ``np.argpartition``
        then selects exactly how many connections to keep, avoiding off-by-one
        drift from tied scores on small tensors.
        """
        n = scores.size
        if sparsity <= 0.0:
            return np.ones(n, dtype=bool)
        if sparsity >= 1.0:
            return np.zeros(n, dtype=bool)

        num_prune = int(np.round(sparsity * n))
        num_keep = n - num_prune
        if num_keep <= 0:
            return np.zeros(n, dtype=bool)
        if num_keep >= n:
            return np.ones(n, dtype=bool)

        threshold = float(np.percentile(scores, sparsity * 100.0))
        keep_mask = scores >= threshold

        # Ties at ``threshold`` can skew the budget; partition enforces exact sparsity.
        if int(np.sum(keep_mask)) != num_keep:
            keep_mask = np.zeros(n, dtype=bool)
            if num_keep > 0:
                keep_indices = np.argpartition(scores, n - num_keep)[-num_keep:]
                keep_mask[keep_indices] = True

        return keep_mask

    def step(self) -> float:
        """Advance the schedule and refresh masks if sparsity changed."""
        current_sparsity = self.schedule.step()

        if current_sparsity == self._last_sparsity:
            return current_sparsity

        self._last_sparsity = current_sparsity
        params = self._collect_parameters()
        if not params:
            return current_sparsity

        per_param_scores: list[np.ndarray] = []
        for param in params:
            per_param_scores.append(self.criterion.compute_scores(param))

        all_scores = np.concatenate([scores.ravel() for scores in per_param_scores])
        flat_keep_mask = self._global_keep_mask(current_sparsity, all_scores)

        offset = 0
        for param, scores in zip(params, per_param_scores):
            size = scores.size
            new_mask = flat_keep_mask[offset : offset + size].reshape(scores.shape)
            offset += size

            ensure_mask(param)
            param.mask = new_mask.astype(bool)
            apply_mask(param)

        return current_sparsity

    def actual_sparsity(self) -> float:
        """Fraction of pruned (False) mask entries across all parameters."""
        params = self._collect_parameters()
        if not params:
            return 0.0

        total = 0
        pruned = 0
        for param in params:
            mask = ensure_mask(param)
            total += mask.size
            pruned += int(np.sum(~mask))
        return pruned / total if total > 0 else 0.0
