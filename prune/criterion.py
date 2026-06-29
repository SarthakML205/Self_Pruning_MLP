"""Importance scoring (Magnitude, Taylor/Saliency)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from engine.tensor import Tensor


class PruningCriterion(ABC):
    """Abstract importance scorer for unstructured weight pruning."""

    @abstractmethod
    def compute_scores(self, tensor: Tensor) -> np.ndarray:
        """Return a non-negative importance score per element of ``tensor``."""


class MagnitudeCriterion(PruningCriterion):
    """Baseline criterion: larger absolute weights are more important."""

    def compute_scores(self, tensor: Tensor) -> np.ndarray:
        return np.abs(tensor.data)


class SaliencyCriterion(PruningCriterion):
    """First-order Taylor saliency: ``|weight * gradient|``.

    Must be evaluated after ``loss.backward()`` and before ``optimizer.zero_grad()``
    so ``tensor.grad`` reflects the current mini-batch loss surface.
    """

    def compute_scores(self, tensor: Tensor) -> np.ndarray:
        if tensor.grad is None:
            raise ValueError(
                "SaliencyCriterion requires tensor.grad; call after backward()."
            )
        return np.abs(tensor.data * tensor.grad)
