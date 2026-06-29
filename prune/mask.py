"""Boolean masking utilities for tensors."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def ensure_mask(tensor: Tensor) -> np.ndarray:
    """Return a strict boolean mask for ``tensor``, creating all-True if absent."""
    if tensor.mask is None:
        tensor.mask = np.ones(tensor.data.shape, dtype=bool)
    return tensor.mask


def apply_mask(tensor: Tensor) -> None:
    """Hard-zero parameter values at pruned (False) indices.

    Avoids "fake sparsity" from tiny floating-point residuals.
    """
    if tensor.mask is None:
        return
    tensor.data = tensor.data * tensor.mask.astype(np.float64)
