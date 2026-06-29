"""Linear/Dense layer implementation."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor
from nn.module import Module


def _kaiming_uniform(in_features: int, out_features: int) -> np.ndarray:
    """Kaiming/He initialization: mean 0, variance ``2 / in_features``.

  Fan-in scaling keeps pre-activation variance stable through ReLU layers,
  preventing vanishing or exploding activations early in training.
    """
    std = np.sqrt(2.0 / in_features)
    return np.random.randn(in_features, out_features) * std


class Linear(Module):
    """Affine transform: ``y = x @ W + b``.

    Weight shape is ``(in_features, out_features)`` so a batch ``(N, in_features)``
    maps to ``(N, out_features)`` via matrix multiplication.
    """

    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        weight_data = _kaiming_uniform(in_features, out_features)
        self.weight = Tensor(weight_data, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float64), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Bias broadcasts along the batch dimension: (N, out) + (out,)
        return (x @ self.weight) + self.bias
