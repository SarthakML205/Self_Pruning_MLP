"""ReLU, Softmax, and other activation functions."""

from __future__ import annotations

from engine.tensor import Tensor
from nn.module import Module


class ReLU(Module):
    """Rectified linear unit: ``max(0, x)`` element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
