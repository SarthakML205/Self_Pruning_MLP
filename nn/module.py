"""Base Module class for composable neural network layers."""

from __future__ import annotations

from typing import Iterator

from engine.tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses implement ``forward``; ``__call__`` routes invocations there.
    ``parameters`` recursively collects every ``Tensor`` with ``requires_grad=True``
    owned by this module or nested sub-modules.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Compute the module output for input ``x``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return all trainable tensors in this module tree."""
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset gradients on every parameter tensor."""
        for param in self.parameters():
            param.zero_grad()


class Sequential(Module):
    """Chain modules so output of layer *i* feeds layer *i + 1*."""

    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def parameter_iter(module: Module) -> Iterator[Tensor]:
    """Yield trainable tensors from ``module`` (convenience alias)."""
    yield from module.parameters()
