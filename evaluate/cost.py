"""Theoretical FLOP/MAC calculation for sparse layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn.linear import Linear
from nn.module import Module, Sequential


@dataclass(frozen=True)
class FlopReport:
    """Per-token theoretical multiply-accumulate cost for linear layers."""

    dense_flops: int
    sparse_flops: int

    @property
    def savings_fraction(self) -> float:
        if self.dense_flops == 0:
            return 0.0
        return 1.0 - (self.sparse_flops / self.dense_flops)


def _iter_linear_layers(module: Module) -> list[Linear]:
    """Collect every ``Linear`` layer in a module tree."""
    layers: list[Linear] = []
    if isinstance(module, Linear):
        return [module]

    if isinstance(module, Sequential):
        for layer in module.layers:
            layers.extend(_iter_linear_layers(layer))
        return layers

    for value in module.__dict__.values():
        if isinstance(value, Linear):
            layers.append(value)
        elif isinstance(value, Module):
            layers.extend(_iter_linear_layers(value))
    return layers


def _linear_dense_flops(layer: Linear) -> int:
    """Dense GEMM cost per token: ``2 * in_features * out_features`` MACs."""
    return 2 * layer.in_features * layer.out_features


def _linear_sparse_flops(layer: Linear) -> int:
    """Sparse GEMM cost per token: ``2 * (non-zero weights)`` MACs.

    Multiplying a dense matrix by a zero mask still costs full dense FLOPs in
  NumPy. This function counts only *active* connections, which is the honest
  cost model for CSR/CSC or structured sparse tensor cores.
    """
    weight = layer.weight
    if weight.mask is None:
        return _linear_dense_flops(layer)
    return int(2 * np.count_nonzero(weight.mask))


def compute_theoretical_flops(module: Module) -> FlopReport:
    """Sum theoretical per-token FLOPs across all ``Linear`` layers in ``module``."""
    layers = _iter_linear_layers(module)
    dense_total = sum(_linear_dense_flops(layer) for layer in layers)
    sparse_total = sum(_linear_sparse_flops(layer) for layer in layers)
    return FlopReport(dense_flops=dense_total, sparse_flops=sparse_total)
