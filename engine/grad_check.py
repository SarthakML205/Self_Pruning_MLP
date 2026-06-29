"""Finite difference numerical gradient checker."""

from __future__ import annotations

from typing import Callable

import numpy as np

from engine.tensor import Tensor


def _scalar_value(tensor: Tensor) -> float:
    if tensor.data.size != 1:
        raise ValueError("Gradient check requires f(*tensors) to return a scalar Tensor")
    return float(tensor.data)


def check_gradients(
    f: Callable[..., Tensor],
    *tensors: Tensor,
    epsilon: float = 1e-5,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Compare analytical autodiff gradients against central finite differences.

    Raises ``AssertionError`` when any element differs beyond ``rtol``/``atol``.
  """
    differentiable = [t for t in tensors if t.requires_grad]
    if not differentiable:
        raise ValueError("At least one input tensor must have requires_grad=True")

    for tensor in differentiable:
        tensor.zero_grad()

    analytical_output = f(*tensors)
    analytical_output.backward()
    analytical_grads = [t.grad.copy() for t in differentiable]

    numerical_grads: list[np.ndarray] = []
    for tensor in differentiable:
        num_grad = np.zeros_like(tensor.data, dtype=np.float64)

        for index in np.ndindex(tensor.data.shape):
            # Pruned parameters are frozen: analytical grad is forced to zero and
            # finite differences at dead indices are not meaningful in Phase 1.
            if tensor.mask is not None and not bool(tensor.mask[index]):
                continue

            original = float(tensor.data[index])

            tensor.data[index] = original + epsilon
            plus = _scalar_value(f(*tensors))

            tensor.data[index] = original - epsilon
            minus = _scalar_value(f(*tensors))

            tensor.data[index] = original
            num_grad[index] = (plus - minus) / (2.0 * epsilon)

        numerical_grads.append(num_grad)

    for tensor, analytical, numerical in zip(differentiable, analytical_grads, numerical_grads):
        if tensor.mask is not None:
            dead = ~tensor.mask
            if not np.all(analytical[dead] == 0.0):
                raise AssertionError("Masked tensor received non-zero gradient at pruned indices")

            live = tensor.mask
            if not np.allclose(analytical[live], numerical[live], rtol=rtol, atol=atol):
                max_diff = float(np.max(np.abs(analytical[live] - numerical[live])))
                raise AssertionError(
                    f"Gradient check failed for tensor shape {tensor.data.shape}. "
                    f"Max |analytical - numerical| at live indices = {max_diff:.3e}"
                )
            continue

        if not np.allclose(analytical, numerical, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(analytical - numerical)))
            raise AssertionError(
                f"Gradient check failed for tensor shape {tensor.data.shape}. "
                f"Max |analytical - numerical| = {max_diff:.3e}"
            )
