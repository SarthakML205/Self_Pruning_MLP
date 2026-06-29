"""Forward and backward operations (Add, MatMul, ReLU, etc.)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from engine.tensor import Tensor, _accumulate_grad, _as_tensor

Axis = Optional[Union[int, Sequence[int]]]


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a gradient to ``target_shape`` by summing over broadcast axes.

    When forward pass broadcasts a smaller tensor to a larger shape (e.g.
    ``(D,)`` into ``(N, D)``), the backward pass must sum gradients along
    those expanded axes so each parameter receives the total sensitivity.
    """
    grad = np.asarray(grad, dtype=np.float64)

    # Collapse leading dimensions introduced by broadcasting a scalar or lower-rank tensor.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum across axes where the target had size 1 but the gradient did not.
    for axis, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    if grad.shape != target_shape:
        grad = grad.reshape(target_shape)
    return grad


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with broadcasting."""
    out_data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.parents = (a, b)

        def _backward(grad_out: np.ndarray) -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1 — sum over broadcast dims when shapes differ.
            if a.requires_grad:
                _accumulate_grad(a, unbroadcast(grad_out, a.data.shape))
            if b.requires_grad:
                _accumulate_grad(b, unbroadcast(grad_out, b.data.shape))

        out._backward = _backward

    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction with broadcasting."""
    out_data = a.data - b.data
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.parents = (a, b)

        def _backward(grad_out: np.ndarray) -> None:
            if a.requires_grad:
                _accumulate_grad(a, unbroadcast(grad_out, a.data.shape))
            if b.requires_grad:
                _accumulate_grad(b, unbroadcast(-grad_out, b.data.shape))

        out._backward = _backward

    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication with broadcasting."""
    out_data = a.data * b.data
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.parents = (a, b)

        def _backward(grad_out: np.ndarray) -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            if a.requires_grad:
                _accumulate_grad(a, unbroadcast(grad_out * b.data, a.data.shape))
            if b.requires_grad:
                _accumulate_grad(b, unbroadcast(grad_out * a.data, b.data.shape))

        out._backward = _backward

    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division with broadcasting."""
    out_data = a.data / b.data
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.parents = (a, b)

        def _backward(grad_out: np.ndarray) -> None:
            if a.requires_grad:
                _accumulate_grad(a, unbroadcast(grad_out / b.data, a.data.shape))
            if b.requires_grad:
                _accumulate_grad(
                    b,
                    unbroadcast(-grad_out * a.data / (b.data ** 2), b.data.shape),
                )

        out._backward = _backward

    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication supporting batched leading dimensions."""
    out_data = a.data @ b.data
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad)

    if requires_grad:
        out.parents = (a, b)

        def _backward(grad_out: np.ndarray) -> None:
            # For Z = A @ B: dL/dA = dL/dZ @ B^T, dL/dB = A^T @ dL/dZ
            if a.requires_grad:
                grad_a = grad_out @ np.swapaxes(b.data, -1, -2)
                _accumulate_grad(a, grad_a)
            if b.requires_grad:
                grad_b = np.swapaxes(a.data, -1, -2) @ grad_out
                _accumulate_grad(b, grad_b)

        out._backward = _backward

    return out


def _normalize_axis(axis: Axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis if axis >= 0 else axis + ndim,)
    return tuple(ax if ax >= 0 else ax + ndim for ax in axis)


def _expand_grad_for_reduction(
    grad_out: np.ndarray,
    input_shape: tuple[int, ...],
    axis: Axis,
    keepdims: bool,
) -> np.ndarray:
    """Broadcast a reduction's output gradient back to the input shape."""
    grad = np.asarray(grad_out, dtype=np.float64)
    axes = _normalize_axis(axis, len(input_shape))

    if not keepdims:
        for ax in sorted(axes):
            grad = np.expand_dims(grad, axis=ax)

    return np.broadcast_to(grad, input_shape).copy()


def sum_op(
    a: Tensor,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum reduction; supports ``axis`` and ``keepdims`` like NumPy."""
    out_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        out.parents = (a,)
        input_shape = a.data.shape

        def _backward(grad_out: np.ndarray) -> None:
            _accumulate_grad(
                a,
                _expand_grad_for_reduction(grad_out, input_shape, axis, keepdims),
            )

        out._backward = _backward

    return out


def mean_op(
    a: Tensor,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean reduction; gradient scales by 1 / (number of reduced elements)."""
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        out.parents = (a,)
        input_shape = a.data.shape
        axes = _normalize_axis(axis, len(input_shape))
        reduced_size = int(np.prod([input_shape[ax] for ax in axes]))

        def _backward(grad_out: np.ndarray) -> None:
            grad = _expand_grad_for_reduction(grad_out, input_shape, axis, keepdims)
            _accumulate_grad(a, grad / reduced_size)

        out._backward = _backward

    return out


def relu(a: Tensor) -> Tensor:
    """ReLU activation: max(0, x). Gradient is 1 where input > 0, else 0."""
    out_data = np.maximum(0.0, a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        out.parents = (a,)
        mask = a.data > 0

        def _backward(grad_out: np.ndarray) -> None:
            _accumulate_grad(a, grad_out * mask)

        out._backward = _backward

    return out


def neg(a: Tensor) -> Tensor:
    """Unary negation."""
    return mul(a, _as_tensor(-1.0, requires_grad=False))
