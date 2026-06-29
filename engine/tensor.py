"""Core Tensor class and computational graph tracking."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int]


def _as_tensor(
    value: Union["Tensor", ArrayLike],
    requires_grad: bool = False,
) -> "Tensor":
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=requires_grad)


def _accumulate_grad(tensor: "Tensor", grad: np.ndarray) -> None:
    """Add ``grad`` into ``tensor.grad`` for reverse-mode accumulation."""
    if not tensor.requires_grad:
        return
    grad = np.asarray(grad, dtype=np.float64)
    if tensor.grad is None:
        tensor.grad = np.zeros_like(tensor.data, dtype=np.float64)
    tensor.grad = tensor.grad + grad


def _topological_sort(root: "Tensor") -> list["Tensor"]:
    """Kahn-style DFS post-order: parents before children, inputs before outputs."""
    visited: set[int] = set()
    order: list[Tensor] = []

    def visit(node: Tensor) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(root)
    return order


class Tensor:
    """Differentiable array with reverse-mode autodiff over a computation DAG."""

    data: np.ndarray
    grad: Optional[np.ndarray]
    requires_grad: bool
    parents: tuple[Tensor, ...]
    _backward: Optional[Callable[[np.ndarray], None]]
    mask: Optional[np.ndarray]

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim == 0:
            self.data = self.data.reshape(())
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float64) if requires_grad else None
        self.parents = ()
        self._backward = None

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != self.data.shape:
                raise ValueError(
                    f"mask shape {mask_arr.shape} must match data shape {self.data.shape}"
                )
            self.mask = mask_arr
        else:
            self.mask = None

    def zero_grad(self) -> None:
        """Reset this tensor's gradient buffer."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Reverse-mode autodiff via topological sort over the computation graph."""
        if not self.requires_grad and grad is None:
            return

        topo = _topological_sort(self)

        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data, dtype=np.float64)

        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar outputs")
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = np.asarray(grad, dtype=np.float64)

        # Walk from output to inputs. Each node's grad is fully accumulated from
        # children before we apply its mask and propagate to parents.
        for node in reversed(topo):
            if node.grad is None or not node.requires_grad:
                continue

            if node.mask is not None:
                node.grad = node.grad * node.mask

            if node._backward is not None:
                node._backward(node.grad)

    def sum(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import sum_op

        return sum_op(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import mean_op

        return mean_op(self, axis=axis, keepdims=keepdims)

    def relu(self) -> Tensor:
        from engine.ops import relu

        return relu(self)

    def __add__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import add

        return add(self, _as_tensor(other))

    def __radd__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import sub

        return sub(self, _as_tensor(other))

    def __rsub__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import sub

        return sub(_as_tensor(other), self)

    def __mul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import mul

        return mul(self, _as_tensor(other))

    def __rmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import div

        return div(self, _as_tensor(other))

    def __rtruediv__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import div

        return div(_as_tensor(other), self)

    def __neg__(self) -> Tensor:
        from engine.ops import neg

        return neg(self)

    def __matmul__(self, other: Union[Tensor, ArrayLike]) -> Tensor:
        from engine.ops import matmul

        return matmul(self, _as_tensor(other))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
