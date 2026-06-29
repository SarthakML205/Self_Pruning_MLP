"""Adam optimizer with momentum masking logic."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from engine.tensor import Tensor


class Adam:
    """Adam optimizer with mask-aware state and parameter updates.

    Standard Adam keeps non-zero momentum (``m``) and variance (``v``) for
    pruned weights whose gradients are zero. On the next step those stale
    moments can push a dead weight away from zero ("zombie resurrection").

    When ``param.mask`` is set we:
      1. Zero ``m`` and ``v`` at masked indices before and after each update.
      2. Re-apply the mask to ``param.data`` so dead weights stay exactly 0.0.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params: list[Tensor] = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0

        # Per-parameter first and second moment buffers keyed by object id.
        self._m: dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }
        self._v: dict[int, np.ndarray] = {
            id(p): np.zeros_like(p.data, dtype=np.float64) for p in self.params
        }

    def zero_grad(self) -> None:
        """Clear gradient buffers on all optimized parameters."""
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        """Apply one Adam update to every parameter with a gradient."""
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t

        for param in self.params:
            if param.grad is None:
                continue

            grad = np.asarray(param.grad, dtype=np.float64)
            param_id = id(param)

            m = self._m[param_id]
            v = self._v[param_id]

            # EMA of gradient and squared gradient.
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)

            # Pruned indices must not accumulate optimizer history.
            if param.mask is not None:
                m = m * param.mask
                v = v * param.mask

            self._m[param_id] = m
            self._v[param_id] = v

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data = param.data - update

            # Hard-enforce sparsity: dead weights remain exactly zero.
            if param.mask is not None:
                param.data = param.data * param.mask
