"""CrossEntropy loss implementation."""

from __future__ import annotations

from typing import Union

import numpy as np

from engine.tensor import Tensor, _accumulate_grad

TargetArray = Union[np.ndarray, list[int]]


class CrossEntropyLoss:
    """Numerically stable softmax cross-entropy over class logits.

    For logits ``z`` and one-hot target class index ``y`` per sample:

        loss = -log( exp(z_y) / sum_j exp(z_j) )

    Naive ``exp(z)`` overflows when ``z`` is large. We subtract the per-row
    maximum before exponentiation (log-sum-exp trick):

        log_sum_exp(z) = max(z) + log( sum_j exp(z_j - max(z)) )

    so all exponentials stay in ``(0, 1]`` and the loss remains finite.
    """

    def __call__(self, logits: Tensor, targets: TargetArray) -> Tensor:
        return self.forward(logits, targets)

    def forward(self, logits: Tensor, targets: TargetArray) -> Tensor:
        logits_data = np.asarray(logits.data, dtype=np.float64)
        if logits_data.ndim != 2:
            raise ValueError(
                f"logits must be 2-D (batch, classes), got shape {logits_data.shape}"
            )

        targets_arr = np.asarray(targets, dtype=np.int64).reshape(-1)
        batch_size, num_classes = logits_data.shape
        if targets_arr.shape[0] != batch_size:
            raise ValueError(
                f"targets length {targets_arr.shape[0]} != batch size {batch_size}"
            )

        # --- Stable log-softmax (per row) ---
        # max_i prevents exp overflow; subtracting it does not change softmax.
        row_max = np.max(logits_data, axis=1, keepdims=True)
        shifted = logits_data - row_max
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        log_sum_exp = row_max + np.log(sum_exp)

        batch_idx = np.arange(batch_size)
        correct_logits = logits_data[batch_idx, targets_arr]
        per_sample_loss = log_sum_exp.reshape(-1) - correct_logits
        loss_value = float(np.mean(per_sample_loss))

        out = Tensor(np.array(loss_value), requires_grad=logits.requires_grad)

        if logits.requires_grad:
            out.parents = (logits,)

            # softmax(z) = exp(z - max) / sum(exp(z - max))
            softmax = exp_shifted / sum_exp

            # d/dz mean( -log p_y ) = (softmax - one_hot(y)) / N
            grad_logits = softmax.copy()
            grad_logits[batch_idx, targets_arr] -= 1.0
            grad_logits /= batch_size

            def _backward(grad_out: np.ndarray) -> None:
                _accumulate_grad(logits, grad_out * grad_logits)

            out._backward = _backward

        return out
