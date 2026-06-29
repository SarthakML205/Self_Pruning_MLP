"""Tests for the autodiff engine."""

from __future__ import annotations

import unittest

import numpy as np

from engine.grad_check import check_gradients
from engine.ops import div, matmul, mean_op, mul, relu, sub, sum_op
from engine.tensor import Tensor


class TestEngineGradCheck(unittest.TestCase):
    def test_add_elementwise(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a + b).sum()

        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.random.randn(4, 3), requires_grad=True)
        check_gradients(f, a, b)

    def test_subtract(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a - b).sum()

        a = Tensor(np.random.randn(3, 2), requires_grad=True)
        b = Tensor(np.random.randn(3, 2), requires_grad=True)
        check_gradients(f, a, b)

    def test_multiply(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a * b).sum()

        a = Tensor(np.random.randn(2, 5), requires_grad=True)
        b = Tensor(np.random.randn(2, 5), requires_grad=True)
        check_gradients(f, a, b)

    def test_divide(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a / b).sum()

        a = Tensor(np.random.randn(3, 3) + 1.0, requires_grad=True)
        b = Tensor(np.random.randn(3, 3) + 2.0, requires_grad=True)
        check_gradients(f, a, b)

    def test_matmul(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a @ b).sum()

        a = Tensor(np.random.randn(4, 5), requires_grad=True)
        b = Tensor(np.random.randn(5, 2), requires_grad=True)
        check_gradients(f, a, b)

    def test_relu(self) -> None:
        def f(a: Tensor) -> Tensor:
            return relu(a).sum()

        a = Tensor(np.random.randn(6, 4), requires_grad=True)
        check_gradients(f, a)

    def test_sum_axis(self) -> None:
        def f(a: Tensor) -> Tensor:
            return sum_op(a, axis=1).sum()

        a = Tensor(np.random.randn(5, 4), requires_grad=True)
        check_gradients(f, a)

    def test_mean_axis(self) -> None:
        def f(a: Tensor) -> Tensor:
            return mean_op(a, axis=0).sum()

        a = Tensor(np.random.randn(6, 3), requires_grad=True)
        check_gradients(f, a)

    def test_broadcast_add(self) -> None:
        """(N, D) + (D,) must sum gradients along N for the bias vector."""

        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a + b).sum()

        a = Tensor(np.random.randn(8, 4), requires_grad=True)
        b = Tensor(np.random.randn(4), requires_grad=True)
        check_gradients(f, a, b)

        a.zero_grad()
        b.zero_grad()
        out = (a + b).sum()
        out.backward()

        self.assertEqual(a.grad.shape, (8, 4))
        self.assertEqual(b.grad.shape, (4,))
        expected_b = np.sum(a.grad, axis=0)
        np.testing.assert_allclose(b.grad, expected_b, rtol=1e-10, atol=1e-10)

    def test_broadcast_multiply(self) -> None:
        def f(a: Tensor, b: Tensor) -> Tensor:
            return (a * b).sum()

        a = Tensor(np.random.randn(7, 3), requires_grad=True)
        b = Tensor(np.random.randn(3), requires_grad=True)
        check_gradients(f, a, b)

    def test_masked_gradients_are_zero(self) -> None:
        mask = np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
            ],
            dtype=bool,
        )
        weights = Tensor(np.random.randn(3, 3), requires_grad=True, mask=mask)

        def f(w: Tensor, x: Tensor) -> Tensor:
            return ((x @ w).relu().sum())

        x = Tensor(np.random.randn(5, 3), requires_grad=True)
        check_gradients(f, weights, x)

        weights.zero_grad()
        x.zero_grad()
        loss = (x @ weights).relu().sum()
        loss.backward()

        self.assertIsNotNone(weights.grad)
        assert weights.grad is not None
        dead_indices = np.where(~mask)
        self.assertTrue(np.all(weights.grad[dead_indices] == 0.0))

    def test_masked_gradients_stay_zero_through_chain(self) -> None:
        mask = np.array([[True, False], [False, True]], dtype=bool)
        w = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True, mask=mask)
        x = Tensor(np.array([[0.5, -1.0]]), requires_grad=False)

        y = relu(x @ w)
        loss = y.sum()
        loss.backward()

        assert w.grad is not None
        self.assertEqual(float(w.grad[0, 1]), 0.0)
        self.assertEqual(float(w.grad[1, 0]), 0.0)

    def test_composed_graph(self) -> None:
        def f(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
            return mean_op(relu(a @ b) * c).sum()

        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 2), requires_grad=True)
        c = Tensor(np.random.randn(3, 2) + 0.5, requires_grad=True)
        check_gradients(f, a, b, c)


if __name__ == "__main__":
    unittest.main()
