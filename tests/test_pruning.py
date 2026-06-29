"""Tests for pruning logic and mask-aware optimization."""

from __future__ import annotations

import unittest

import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from nn.module import Sequential
from optim.adam import Adam
from prune.criterion import MagnitudeCriterion, SaliencyCriterion
from prune.pruner import Pruner
from prune.schedule import CubicSchedule


class TestCubicSchedule(unittest.TestCase):
    def test_endpoints(self) -> None:
        schedule = CubicSchedule(target_sparsity=0.9, total_steps=100, initial_sparsity=0.0)
        self.assertAlmostEqual(schedule.sparsity_at(0), 0.0)
        self.assertAlmostEqual(schedule.sparsity_at(100), 0.9)

    def test_cubic_decay_shape(self) -> None:
        schedule = CubicSchedule(target_sparsity=1.0, total_steps=10, initial_sparsity=0.0)
        values = [schedule.sparsity_at(step) for step in range(11)]
        self.assertTrue(all(values[i] <= values[i + 1] for i in range(len(values) - 1)))


class TestPruningCriteria(unittest.TestCase):
    def test_magnitude_scores(self) -> None:
        data = np.array([[1.0, -2.0], [3.0, -4.0]])
        tensor = Tensor(data, requires_grad=True)
        scores = MagnitudeCriterion().compute_scores(tensor)
        np.testing.assert_allclose(scores, np.abs(data))

    def test_saliency_requires_grad(self) -> None:
        tensor = Tensor(np.ones((2, 2)), requires_grad=False)
        with self.assertRaises(ValueError):
            SaliencyCriterion().compute_scores(tensor)

    def test_saliency_scores(self) -> None:
        tensor = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        tensor.grad = np.array([[0.5, -3.0]])
        scores = SaliencyCriterion().compute_scores(tensor)
        np.testing.assert_allclose(scores, np.abs(tensor.data * tensor.grad))


class TestPruner(unittest.TestCase):
    def test_pruner_enforces_target_sparsity(self) -> None:
        model = Sequential(Linear(2, 2), Linear(2, 2))
        schedule = CubicSchedule(target_sparsity=0.5, total_steps=1, initial_sparsity=0.0)
        pruner = Pruner(model, schedule, MagnitudeCriterion())

        for param in model.parameters():
            param.grad = np.ones_like(param.data)

        pruner.step()
        self.assertAlmostEqual(pruner.actual_sparsity(), 0.5, places=2)

    def test_pruned_weights_are_exactly_zero(self) -> None:
        model = Sequential(Linear(2, 2))
        schedule = CubicSchedule(target_sparsity=0.75, total_steps=1)
        pruner = Pruner(model, schedule, MagnitudeCriterion())

        for param in model.parameters():
            param.grad = np.ones_like(param.data)

        pruner.step()
        for param in model.parameters():
            dead = ~param.mask
            self.assertTrue(np.all(param.data[dead] == 0.0))


class TestAdamMasking(unittest.TestCase):
    def test_adam_does_not_resurrect_dead_weights(self) -> None:
        mask = np.array([[True, False], [False, True]], dtype=bool)
        weight = Tensor(np.zeros((2, 2)), requires_grad=True, mask=mask)
        optimizer = Adam([weight], lr=0.1)

        for _ in range(5):
            weight.grad = np.array([[0.5, 1.0], [2.0, 0.3]])
            optimizer.step()

        dead = ~mask
        self.assertTrue(np.all(weight.data[dead] == 0.0))
        self.assertTrue(np.all(optimizer._m[id(weight)][dead] == 0.0))
        self.assertTrue(np.all(optimizer._v[id(weight)][dead] == 0.0))


class TestMaskedLinear(unittest.TestCase):
    def test_forward_respects_mask(self) -> None:
        layer = Linear(2, 1)
        layer.weight.data = np.array([[1.0], [1.0]])
        layer.weight.mask = np.array([[True], [False]], dtype=bool)
        layer.weight.data = layer.weight.data * layer.weight.mask

        x = Tensor(np.array([[1.0, 1.0]]), requires_grad=False)
        out = layer(x)
        np.testing.assert_allclose(out.data, np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
