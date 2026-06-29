"""Tests for theoretical FLOP accounting."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate.cost import compute_theoretical_flops
from nn.linear import Linear
from nn.module import Sequential
from train.trainer import MLP


class TestTheoreticalFlops(unittest.TestCase):
    def test_dense_mlp_flops(self) -> None:
        model = MLP(in_features=64, hidden_features=128, out_features=10)
        report = compute_theoretical_flops(model)
        # Linear(64,128): 2*64*128 = 16384; Linear(128,10): 2*128*10 = 2560
        self.assertEqual(report.dense_flops, 16384 + 2560)
        self.assertEqual(report.sparse_flops, report.dense_flops)
        self.assertAlmostEqual(report.savings_fraction, 0.0)

    def test_sparse_flops_follow_active_mask(self) -> None:
        layer = Linear(4, 2)
        layer.weight.mask = np.array(
            [[True, False], [True, True], [False, False], [True, False]],
            dtype=bool,
        )
        report = compute_theoretical_flops(Sequential(layer))
        self.assertEqual(report.dense_flops, 2 * 4 * 2)
        self.assertEqual(report.sparse_flops, 2 * int(np.count_nonzero(layer.weight.mask)))


if __name__ == "__main__":
    unittest.main()
