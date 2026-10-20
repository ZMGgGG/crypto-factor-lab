from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.validation import walk_forward_factor_evaluation


class WalkForwardEvaluationTest(unittest.TestCase):
    def test_uses_strictly_ordered_rolling_windows(self) -> None:
        index = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
        factor = pd.Series(np.arange(30), index=index)
        target = factor / 100

        report = walk_forward_factor_evaluation(
            factor,
            target,
            train_bars=10,
            test_bars=5,
            step_bars=5,
            purge_bars=2,
        )

        self.assertEqual(len(report.folds), 3)
        self.assertTrue((report.folds["train_end"] < report.folds["test_start"]).all())
        self.assertTrue((report.folds["purge_bars"] == 2).all())
        first_train_end = report.folds.iloc[0]["train_end"]
        first_test_start = report.folds.iloc[0]["test_start"]
        self.assertEqual((first_test_start - first_train_end) / pd.Timedelta(hours=1), 3)
        self.assertAlmostEqual(report.mean_test_rank_ic, 1.0)
        self.assertAlmostEqual(report.positive_test_rank_ic_ratio, 1.0)

    def test_rejects_insufficient_history(self) -> None:
        values = pd.Series(range(10))
        with self.assertRaisesRegex(ValueError, "not enough observations"):
            walk_forward_factor_evaluation(values, values, train_bars=8, test_bars=5)
