from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.factors import price_momentum
from factor_lab.labels import forward_return
from factor_lab.pipeline import run_momentum_experiment
from tests.test_data import valid_ohlcv


class FactorPipelineTest(unittest.TestCase):
    def test_future_label_keeps_unavailable_tail_as_missing(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        target = forward_return(close, horizon=2)

        self.assertTrue(target.iloc[-2:].isna().all())
        self.assertAlmostEqual(target.iloc[0], 0.02)

    def test_factor_is_unchanged_when_only_future_prices_change(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        changed = close.copy()
        changed.iloc[4:] = [1000.0, 2000.0]

        baseline = price_momentum(close, window=2)
        mutated = price_momentum(changed, window=2)

        np.testing.assert_allclose(baseline.iloc[:4], mutated.iloc[:4], equal_nan=True)

    def test_end_to_end_experiment_returns_diagnostics(self) -> None:
        data = pd.concat([valid_ohlcv()] * 8, ignore_index=True)
        data["timestamp"] = pd.date_range("2025-01-01", periods=len(data), freq="h", tz="UTC")
        data["close"] = np.linspace(100.0, 160.0, len(data))
        data["open"] = data["close"] - 0.5
        data["high"] = data["close"] + 1.0
        data["low"] = data["close"] - 1.0

        experiment = run_momentum_experiment(data, 2, 1, expected_frequency="1h")

        self.assertGreater(experiment.report.observations, 5)
        self.assertEqual(experiment.factor.name, "price_momentum_2")
