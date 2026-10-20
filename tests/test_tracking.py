from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.execution import CostModel
from factor_lab.expression import call, field
from factor_lab.tracking import evaluate_tracking


class TrackingTest(unittest.TestCase):
    def test_tracking_uses_frozen_thresholds_on_later_data(self) -> None:
        research_index = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
        observation_index = pd.date_range("2025-01-02 06:00", periods=12, freq="h", tz="UTC")
        research = pd.DataFrame({"close": 100 + np.arange(30, dtype=float)}, index=research_index)
        observation = pd.DataFrame({"close": 130 + np.arange(12, dtype=float)}, index=observation_index)
        report = evaluate_tracking(
            call("return", field("close"), window=2), research, observation,
            pd.Series(0.0, index=observation_index), signal_semantics="direct",
            forward_horizon=2, entry_quantile=0.8, development_end=research_index[-1],
            tracking_start=observation_index[0],
            cost_model=CostModel(0, 0, 0), frozen_thresholds=(0.01, 0.02),
        )
        self.assertEqual(report.summary["entry_lower"], 0.01)
        self.assertEqual(report.summary["entry_upper"], 0.02)
        self.assertEqual(report.summary["observations"], 12)
        self.assertIn("benchmark_net_equity", report.timeline)

    def test_tracking_uses_explicit_start_even_with_overlapping_warmup_data(self) -> None:
        index = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        data = pd.DataFrame({"close": 100 + np.arange(10)}, index=index)
        report = evaluate_tracking(
            call("return", field("close"), window=2), data, data,
            pd.Series(0.0, index=index), signal_semantics="direct",
            forward_horizon=2, entry_quantile=0.8, development_end=index[4],
            tracking_start=index[5], cost_model=CostModel(0, 0, 0),
            frozen_thresholds=(0.01, 0.02), observation_bars=3,
        )
        self.assertEqual(report.summary["observation_start"], index[5].isoformat())
        self.assertEqual(report.summary["observations"], 3)

    def test_tracking_waits_when_no_data_exists_after_freeze(self) -> None:
        research_index = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
        observation_index = pd.date_range("2025-01-02", periods=5, freq="h", tz="UTC")
        research = pd.DataFrame({"close": 100 + np.arange(10)}, index=research_index)
        observation = pd.DataFrame({"close": 110 + np.arange(5)}, index=observation_index)
        report = evaluate_tracking(
            call("return", field("close"), window=2), research, observation,
            pd.Series(0.0, index=observation_index), signal_semantics="direct",
            forward_horizon=2, entry_quantile=0.8, development_end=research_index[-1],
            tracking_start=observation_index[-1] + pd.Timedelta(hours=1),
            cost_model=CostModel(), frozen_thresholds=(0.01, 0.02),
        )
        self.assertEqual(report.summary["tracking_status"], "waiting_data")
        self.assertEqual(report.summary["observations"], 0)
        self.assertIsNone(report.summary["net_return"])
