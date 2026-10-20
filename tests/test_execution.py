from __future__ import annotations

import unittest

import pandas as pd

from factor_lab.execution import CostModel, backtest_positions, quantile_positions, rebalance_every


class ExecutionBacktestTest(unittest.TestCase):
    def test_quantile_positions_fit_thresholds_only_on_development_data(self) -> None:
        development = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        live = pd.Series([-100.0, 2.5, 100.0])

        position, lower, upper = quantile_positions(development, live, 0.8)
        changed_position, changed_lower, changed_upper = quantile_positions(
            development, live.mul(10), 0.8
        )

        self.assertEqual(position.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual((lower, upper), (changed_lower, changed_upper))
        self.assertNotEqual(position.tolist(), changed_position.tolist())

    def test_rebalance_every_holds_signal_for_requested_bars(self) -> None:
        desired = pd.Series([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])

        held = rebalance_every(desired, 2)

        self.assertEqual(held.tolist(), [1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

    def test_position_is_delayed_and_cost_applies_to_turnover(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        close = pd.Series([100.0, 110.0, 121.0], index=index)
        desired = pd.Series([1.0, 1.0, 1.0], index=index)

        result = backtest_positions(
            close,
            desired,
            CostModel(fee_bps=10, half_spread_bps=0, slippage_bps=0),
            periods_per_year=24 * 365,
        )

        self.assertEqual(result.timeline["position"].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(result.timeline["turnover"].tolist(), [0.0, 1.0, 0.0])
        self.assertAlmostEqual(result.gross_return, 0.21)
        self.assertAlmostEqual(result.net_return, 0.2089)

    def test_positive_funding_is_paid_by_long_position(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        close = pd.Series([100.0, 100.0, 100.0], index=index)
        desired = pd.Series([1.0, 1.0, 1.0], index=index)
        funding = pd.Series([0.0, 0.001, 0.0], index=index)

        result = backtest_positions(
            close,
            desired,
            CostModel(fee_bps=0, half_spread_bps=0, slippage_bps=0),
            periods_per_year=24 * 365,
            funding_rate=funding,
        )

        self.assertAlmostEqual(result.net_return, -0.001)

    def test_rejects_position_outside_unit_notional(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            backtest_positions(
                pd.Series([100.0, 101.0], index=index),
                pd.Series([0.0, 1.5], index=index),
                CostModel(),
                periods_per_year=24 * 365,
            )
