from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.data import DataContractError, validate_ohlcv_panel
from factor_lab.evaluation import evaluate_cross_sectional_factor
from factor_lab.factors import cross_sectional_rank
from factor_lab.labels import panel_forward_return
from factor_lab.pipeline import run_cross_sectional_momentum_experiment
from factor_lab.universe import liquidity_universe


def multi_asset_panel(periods: int = 16, symbols: int = 6) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    rows = []
    for symbol_number in range(symbols):
        symbol = f"ASSET{symbol_number:02d}-USDT"
        returns = 0.001 * (symbol_number + 1) + np.linspace(-0.002, 0.003, periods)
        close = 100 * (symbol_number + 1) * np.exp(np.cumsum(returns))
        for timestamp, price in zip(timestamps, close, strict=True):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": price * 0.999,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "volume": 100 + symbol_number,
                }
            )
    return pd.DataFrame(rows)


class CrossSectionalDataTest(unittest.TestCase):
    def test_rejects_duplicate_timestamp_and_symbol(self) -> None:
        panel = multi_asset_panel()
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(DataContractError, "duplicate timestamp and symbol"):
            validate_ohlcv_panel(panel)

    def test_liquidity_universe_uses_no_future_dollar_volume(self) -> None:
        panel = validate_ohlcv_panel(multi_asset_panel(periods=6, symbols=2), expected_frequency="1h")
        baseline = liquidity_universe(panel, lookback=2, minimum_dollar_volume=1.0)
        changed = panel.copy()
        changed.loc[(slice(None), "ASSET00-USDT"), "volume"] = [100, 100, 100, 100, 1_000_000, 1_000_000]
        mutated = liquidity_universe(changed, lookback=2, minimum_dollar_volume=1.0)

        early_times = panel.index.get_level_values("timestamp").unique()[:4]
        pd.testing.assert_series_equal(
            baseline.loc[(early_times, slice(None))],
            mutated.loc[(early_times, slice(None))],
        )


class CrossSectionalFactorTest(unittest.TestCase):
    def test_rank_is_calculated_within_each_timestamp(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"), ["A", "B", "C"]],
            names=["timestamp", "symbol"],
        )
        values = pd.Series([3, 1, 2, 30, 10, 20], index=index, name="signal")

        ranks = cross_sectional_rank(values)

        self.assertEqual(ranks.xs("A", level="symbol").tolist(), [1.0, 1.0])
        self.assertAlmostEqual(ranks.xs("B", level="symbol").iloc[0], 1 / 3)

    def test_panel_forward_return_has_a_tail_for_every_asset(self) -> None:
        panel = validate_ohlcv_panel(multi_asset_panel(periods=5, symbols=3))
        target = panel_forward_return(panel, horizon=2)

        for symbol in panel.index.get_level_values("symbol").unique():
            self.assertTrue(target.xs(symbol, level="symbol").iloc[-2:].isna().all())

    def test_cross_sectional_report_uses_per_timestamp_ic(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
        index = pd.MultiIndex.from_product([timestamps, list("ABCDE")], names=["timestamp", "symbol"])
        factor = pd.Series(np.tile(np.arange(5), 4), index=index, name="factor")
        target = (factor / 100).rename("target")

        report = evaluate_cross_sectional_factor(factor, target, quantiles=5)

        self.assertEqual(report.periods, 4)
        self.assertEqual(report.observations, 20)
        self.assertAlmostEqual(report.mean_rank_ic, 1.0)
        self.assertGreater(report.quantile_returns.iloc[-1], report.quantile_returns.iloc[0])

    def test_cross_sectional_pipeline_returns_report(self) -> None:
        experiment = run_cross_sectional_momentum_experiment(
            multi_asset_panel(),
            momentum_window=2,
            forward_horizon=1,
            liquidity_lookback=2,
            minimum_dollar_volume=1.0,
            expected_frequency="1h",
        )

        self.assertGreater(experiment.report.periods, 5)
        self.assertEqual(experiment.universe.name, "eligible")
