from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.factors import (
    btc_factor_suite,
    funding_rate_zscore,
    relative_volume,
)
from factor_lab.suite import run_btc_factor_suite


def btc_like_data(periods: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC"),
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.lognormal(5, 0.3, periods),
        }
    )


class FactorSuiteTest(unittest.TestCase):
    def test_relative_volume_does_not_use_future_volume(self) -> None:
        volume = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        changed = volume.copy()
        changed.iloc[4] = 10_000.0

        baseline = relative_volume(volume, window=2)
        mutated = relative_volume(changed, window=2)

        pd.testing.assert_series_equal(baseline.iloc[:4], mutated.iloc[:4])

    def test_funding_factor_is_delayed_one_bar(self) -> None:
        index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
        funding = pd.Series(np.arange(8, dtype=float), index=index)

        factor = funding_rate_zscore(funding, window=2)

        self.assertTrue(pd.isna(factor.iloc[1]))
        self.assertFalse(pd.isna(factor.iloc[2]))
        self.assertEqual(factor.index.equals(index), True)

    def test_suite_contains_funding_and_common_oos_protocol(self) -> None:
        data = btc_like_data()
        funding_index = data["timestamp"].iloc[::8]
        funding = pd.Series(np.sin(np.arange(len(funding_index))), index=funding_index)

        experiment = run_btc_factor_suite(
            data,
            momentum_window=24,
            forward_horizon=4,
            train_bars=360,
            test_bars=120,
            funding_rate=funding,
            expected_frequency="1h",
        )

        self.assertEqual(len(experiment.factors), 6)
        self.assertIn("funding_rate_zscore", experiment.summary.index)
        self.assertTrue((experiment.summary["oos_folds"] == 2).all())
        self.assertTrue((experiment.summary["observations"] > 0).all())
        self.assertIn("net_return", experiment.summary.columns)
        self.assertTrue(pd.isna(experiment.summary.loc["realized_volatility", "net_return"]))
        self.assertEqual(experiment.summary.loc["realized_volatility", "signal_semantics"], "neutral")
        self.assertEqual(experiment.correlations.shape, (6, 6))
        self.assertIn("equal_weight_all", experiment.combination_summary.index)
        self.assertEqual(len(experiment.combination_summary), 3)
        self.assertEqual(set(experiment.incremental_contribution.index), {"price_momentum", "price_reversal"})
        self.assertIn("incremental_net_return", experiment.incremental_contribution.columns)
