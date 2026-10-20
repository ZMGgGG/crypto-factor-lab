from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.expression import Expression, call, evaluate_expression, field
from factor_lab.factors import price_momentum, realized_volatility, relative_volume, trend_efficiency


class ExpressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 102.0, 104.0],
                "volume": [10.0, 12.0, 9.0, 15.0, 20.0],
                "high": [101.0, 102.0, 104.0, 103.0, 105.0],
                "low": [99.0, 100.0, 102.0, 101.0, 103.0],
            },
            index=pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
        )

    def test_expression_is_serializable_normalized_and_stable(self) -> None:
        left = call("add", field("close"), field("volume"))
        right = call("add", field("volume"), field("close"))

        self.assertEqual(left.canonical(), right.canonical())
        self.assertEqual(left.expression_id, right.expression_id)
        self.assertEqual(Expression.from_dict(left.to_dict()), left.normalized())
        self.assertEqual(left.fields, frozenset({"close", "volume"}))
        self.assertEqual(left.complexity, 3)
        self.assertEqual(left.depth, 2)
        self.assertEqual(left.operators, frozenset({"add"}))
        self.assertEqual(call("return", field("close"), window=24).lookback_windows, (24,))
        self.assertEqual(
            call("return", field("close"), window=24).formula_zh(),
            "24小时收益率(收盘价)",
        )
        self.assertEqual(
            call(
                "div",
                field("volume"),
                call("rolling_mean", call("shift", field("volume"), window=1), window=24),
            ).formula_zh(),
            "(成交量 ÷ 24小时滚动均值(1小时滞后(成交量)))",
        )

    def test_return_and_relative_volume_use_only_past_window(self) -> None:
        momentum = call("return", field("close"), window=2)
        relative_volume = call(
            "div",
            field("volume"),
            call("rolling_mean", call("shift", field("volume"), window=1), window=2),
        )

        momentum_values = evaluate_expression(momentum, self.data)
        volume_values = evaluate_expression(relative_volume, self.data)
        self.assertTrue(pd.isna(momentum_values.iloc[0:2]).all())
        self.assertAlmostEqual(momentum_values.iloc[2], 0.03)
        self.assertTrue(pd.isna(volume_values.iloc[0:2]).all())
        self.assertAlmostEqual(volume_values.iloc[2], 9 / 11)
        self.assertAlmostEqual(volume_values.iloc[3], 15 / 10.5)

        changed = self.data.copy()
        changed.loc[changed.index[4], "close"] = 10_000
        changed.loc[changed.index[4], "volume"] = 10_000
        pd.testing.assert_series_equal(
            momentum_values.iloc[:4], evaluate_expression(momentum, changed).iloc[:4]
        )
        pd.testing.assert_series_equal(
            volume_values.iloc[:4], evaluate_expression(relative_volume, changed).iloc[:4]
        )

    def test_invalid_operations_and_zero_division_are_rejected_safely(self) -> None:
        with self.assertRaises(ValueError):
            call("return", field("close"), window=0)
        with self.assertRaises(ValueError):
            Expression.from_dict({"op": "python", "args": []})

        expression = call("div", field("close"), call("sub", field("close"), field("close")))
        result = evaluate_expression(expression, self.data)
        self.assertTrue(result.isna().all())
        self.assertFalse(np.isinf(result.to_numpy(dtype=float)).any())

    def test_expressions_reproduce_existing_handwritten_factors(self) -> None:
        close = self.data["close"]
        volume = self.data["volume"]
        expressions = {
            "momentum": call("return", field("close"), window=2),
            "volatility": call(
                "rolling_std",
                call("log_return", field("close"), window=1),
                window=2,
            ),
            "relative_volume": call(
                "div",
                field("volume"),
                call("rolling_mean", call("shift", field("volume"), window=1), window=2),
            ),
            "trend_efficiency": call(
                "div",
                call("abs", call("delta", field("close"), window=2)),
                call(
                    "rolling_sum",
                    call("abs", call("delta", field("close"), window=1)),
                    window=2,
                ),
            ),
        }
        expected = {
            "momentum": price_momentum(close, 2),
            "volatility": realized_volatility(close, 2),
            "relative_volume": relative_volume(volume, 2),
            "trend_efficiency": trend_efficiency(close, 2),
        }

        for name, expression in expressions.items():
            np.testing.assert_allclose(
                evaluate_expression(expression, self.data),
                expected[name],
                equal_nan=True,
                err_msg=name,
            )

    def test_time_operations_can_be_grouped_by_asset(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"), ["BTC", "ETH"]],
            names=["timestamp", "symbol"],
        )
        panel = pd.DataFrame(
            {"close": [100.0, 10.0, 110.0, 20.0, 121.0, 40.0]},
            index=index,
        )
        expression = call("return", field("close"), window=1)

        result = evaluate_expression(expression, panel, group_level="symbol")

        self.assertTrue(result.groupby(level="symbol").head(1).isna().all())
        self.assertAlmostEqual(result.loc[(index.levels[0][1], "BTC")], 0.1)
        self.assertAlmostEqual(result.loc[(index.levels[0][1], "ETH")], 1.0)
