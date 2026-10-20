from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.expression import call, field
from factor_lab.execution import CostModel
from factor_lab.mining import SelectionPolicy, TimeSplit, mine_candidates, screen_candidates


class MiningTest(unittest.TestCase):
    def setUp(self) -> None:
        periods = 120
        index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
        close = pd.Series(100.0 + np.arange(periods, dtype=float), index=index)
        self.data = pd.DataFrame({"close": close, "volume": np.linspace(10, 20, periods)}, index=index)
        self.target = close.shift(-1).div(close).sub(1).rename("forward_return_1")

    def test_split_boundaries_are_chronological_and_purged(self) -> None:
        split = TimeSplit(0.5, 0.25, 0.25)
        train_end, validation_end, end = split.boundaries(100, purge_bars=4)

        self.assertEqual((train_end, validation_end, end), (46, 71, 100))
        train, validation, holdout = split.segments(100, purge_bars=4)
        self.assertEqual((train.start, train.stop), (0, 46))
        self.assertEqual((validation.start, validation.stop), (50, 71))
        self.assertEqual((holdout.start, holdout.stop), (75, 100))
        self.assertLess(train_end, validation_end)
        self.assertEqual(end, 100)

    def test_mining_ranks_validation_without_using_holdout(self) -> None:
        expressions = (
            call("return", field("close"), window=1),
            call("rolling_mean", field("volume"), window=4),
            call("return", field("missing"), window=1),
            call("neg", field("close")),
        )
        result = mine_candidates(
            expressions,
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            purge_bars=1,
            min_observations=10,
            correlation_threshold=None,
        )

        self.assertEqual(len(result.table), 4)
        self.assertEqual(result.purge_bars, 1)
        self.assertIn("selection_score", result.table.columns)
        valid = result.table[result.table["status"] == "valid"]
        self.assertEqual(len(valid), 2)
        self.assertFalse(any(column.startswith("holdout_") for column in result.table.columns))
        errors = set(result.table[result.table["status"] == "error"]["error"])
        rejections = set(result.table[result.table["status"] == "rejected"]["error"])
        self.assertIn("'expression requires missing field: missing'", errors)
        self.assertIn("candidate does not contain a temporal operator", rejections)

    def test_holdout_changes_do_not_affect_mining_output(self) -> None:
        expression = call("return", field("close"), window=1)
        result = mine_candidates(
            (expression,),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            complexity_penalty=0.01,
            correlation_threshold=None,
        )
        row = result.table.iloc[0]
        expected = abs(row["validation_rank_ic"]) - 0.01 * row["complexity"]
        self.assertAlmostEqual(row["selection_score"], expected)

        changed_target = self.target.copy()
        changed_target.iloc[90:] = np.linspace(-10.0, 10.0, len(changed_target.iloc[90:]))
        changed = mine_candidates(
            (expression,),
            self.data,
            changed_target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            complexity_penalty=0.01,
            correlation_threshold=None,
        ).table.iloc[0]
        self.assertAlmostEqual(row["selection_score"], changed["selection_score"])
        pd.testing.assert_series_equal(row, changed)

        changed_data = self.data.copy()
        changed_data.iloc[90:, changed_data.columns.get_loc("close")] *= 100.0
        changed_data_result = mine_candidates(
            (expression,),
            changed_data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            complexity_penalty=0.01,
            correlation_threshold=None,
        ).table.iloc[0]
        pd.testing.assert_series_equal(row, changed_data_result)

    def test_invalid_split_and_index_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TimeSplit(0.5, 0.5, 0.5)
        shuffled = self.data.iloc[::-1]
        shuffled_target = self.target.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "sorted"):
            mine_candidates((call("return", field("close"), window=1),), shuffled, shuffled_target)

    def test_validation_correlation_marks_lower_ranked_candidate_duplicate(self) -> None:
        expressions = (
            call("return", field("close"), window=1),
            call("mul", call("return", field("close"), window=1), field("close")),
        )
        result = mine_candidates(
            expressions,
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=0.99,
            min_correlation_observations=10,
        )

        self.assertEqual((result.table["status"] == "valid").sum(), 1)
        duplicate = result.table[result.table["status"] == "duplicate"].iloc[0]
        representative = result.table[result.table["status"] == "valid"].iloc[0]
        self.assertEqual(duplicate["duplicate_of"], representative["expression_id"])
        self.assertGreaterEqual(abs(duplicate["duplicate_correlation"]), 0.99)

    def test_cost_metrics_are_reported_for_diagnostic_sign_position(self) -> None:
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
            cost_model=CostModel(fee_bps=10, half_spread_bps=0, slippage_bps=0),
        )

        row = result.table.iloc[0]
        self.assertEqual(row["status"], "valid")
        self.assertIn("validation_strategy_net_return", result.table.columns)
        self.assertIn("validation_strategy_total_turnover", result.table.columns)
        self.assertGreaterEqual(row["validation_strategy_total_turnover"], 0)

    def test_rolling_stability_and_hac_metrics_are_reported(self) -> None:
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
            stability_windows=3,
            hac_lags=1,
        )

        row = result.table.iloc[0]
        self.assertEqual(row["validation_stability_folds"], 3)
        self.assertGreaterEqual(row["validation_stability_positive_ratio"], 0)
        self.assertGreaterEqual(row["validation_stability_directional_ratio"], 0)
        self.assertIn(row["suggested_signal_semantics"], {"direct", "inverse"})
        self.assertIn("validation_hac_t", result.table.columns)
        self.assertIn("validation_hac_pvalue", result.table.columns)

    def test_negative_ic_candidate_is_ranked_by_absolute_ic_and_directional_stability(self) -> None:
        inverse = call("neg", call("return", field("close"), window=1))
        result = mine_candidates(
            (inverse,),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
            stability_windows=4,
        )
        row = result.table.iloc[0]

        self.assertLess(row["validation_rank_ic"], 0)
        self.assertEqual(row["suggested_signal_semantics"], "inverse")
        self.assertEqual(row["validation_abs_rank_ic"], abs(row["validation_rank_ic"]))
        self.assertGreaterEqual(row["validation_stability_directional_ratio"], 0.75)
        screened = screen_candidates(
            result.table,
            SelectionPolicy(
                min_validation_abs_rank_ic=0.0,
                min_stability_directional_ratio=0.75,
            ),
        )
        self.assertEqual(screened.iloc[0]["screen_status"], "selected")

    def test_fdr_can_mark_candidates_without_calculating_holdout_metrics(self) -> None:
        expression = call("return", field("close"), window=1)
        result = mine_candidates(
            (expression,),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
            fdr_q=0.05,
        )

        self.assertIn("fdr_rejected", result.table.columns)
        self.assertFalse(any(column.startswith("holdout_") for column in result.table.columns))

    def test_selection_policy_records_explicit_rejection_reasons(self) -> None:
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
            cost_model=CostModel(fee_bps=10, half_spread_bps=0, slippage_bps=0),
        )
        screened = screen_candidates(
            result.table,
            SelectionPolicy(min_validation_abs_rank_ic=1.1, max_validation_turnover=0.0),
        )

        row = screened.iloc[0]
        self.assertEqual(row["screen_status"], "screen_rejected")
        self.assertIn("validation_abs_rank_ic", row["screen_reason"])
        self.assertIn("validation_turnover", row["screen_reason"])

    def test_selection_policy_ignores_unrelated_columns(self) -> None:
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
        )
        policy = SelectionPolicy(min_validation_abs_rank_ic=0.0)
        first = screen_candidates(result.table, policy)
        changed = result.table.copy()
        changed.loc[:, "unused_metric"] = -999.0
        second = screen_candidates(changed, policy)
        self.assertEqual(first["screen_status"].tolist(), second["screen_status"].tolist())

    def test_empty_policy_marks_candidate_eligible_not_selected(self) -> None:
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            self.target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
        )
        screened = screen_candidates(result.table)
        self.assertEqual(screened.iloc[0]["screen_status"], "eligible")

    def test_missing_holdout_target_does_not_control_candidate_validity(self) -> None:
        target = self.target.copy()
        target.iloc[90:] = np.nan
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            self.data,
            target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
        )
        self.assertEqual(result.table.iloc[0]["status"], "valid")
