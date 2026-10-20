"""Time-split candidate evaluation for factor-expression mining."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from factor_lab.expression import Expression, evaluate_expression
from factor_lab.execution import CostModel, backtest_positions


TEMPORAL_OPERATORS = {
    "return", "log_return", "delta", "shift", "rolling_mean", "rolling_std",
    "rolling_sum", "rolling_min", "rolling_max", "zscore",
}

@dataclass(frozen=True)
class TimeSplit:
    """Chronological train/validation/holdout proportions."""

    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    holdout_ratio: float = 0.2

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.validation_ratio, self.holdout_ratio)
        if any(not math.isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("split ratios must be finite and positive")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to 1")

    def boundaries(self, observations: int, purge_bars: int = 0) -> tuple[int, int, int]:
        train, validation, holdout = self.segments(observations, purge_bars)
        return train.stop, validation.stop, holdout.stop

    def segments(self, observations: int, purge_bars: int = 0) -> tuple[slice, slice, slice]:
        """Return chronological slices with purge gaps before validation and holdout."""
        if observations < 3:
            raise ValueError("at least three observations are required for a time split")
        if purge_bars < 0:
            raise ValueError("purge_bars cannot be negative")
        validation_start = max(1, int(observations * self.train_ratio))
        holdout_start = max(
            validation_start + 1,
            int(observations * (self.train_ratio + self.validation_ratio)),
        )
        if holdout_start >= observations:
            holdout_start = observations - 1
        train_end = validation_start - purge_bars
        validation_end = holdout_start - purge_bars
        if train_end < 1 or validation_end <= validation_start or observations <= holdout_start:
            raise ValueError("purge_bars leaves an empty time segment")
        return (
            slice(0, train_end),
            slice(validation_start, validation_end),
            slice(holdout_start, observations),
        )


@dataclass(frozen=True)
class MiningResult:
    """Candidate metrics and the split boundaries used to calculate them."""

    table: pd.DataFrame
    train_end: object
    validation_end: object
    purge_bars: int


@dataclass(frozen=True)
class SelectionPolicy:
    """Explicit validation-only gates applied after candidate diagnostics."""

    min_validation_abs_rank_ic: float | None = None
    min_stability_directional_ratio: float | None = None
    min_validation_sharpe: float | None = None
    min_validation_net_return: float | None = None
    min_validation_drawdown: float | None = None
    max_validation_turnover: float | None = None
    require_fdr_pass: bool = False

    @property
    def has_gates(self) -> bool:
        return self.require_fdr_pass or any(
            value is not None
            for value in (
                self.min_validation_abs_rank_ic,
                self.min_stability_directional_ratio,
                self.min_validation_sharpe,
                self.min_validation_net_return,
                self.min_validation_drawdown,
                self.max_validation_turnover,
            )
        )

    def __post_init__(self) -> None:
        bounded = {
            "min_validation_abs_rank_ic": self.min_validation_abs_rank_ic,
            "min_stability_directional_ratio": self.min_stability_directional_ratio,
            "min_validation_sharpe": self.min_validation_sharpe,
            "min_validation_net_return": self.min_validation_net_return,
            "min_validation_drawdown": self.min_validation_drawdown,
            "max_validation_turnover": self.max_validation_turnover,
        }
        for name, value in bounded.items():
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
        if self.min_validation_abs_rank_ic is not None and self.min_validation_abs_rank_ic < 0:
            raise ValueError("min_validation_abs_rank_ic cannot be negative")
        if self.min_stability_directional_ratio is not None and not 0 <= self.min_stability_directional_ratio <= 1:
            raise ValueError("min_stability_directional_ratio must be between zero and one")
        if self.max_validation_turnover is not None and self.max_validation_turnover < 0:
            raise ValueError("max_validation_turnover cannot be negative")


def _rank_ic(factor: pd.Series, target: pd.Series) -> float:
    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
    if len(data) < 2 or data["factor"].nunique() < 2 or data["target"].nunique() < 2:
        return float("nan")
    return float(data["factor"].rank().corr(data["target"].rank()))


def _linear_ic(factor: pd.Series, target: pd.Series) -> float:
    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
    if len(data) < 2 or data["factor"].nunique() < 2 or data["target"].nunique() < 2:
        return float("nan")
    return float(data["factor"].corr(data["target"]))


def _segment_metrics(factor: pd.Series, target: pd.Series, start: int, end: int) -> dict[str, float | int]:
    segment_factor = factor.iloc[start:end]
    segment_target = target.iloc[start:end]
    complete = pd.concat(
        [segment_factor.rename("factor"), segment_target.rename("target")], axis=1
    ).dropna()
    return {
        "observations": int(len(complete)),
        "linear_ic": _linear_ic(segment_factor, segment_target),
        "rank_ic": _rank_ic(segment_factor, segment_target),
    }


def _rolling_rank_ics(
    factor: pd.Series,
    target: pd.Series,
    start: int,
    end: int,
    windows: int,
) -> list[float]:
    if windows < 1:
        raise ValueError("stability_windows must be positive")
    boundaries = np.linspace(start, end, min(windows, end - start) + 1, dtype=int)
    values: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        value = _rank_ic(factor.iloc[left:right], target.iloc[left:right])
        if pd.notna(value):
            values.append(float(value))
    return values


def _hac_mean_test(values: list[float], lags: int) -> tuple[float, float]:
    """Return a Newey-West-style mean t statistic and normal-tail p value."""
    if lags < 0:
        raise ValueError("hac_lags cannot be negative")
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan"), float("nan")
    mean = float(array.mean())
    centered = array - mean
    gamma_zero = float(np.mean(centered * centered))
    used_lags = min(lags, len(array) - 1)
    long_run = gamma_zero
    for lag in range(1, used_lags + 1):
        gamma = float(np.mean(centered[lag:] * centered[:-lag]))
        weight = 1.0 - lag / (used_lags + 1)
        long_run += 2.0 * weight * gamma
    variance_mean = max(long_run / len(array), 0.0)
    if variance_mean == 0:
        return float("nan"), float("nan")
    statistic = mean / math.sqrt(variance_mean)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return float(statistic), float(p_value)


def _strategy_metrics(
    close: pd.Series,
    factor: pd.Series,
    start: int,
    end: int,
    cost_model: CostModel,
    periods_per_year: int,
    funding_rate: pd.Series | None,
) -> dict[str, float | int]:
    """Run the common sign-based diagnostic position on one time segment."""
    segment_close = close.iloc[start:end]
    segment_factor = factor.iloc[start:end]
    desired_position = segment_factor.gt(0).astype(float).sub(segment_factor.lt(0).astype(float))
    desired_position = desired_position.where(segment_factor.notna(), 0.0)
    funding_segment = None if funding_rate is None else funding_rate.reindex(segment_close.index)
    strategy = backtest_positions(
        segment_close,
        desired_position,
        cost_model,
        periods_per_year,
        funding_rate=funding_segment,
    )
    return {
        "gross_return": strategy.gross_return,
        "net_return": strategy.net_return,
        "annualized_sharpe": strategy.annualized_sharpe,
        "max_drawdown": strategy.max_drawdown,
        "total_turnover": strategy.total_turnover,
        "rebalance_events": strategy.rebalance_events,
    }


def mine_candidates(
    expressions: Iterable[Expression],
    data: pd.DataFrame,
    target: pd.Series,
    split: TimeSplit = TimeSplit(),
    purge_bars: int = 0,
    min_observations: int = 20,
    min_unique_values: int = 5,
    max_missing_rate: float = 0.5,
    complexity_penalty: float = 0.001,
    correlation_threshold: float | None = 0.95,
    min_correlation_observations: int = 20,
    group_level: str | None = None,
    require_temporal_operator: bool = True,
    cost_model: CostModel | None = None,
    periods_per_year: int = 24 * 365,
    funding_rate: pd.Series | None = None,
    stability_windows: int = 4,
    hac_lags: int = 1,
    fdr_q: float | None = None,
) -> MiningResult:
    """Evaluate candidates without allowing holdout metrics to affect ranking."""
    if min_observations < 2:
        raise ValueError("min_observations must be at least two")
    if min_unique_values < 2:
        raise ValueError("min_unique_values must be at least two")
    if not math.isfinite(max_missing_rate) or not 0 <= max_missing_rate < 1:
        raise ValueError("max_missing_rate must be between zero and one")
    if not math.isfinite(complexity_penalty) or complexity_penalty < 0:
        raise ValueError("complexity_penalty must be finite and non-negative")
    if correlation_threshold is not None and (
        not math.isfinite(correlation_threshold) or not 0 < correlation_threshold <= 1
    ):
        raise ValueError("correlation_threshold must be in (0, 1] or None")
    if min_correlation_observations < 2:
        raise ValueError("min_correlation_observations must be at least two")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    if stability_windows < 1:
        raise ValueError("stability_windows must be positive")
    if hac_lags < 0:
        raise ValueError("hac_lags cannot be negative")
    if fdr_q is not None and (not math.isfinite(fdr_q) or not 0 < fdr_q <= 1):
        raise ValueError("fdr_q must be in (0, 1] or None")
    if funding_rate is not None and not funding_rate.index.is_unique:
        raise ValueError("funding_rate index must be unique")
    if not data.index.equals(target.index):
        raise ValueError("data and target must use the same index")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("data index must be unique and sorted")

    train_slice, validation_slice, _ = split.segments(len(data), purge_bars)
    rows: list[dict[str, object]] = []
    validation_values: dict[str, pd.Series] = {}
    for expression in expressions:
        expression = expression.normalized()
        row: dict[str, object] = {
            "expression_id": expression.expression_id,
            "formula": expression.formula(),
            "formula_zh": expression.formula_zh(),
            "canonical": expression.canonical(),
            "fields": ",".join(sorted(expression.fields)),
            "operators": ",".join(sorted(expression.operators)),
            "lookback_windows": ",".join(str(window) for window in expression.lookback_windows),
            "depth": expression.depth,
            "complexity": expression.complexity,
            "status": "rejected",
            "duplicate_of": None,
            "duplicate_correlation": float("nan"),
            "error": None,
        }
        try:
            if require_temporal_operator and not expression.operators.intersection(TEMPORAL_OPERATORS):
                row["error"] = "candidate does not contain a temporal operator"
                rows.append(row)
                continue
            factor = evaluate_expression(expression, data, group_level=group_level)
            development_factor = factor.iloc[:validation_slice.stop]
            row["missing_rate"] = float(development_factor.isna().mean())
            row["unique_values"] = int(development_factor.nunique(dropna=True))
            if row["missing_rate"] > max_missing_rate:
                row["error"] = "candidate missing rate exceeds the configured maximum"
                rows.append(row)
                continue
            if row["unique_values"] < min_unique_values:
                row["error"] = "candidate has insufficient distinct values"
                rows.append(row)
                continue
            train = _segment_metrics(factor, target, train_slice.start, train_slice.stop)
            validation = _segment_metrics(
                factor, target, validation_slice.start, validation_slice.stop
            )
            stability_ics = _rolling_rank_ics(
                factor, target, validation_slice.start, validation_slice.stop, stability_windows
            )
            hac_t, hac_p = _hac_mean_test(stability_ics, hac_lags)
            for prefix, metrics in (("train", train), ("validation", validation)):
                for name, value in metrics.items():
                    row[f"{prefix}_{name}"] = value
            row["validation_stability_folds"] = len(stability_ics)
            row["validation_stability_mean_rank_ic"] = float(np.mean(stability_ics)) if stability_ics else float("nan")
            row["validation_stability_std_rank_ic"] = float(np.std(stability_ics, ddof=1)) if len(stability_ics) > 1 else float("nan")
            row["validation_stability_positive_ratio"] = float(np.mean(np.asarray(stability_ics) > 0)) if stability_ics else float("nan")
            validation_rank_ic = float(validation["rank_ic"])
            validation_direction = np.sign(validation_rank_ic) if pd.notna(validation_rank_ic) else 0
            row["validation_abs_rank_ic"] = abs(validation_rank_ic) if pd.notna(validation_rank_ic) else float("nan")
            row["validation_stability_directional_ratio"] = (
                float(np.mean(np.sign(np.asarray(stability_ics)) == validation_direction))
                if stability_ics and validation_direction != 0
                else float("nan")
            )
            row["suggested_signal_semantics"] = (
                "direct" if validation_direction > 0 else "inverse" if validation_direction < 0 else None
            )
            row["validation_hac_t"] = hac_t
            row["validation_hac_pvalue"] = hac_p
            row["fdr_rejected"] = False
            if cost_model is not None:
                for prefix, start, end in (
                    ("train", train_slice.start, train_slice.stop),
                    ("validation", validation_slice.start, validation_slice.stop),
                ):
                    strategy_metrics = _strategy_metrics(
                        data["close"], factor, start, end, cost_model, periods_per_year, funding_rate
                    )
                    for name, value in strategy_metrics.items():
                        row[f"{prefix}_strategy_{name}"] = value
            row["selection_score"] = abs(float(validation["rank_ic"])) - complexity_penalty * expression.complexity if pd.notna(validation["rank_ic"]) else float("nan")
            row["status"] = "valid" if all(
                metrics["observations"] >= min_observations for metrics in (train, validation)
            ) else "rejected"
            if row["status"] == "rejected":
                row["error"] = "insufficient complete observations in one or more splits"
            else:
                validation_values[expression.expression_id] = factor.iloc[validation_slice]
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            row["error"] = str(error)
            row["status"] = "error"
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["expression_id", "formula", "status"])
    else:
        for column in ("selection_score", "validation_abs_rank_ic", "complexity", "expression_id"):
            if column not in table:
                table[column] = float("nan")
        status_order = {"valid": 0, "rejected": 1, "error": 2}
        table["_status_order"] = table["status"].map(status_order).fillna(3)
        table = table.sort_values(
            ["_status_order", "selection_score", "validation_abs_rank_ic", "complexity", "expression_id"],
            ascending=[True, False, False, True, True],
            na_position="last",
        ).drop(columns="_status_order").reset_index(drop=True)
        if fdr_q is not None:
            valid_indices = table.index[
                (table["status"] == "valid") & table["validation_hac_pvalue"].notna()
            ]
            p_values = table.loc[valid_indices, "validation_hac_pvalue"].astype(float)
            ordered = p_values.sort_values()
            if not ordered.empty:
                thresholds = fdr_q * np.arange(1, len(ordered) + 1) / len(ordered)
                passing = ordered.to_numpy() <= thresholds
                if passing.any():
                    cutoff = float(ordered.iloc[np.flatnonzero(passing)[-1]])
                    rejected_indices = valid_indices[p_values > cutoff]
                else:
                    rejected_indices = valid_indices
                table.loc[rejected_indices, "status"] = "fdr_rejected"
                table.loc[rejected_indices, "fdr_rejected"] = True
                table.loc[rejected_indices, "error"] = "validation HAC p-value failed FDR threshold"
        if correlation_threshold is not None:
            representatives: list[str] = []
            for row_index in table.index[table["status"] == "valid"]:
                expression_id = table.at[row_index, "expression_id"]
                candidate = validation_values[expression_id]
                duplicate_of = None
                duplicate_correlation = float("nan")
                for representative_id in representatives:
                    paired = pd.concat(
                        [candidate, validation_values[representative_id]], axis=1
                    ).dropna()
                    if len(paired) < min_correlation_observations:
                        continue
                    correlation = float(
                        paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank())
                    )
                    if pd.notna(correlation) and abs(correlation) >= correlation_threshold:
                        duplicate_of = representative_id
                        duplicate_correlation = correlation
                        break
                if duplicate_of is None:
                    representatives.append(expression_id)
                else:
                    table.at[row_index, "status"] = "duplicate"
                    table.at[row_index, "duplicate_of"] = duplicate_of
                    table.at[row_index, "duplicate_correlation"] = duplicate_correlation
                    table.at[row_index, "error"] = "validation factor correlation exceeds threshold"
            status_order = {"valid": 0, "duplicate": 1, "fdr_rejected": 2, "rejected": 3, "error": 4}
            table["_status_order"] = table["status"].map(status_order).fillna(4)
            table = table.sort_values(
                ["_status_order", "selection_score", "expression_id"],
                ascending=[True, False, True],
                na_position="last",
            ).drop(columns="_status_order").reset_index(drop=True)
    return MiningResult(
        table=table,
        train_end=data.index[train_slice.stop - 1],
        validation_end=data.index[validation_slice.stop - 1],
        purge_bars=purge_bars,
    )


def screen_candidates(table: pd.DataFrame, policy: SelectionPolicy = SelectionPolicy()) -> pd.DataFrame:
    """Apply transparent validation-only gates without changing raw candidate status."""
    result = table.copy()
    screen_status: list[str] = []
    screen_reasons: list[str] = []

    def fails_min(row: pd.Series, column: str, threshold: float | None) -> bool:
        return threshold is not None and (pd.isna(row.get(column)) or row[column] < threshold)

    for _, row in result.iterrows():
        if row.get("status") != "valid":
            screen_status.append("ineligible")
            screen_reasons.append(str(row.get("status", "unknown")))
            continue
        reasons: list[str] = []
        if fails_min(row, "validation_abs_rank_ic", policy.min_validation_abs_rank_ic):
            reasons.append("validation_abs_rank_ic")
        if fails_min(row, "validation_stability_directional_ratio", policy.min_stability_directional_ratio):
            reasons.append("stability_directional_ratio")
        if fails_min(row, "validation_strategy_annualized_sharpe", policy.min_validation_sharpe):
            reasons.append("validation_sharpe")
        if fails_min(row, "validation_strategy_net_return", policy.min_validation_net_return):
            reasons.append("validation_net_return")
        if fails_min(row, "validation_strategy_max_drawdown", policy.min_validation_drawdown):
            reasons.append("validation_drawdown")
        if policy.max_validation_turnover is not None and (
            pd.isna(row.get("validation_strategy_total_turnover"))
            or row["validation_strategy_total_turnover"] > policy.max_validation_turnover
        ):
            reasons.append("validation_turnover")
        if policy.require_fdr_pass and (
            pd.isna(row.get("validation_hac_pvalue"))
            or bool(row.get("fdr_rejected", False))
        ):
            reasons.append("fdr_not_passed")
        if reasons:
            screen_status.append("screen_rejected")
        elif policy.has_gates:
            screen_status.append("selected")
        else:
            screen_status.append("eligible")
        screen_reasons.append("" if not reasons else ",".join(reasons))

    result["screen_status"] = screen_status
    result["screen_reason"] = screen_reasons
    return result
