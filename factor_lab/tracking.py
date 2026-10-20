"""Reproducible observation-period evaluation for frozen factor candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factor_lab.execution import CostModel, backtest_positions, quantile_positions, rebalance_every
from factor_lab.expression import Expression, evaluate_expression


@dataclass(frozen=True)
class TrackingReport:
    summary: dict[str, object]
    timeline: pd.DataFrame


def evaluate_tracking(
    expression: Expression,
    research_data: pd.DataFrame,
    observation_data: pd.DataFrame,
    observation_funding: pd.Series,
    *,
    signal_semantics: str,
    forward_horizon: int,
    entry_quantile: float,
    development_end: object,
    tracking_start: object,
    cost_model: CostModel,
    tracking_end: object | None = None,
    observation_bars: int = 24 * 90,
    frozen_thresholds: tuple[float, float] | None = None,
) -> TrackingReport:
    if signal_semantics not in {"direct", "inverse"}:
        raise ValueError("tracking requires direct or inverse signal semantics")
    combined = pd.concat([research_data, observation_data]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    factor = evaluate_expression(expression, combined)
    multiplier = -1.0 if signal_semantics == "inverse" else 1.0
    aligned = factor.mul(multiplier)
    development = aligned.loc[:development_end]
    frozen_at = pd.Timestamp(tracking_start)
    if frozen_at.tzinfo is None:
        frozen_at = frozen_at.tz_localize("UTC")
    else:
        frozen_at = frozen_at.tz_convert("UTC")
    planned_end = pd.Timestamp(tracking_end) if tracking_end is not None else frozen_at + pd.Timedelta(hours=observation_bars - 1)
    if planned_end.tzinfo is None:
        planned_end = planned_end.tz_localize("UTC")
    else:
        planned_end = planned_end.tz_convert("UTC")
    tracked_data = combined.loc[(combined.index >= frozen_at) & (combined.index <= planned_end)]
    if tracked_data.empty:
        lower, upper = frozen_thresholds or (
            float(development.dropna().quantile(1.0 - entry_quantile)),
            float(development.dropna().quantile(entry_quantile)),
        )
        return TrackingReport(
            summary={
                "tracking_status": "waiting_data",
                "used_data_end": pd.Timestamp(development_end).isoformat(),
                "planned_observation_start": frozen_at.isoformat(),
                "planned_observation_end": planned_end.isoformat(),
                "planned_observation_bars": observation_bars,
                "data_available_through": combined.index[-1].isoformat(),
                "observation_start": None,
                "observation_end": None,
                "observations": 0,
                "signal_semantics": signal_semantics,
                "forward_horizon": forward_horizon,
                "entry_quantile": entry_quantile,
                "entry_lower": lower,
                "entry_upper": upper,
                "active_position_ratio": None,
                "gross_return": None,
                "net_return": None,
                "annualized_sharpe": None,
                "max_drawdown": None,
                "total_turnover": None,
                "buy_hold_net_return": None,
                "excess_net_return_vs_buy_hold": None,
            },
            timeline=pd.DataFrame(),
        )
    observed = aligned.reindex(tracked_data.index)
    if frozen_thresholds is None:
        desired, lower, upper = quantile_positions(development, observed, entry_quantile)
    else:
        lower, upper = frozen_thresholds
        desired = observed.ge(upper).astype(float).sub(observed.le(lower).astype(float))
        desired = desired.where(observed.notna(), 0.0)
    desired = rebalance_every(desired, forward_horizon)
    strategy = backtest_positions(
        tracked_data["close"], desired, cost_model, 24 * 365,
        observation_funding.reindex(tracked_data.index),
    )
    benchmark = backtest_positions(
        tracked_data["close"], pd.Series(1.0, index=tracked_data.index),
        cost_model, 24 * 365, observation_funding.reindex(tracked_data.index),
    )
    timeline = strategy.timeline.copy()
    timeline["factor_value"] = observed
    timeline["benchmark_net_equity"] = benchmark.timeline["net_equity"]
    return TrackingReport(
        summary={
            "tracking_status": "completed" if tracked_data.index[-1] >= planned_end else "observing",
            "used_data_end": pd.Timestamp(development_end).isoformat(),
            "planned_observation_start": frozen_at.isoformat(),
            "planned_observation_end": planned_end.isoformat(),
            "planned_observation_bars": observation_bars,
            "data_available_through": combined.index[-1].isoformat(),
            "observation_start": tracked_data.index[0].isoformat(),
            "observation_end": tracked_data.index[-1].isoformat(),
            "observations": len(tracked_data),
            "signal_semantics": signal_semantics,
            "forward_horizon": forward_horizon,
            "entry_quantile": entry_quantile,
            "entry_lower": lower,
            "entry_upper": upper,
            "active_position_ratio": float(strategy.timeline["position"].ne(0).mean()),
            "gross_return": strategy.gross_return,
            "net_return": strategy.net_return,
            "annualized_sharpe": strategy.annualized_sharpe,
            "max_drawdown": strategy.max_drawdown,
            "total_turnover": strategy.total_turnover,
            "buy_hold_net_return": benchmark.net_return,
            "excess_net_return_vs_buy_hold": strategy.net_return - benchmark.net_return,
        },
        timeline=timeline,
    )
