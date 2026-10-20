"""Minimal execution-aware backtesting for research validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """One-way trading cost assumptions in basis points."""

    fee_bps: float = 4.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        if min(self.fee_bps, self.half_spread_bps, self.slippage_bps) < 0:
            raise ValueError("cost assumptions cannot be negative")

    @property
    def one_way_rate(self) -> float:
        return (self.fee_bps + self.half_spread_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class StrategyBacktest:
    timeline: pd.DataFrame
    gross_return: float
    net_return: float
    annualized_sharpe: float
    max_drawdown: float
    total_turnover: float
    rebalance_events: int


def rebalance_every(desired_position: pd.Series, bars: int) -> pd.Series:
    """Hold each observed signal until the next non-overlapping rebalance."""
    if bars < 1:
        raise ValueError("rebalance bars must be positive")
    result = desired_position.astype(float).copy()
    result.iloc[np.arange(len(result)) % bars != 0] = float("nan")
    return result.ffill().fillna(0.0)


def quantile_positions(
    development_factor: pd.Series,
    live_factor: pd.Series,
    quantile: float,
) -> tuple[pd.Series, float, float]:
    """Fit entry thresholds on development data and apply them unchanged to live data."""
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1")
    fitted = development_factor.dropna()
    if len(fitted) < 2 or fitted.nunique() < 2:
        raise ValueError("development factor needs at least two varying observations")
    lower = float(fitted.quantile(1.0 - quantile))
    upper = float(fitted.quantile(quantile))
    position = live_factor.ge(upper).astype(float).sub(live_factor.le(lower).astype(float))
    return position.where(live_factor.notna(), 0.0), lower, upper


def backtest_positions(
    close: pd.Series,
    desired_position: pd.Series,
    cost_model: CostModel,
    periods_per_year: int,
    funding_rate: pd.Series | None = None,
) -> StrategyBacktest:
    """Backtest positions decided at each close and executed for the next bar."""
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")

    inputs = pd.concat(
        [close.rename("close"), desired_position.rename("desired_position")],
        axis=1,
    ).sort_index()
    if inputs["close"].isna().any() or (inputs["close"] <= 0).any():
        raise ValueError("close prices must be positive and complete")
    if (inputs["desired_position"].dropna().abs() > 1).any():
        raise ValueError("desired positions must remain between -1 and 1")

    timeline = inputs.copy()
    timeline["asset_return"] = timeline["close"].pct_change(fill_method=None).fillna(0.0)
    timeline["position"] = timeline["desired_position"].shift(1).fillna(0.0)
    timeline["turnover"] = timeline["position"].diff().abs().fillna(timeline["position"].abs())
    timeline["trading_cost"] = timeline["turnover"] * cost_model.one_way_rate

    if funding_rate is None:
        timeline["funding_rate"] = 0.0
    else:
        timeline["funding_rate"] = funding_rate.reindex(timeline.index).fillna(0.0)
    timeline["funding_cost"] = timeline["position"] * timeline["funding_rate"]
    timeline["gross_strategy_return"] = timeline["position"] * timeline["asset_return"]
    timeline["net_strategy_return"] = (
        timeline["gross_strategy_return"]
        - timeline["trading_cost"]
        - timeline["funding_cost"]
    )
    timeline["gross_equity"] = (1 + timeline["gross_strategy_return"]).cumprod()
    timeline["net_equity"] = (1 + timeline["net_strategy_return"]).cumprod()
    timeline["drawdown"] = timeline["net_equity"].div(timeline["net_equity"].cummax()).sub(1)

    net_returns = timeline["net_strategy_return"]
    volatility = net_returns.std(ddof=1)
    sharpe = float("nan") if pd.isna(volatility) or volatility == 0 else net_returns.mean() / volatility * sqrt(periods_per_year)

    return StrategyBacktest(
        timeline=timeline,
        gross_return=timeline["gross_equity"].iloc[-1] - 1,
        net_return=timeline["net_equity"].iloc[-1] - 1,
        annualized_sharpe=sharpe,
        max_drawdown=timeline["drawdown"].min(),
        total_turnover=timeline["turnover"].sum(),
        rebalance_events=int(timeline["turnover"].gt(0).sum()),
    )
