"""Minimal single-factor diagnostics for stage-one research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorReport:
    observations: int
    linear_ic: float
    rank_ic: float
    quantile_returns: pd.Series


@dataclass(frozen=True)
class CrossSectionalFactorReport:
    observations: int
    periods: int
    mean_ic: float
    mean_rank_ic: float
    rank_ic_ir: float
    per_period_ic: pd.Series
    per_period_rank_ic: pd.Series
    quantile_returns: pd.Series


def evaluate_factor(
    factor: pd.Series,
    target: pd.Series,
    quantiles: int = 5,
) -> FactorReport:
    """Evaluate one factor against a precomputed future-return target."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")

    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
    if len(data) < quantiles:
        raise ValueError("not enough complete observations for the requested quantiles")

    bins = pd.qcut(data["factor"], q=quantiles, labels=False, duplicates="drop")
    if bins.nunique() < 2:
        raise ValueError("factor has insufficient variation for quantile analysis")

    return FactorReport(
        observations=len(data),
        linear_ic=data["factor"].corr(data["target"]),
        rank_ic=data["factor"].rank().corr(data["target"].rank()),
        quantile_returns=data.groupby(bins, observed=True)["target"].mean(),
    )


def evaluate_cross_sectional_factor(
    factor: pd.Series,
    target: pd.Series,
    quantiles: int = 5,
    minimum_assets: int | None = None,
) -> CrossSectionalFactorReport:
    """Evaluate a factor by comparing assets within each timestamp."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")
    if not isinstance(factor.index, pd.MultiIndex) or "timestamp" not in factor.index.names:
        raise ValueError("factor must use a MultiIndex containing timestamp")

    minimum_assets = minimum_assets or quantiles
    if minimum_assets < quantiles:
        raise ValueError("minimum_assets cannot be less than quantiles")

    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
    period_ics: dict[object, float] = {}
    period_rank_ics: dict[object, float] = {}
    period_quantile_returns: list[pd.Series] = []
    observations = 0

    for timestamp, period_data in data.groupby(level="timestamp", sort=True):
        if len(period_data) < minimum_assets:
            continue
        bins = pd.qcut(period_data["factor"], q=quantiles, labels=False, duplicates="drop")
        if bins.nunique() < 2:
            continue
        observations += len(period_data)
        period_ics[timestamp] = period_data["factor"].corr(period_data["target"])
        period_rank_ics[timestamp] = period_data["factor"].rank().corr(period_data["target"].rank())
        period_quantile_returns.append(
            period_data.groupby(bins, observed=True)["target"].mean().rename(timestamp)
        )

    if not period_ics:
        raise ValueError("no timestamps satisfy the cross-sectional evaluation requirements")

    ic_series = pd.Series(period_ics, name="ic")
    rank_ic_series = pd.Series(period_rank_ics, name="rank_ic")
    rank_ic_std = rank_ic_series.std(ddof=1)
    rank_ic_ir = float("nan") if pd.isna(rank_ic_std) or rank_ic_std == 0 else rank_ic_series.mean() / rank_ic_std
    quantile_returns = pd.DataFrame(period_quantile_returns).mean().rename("mean_forward_return")

    return CrossSectionalFactorReport(
        observations=observations,
        periods=len(ic_series),
        mean_ic=ic_series.mean(),
        mean_rank_ic=rank_ic_series.mean(),
        rank_ic_ir=rank_ic_ir,
        per_period_ic=ic_series,
        per_period_rank_ic=rank_ic_series,
        quantile_returns=quantile_returns,
    )
