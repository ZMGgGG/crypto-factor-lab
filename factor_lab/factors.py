"""Stage-one factor definitions."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd


def price_momentum(close: pd.Series, window: int) -> pd.Series:
    """Return trailing price momentum available at each bar close."""
    if window < 1:
        raise ValueError("window must be at least one bar")
    return close.pct_change(window, fill_method=None).rename(f"price_momentum_{window}")


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Return trailing log-return volatility without annualization."""
    if window < 2:
        raise ValueError("window must be at least two")
    log_returns = np.log(close).diff()
    return log_returns.rolling(window, min_periods=window).std().rename(
        f"realized_volatility_{window}"
    )


def relative_volume(volume: pd.Series, window: int) -> pd.Series:
    """Compare current volume with the prior window, excluding current volume."""
    if window < 1:
        raise ValueError("window must be at least one")
    prior_mean = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume.div(prior_mean).rename(f"relative_volume_{window}")


def trend_efficiency(close: pd.Series, window: int) -> pd.Series:
    """Measure directional travel divided by total absolute price travel."""
    if window < 1:
        raise ValueError("window must be at least one")
    net_move = close.sub(close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net_move.div(path).rename(f"trend_efficiency_{window}")


def funding_rate_zscore(
    funding_rate: pd.Series,
    window: int,
    target_index: pd.Index | None = None,
) -> pd.Series:
    """Use the last known funding rate, conservatively delayed by one bar."""
    if window < 2:
        raise ValueError("window must be at least two")
    if target_index is not None:
        aligned = funding_rate.reindex(target_index).ffill()
    else:
        aligned = funding_rate.copy()
    aligned = aligned.shift(1)
    mean = aligned.rolling(window, min_periods=window).mean()
    std = aligned.rolling(window, min_periods=window).std()
    return aligned.sub(mean).div(std).rename(f"funding_rate_zscore_{window}")


def btc_factor_suite(
    close: pd.Series,
    volume: pd.Series,
    window: int,
    funding_rate: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Build the starter factor suite; registry is the long-term extension point."""
    momentum = price_momentum(close, window)
    volatility = realized_volatility(close, window)
    volume = relative_volume(volume, window)
    efficiency = trend_efficiency(close, window)
    factors = {
        momentum.name: momentum,
        f"price_reversal_{window}": momentum.mul(-1).rename(f"price_reversal_{window}"),
        volatility.name: volatility,
        volume.name: volume,
        efficiency.name: efficiency,
    }
    if funding_rate is not None:
        funding = funding_rate_zscore(funding_rate, window, target_index=close.index)
        factors[funding.name] = funding
    return factors


def panel_price_momentum(panel: pd.DataFrame, window: int) -> pd.Series:
    """Return asset-level trailing momentum from a timestamp-symbol panel."""
    if window < 1:
        raise ValueError("window must be at least one bar")
    return panel["close"].groupby(level="symbol", group_keys=False).transform(
        lambda values: values.pct_change(window, fill_method=None)
    ).rename(f"price_momentum_{window}")


def cross_sectional_rank(values: pd.Series) -> pd.Series:
    """Rank each asset within its timestamp; higher values receive higher ranks."""
    if not isinstance(values.index, pd.MultiIndex) or "timestamp" not in values.index.names:
        raise ValueError("values must use a MultiIndex containing timestamp")
    return values.groupby(level="timestamp", group_keys=False).rank(pct=True).rename(
        f"{values.name}_rank" if values.name else "cross_sectional_rank"
    )
