"""Historical, point-in-time asset-universe construction."""

from __future__ import annotations

import pandas as pd


def liquidity_universe(
    panel: pd.DataFrame,
    lookback: int,
    minimum_dollar_volume: float,
) -> pd.Series:
    """Mark assets eligible using only their current and prior dollar volume."""
    if lookback < 1:
        raise ValueError("lookback must be at least one bar")
    if minimum_dollar_volume < 0:
        raise ValueError("minimum_dollar_volume cannot be negative")

    dollar_volume = panel["close"] * panel["volume"]
    rolling_volume = dollar_volume.groupby(level="symbol", group_keys=False).transform(
        lambda values: values.rolling(lookback, min_periods=lookback).mean()
    )
    return rolling_volume.ge(minimum_dollar_volume).rename("eligible")
