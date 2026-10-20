"""Future labels kept separate from factor construction."""

from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return the close-to-close future return; tail values remain unavailable."""
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")
    return (close.shift(-horizon).div(close).sub(1)).rename(f"forward_return_{horizon}")


def panel_forward_return(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Return future close-to-close returns independently for every asset."""
    if horizon < 1:
        raise ValueError("horizon must be at least one bar")
    close = panel["close"]
    future_close = close.groupby(level="symbol", group_keys=False).shift(-horizon)
    return future_close.div(close).sub(1).rename(f"forward_return_{horizon}")
