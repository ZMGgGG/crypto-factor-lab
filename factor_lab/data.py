"""OHLCV data contracts for factor research."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
PANEL_INDEX_NAMES = ("timestamp", "symbol")


class DataContractError(ValueError):
    """Raised when market data cannot safely enter the research pipeline."""


def validate_ohlcv(frame: pd.DataFrame, expected_frequency: str | None = None) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame after enforcing the stage-one contract."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise DataContractError(f"missing required columns: {sorted(missing)}")

    data = frame.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.set_index("timestamp")
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise DataContractError("provide a timestamp column or DatetimeIndex")
    else:
        data.index = pd.to_datetime(data.index, utc=True)

    data = data.sort_index()
    if data.index.has_duplicates:
        raise DataContractError("duplicate timestamps are not allowed")
    if data.index.hasnans:
        raise DataContractError("timestamps cannot be missing")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise DataContractError("OHLCV values cannot be missing")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataContractError("prices must be positive")
    if (data["volume"] < 0).any():
        raise DataContractError("volume cannot be negative")
    if (data["high"] < data[["open", "low", "close"]].max(axis=1)).any():
        raise DataContractError("high must be at least open, low, and close")
    if (data["low"] > data[["open", "high", "close"]].min(axis=1)).any():
        raise DataContractError("low must be at most open, high, and close")

    if expected_frequency and len(data) > 1:
        expected = pd.Timedelta(expected_frequency)
        gaps = data.index.to_series().diff().iloc[1:] != expected
        if gaps.any():
            first_gap = gaps[gaps].index[0]
            raise DataContractError(f"unexpected timestamp gap before {first_gap.isoformat()}")

    return data


def validate_ohlcv_panel(
    frame: pd.DataFrame,
    expected_frequency: str | None = None,
) -> pd.DataFrame:
    """Validate long-format OHLCV data indexed by timestamp and symbol."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise DataContractError(f"missing required columns: {sorted(missing)}")

    data = frame.copy()
    if set(PANEL_INDEX_NAMES).issubset(data.columns):
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.set_index(list(PANEL_INDEX_NAMES))
    elif not isinstance(data.index, pd.MultiIndex) or tuple(data.index.names) != PANEL_INDEX_NAMES:
        raise DataContractError("provide timestamp and symbol columns or a matching MultiIndex")
    else:
        timestamps = pd.to_datetime(data.index.get_level_values("timestamp"), utc=True)
        data.index = pd.MultiIndex.from_arrays(
            [timestamps, data.index.get_level_values("symbol")],
            names=PANEL_INDEX_NAMES,
        )

    data = data.sort_index()
    if data.index.has_duplicates:
        raise DataContractError("duplicate timestamp and symbol pairs are not allowed")
    if data.index.get_level_values("timestamp").hasnans:
        raise DataContractError("timestamps cannot be missing")
    symbols = data.index.get_level_values("symbol")
    if symbols.hasnans or (symbols.astype(str).str.strip() == "").any():
        raise DataContractError("symbols cannot be missing or blank")

    for symbol, symbol_data in data.groupby(level="symbol", sort=False):
        try:
            validate_ohlcv(symbol_data.droplevel("symbol"), expected_frequency)
        except DataContractError as error:
            raise DataContractError(f"{symbol}: {error}") from error

    return data
