from __future__ import annotations

import unittest

import pandas as pd

from factor_lab.data import DataContractError, validate_ohlcv


def valid_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )


class ValidateOhlcvTest(unittest.TestCase):
    def test_converts_timestamp_to_utc_index(self) -> None:
        data = validate_ohlcv(valid_ohlcv(), expected_frequency="1h")

        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertEqual(str(data.index.tz), "UTC")

    def test_rejects_duplicate_timestamps(self) -> None:
        data = valid_ohlcv()
        data.loc[1, "timestamp"] = data.loc[0, "timestamp"]

        with self.assertRaisesRegex(DataContractError, "duplicate timestamps"):
            validate_ohlcv(data)

    def test_rejects_timestamp_gap(self) -> None:
        data = valid_ohlcv()
        data.loc[2, "timestamp"] = data.loc[2, "timestamp"] + pd.Timedelta(hours=2)
        data.loc[3, "timestamp"] = data.loc[3, "timestamp"] + pd.Timedelta(hours=2)

        with self.assertRaisesRegex(DataContractError, "unexpected timestamp gap"):
            validate_ohlcv(data, expected_frequency="1h")
