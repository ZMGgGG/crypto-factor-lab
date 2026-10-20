from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from factor_lab.sources.binance import (
    download_btcusdt_funding_rates,
    download_btcusdt_daily_klines,
    download_btcusdt_monthly_klines,
    load_funding_rates,
    monthly_kline_url,
    daily_kline_url,
    dates_between,
    months_between,
    parse_kline_csv,
)


def kline_csv() -> bytes:
    return (
        "1735689600000,100,102,99,101,10,1735693199999,1010,5,6,606,0\n"
        "1735693200000,101,103,100,102,11,1735696799999,1122,6,7,714,0\n"
    ).encode("utf-8")


def zipped_kline_csv(filename: str) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr(filename.replace(".zip", ".csv"), kline_csv())
    return output.getvalue()


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content


class BinanceSourceTest(unittest.TestCase):
    def test_month_range_and_url(self) -> None:
        self.assertEqual(months_between("2024-11", "2025-02"), ["2024-11", "2024-12", "2025-01", "2025-02"])
        self.assertEqual(
            monthly_kline_url("BTCUSDT", "1h", "2025-01"),
            "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01.zip",
        )
        self.assertEqual(dates_between("2025-01-01", "2025-01-02"), ["2025-01-01", "2025-01-02"])
        self.assertEqual(
            daily_kline_url("BTCUSDT", "1h", "2025-01-01"),
            "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip",
        )

    def test_downloads_daily_archive_with_checksum(self) -> None:
        filename = "BTCUSDT-1h-2025-01-01.zip"
        archive = zipped_kline_csv(filename)
        checksum = f"{sha256(archive).hexdigest()}  {filename}\n".encode()
        with TemporaryDirectory() as directory:
            with patch("factor_lab.sources.binance.urlopen", side_effect=[FakeResponse(archive), FakeResponse(checksum)]):
                data, manifest_path = download_btcusdt_daily_klines(
                    "2025-01-01", "2025-01-01", Path(directory)
                )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(data), 2)
            self.assertEqual(manifest["archives"][0]["date"], "2025-01-01")

    def test_parse_normalizes_kline_data(self) -> None:
        data = parse_kline_csv(kline_csv())

        self.assertEqual(data["timestamp"].iloc[0].isoformat(), "2025-01-01T00:00:00+00:00")
        self.assertEqual(data["close"].tolist(), [101, 102])
        self.assertEqual(data["quote_volume"].tolist(), [1010, 1122])

    def test_download_verifies_archives_and_writes_manifest(self) -> None:
        filename = "BTCUSDT-1h-2025-01.zip"
        archive = zipped_kline_csv(filename)
        checksum = f"{sha256(archive).hexdigest()}  {filename}\n".encode("utf-8")

        with TemporaryDirectory() as directory:
            with patch(
                "factor_lab.sources.binance.urlopen",
                side_effect=[FakeResponse(archive), FakeResponse(checksum)],
            ):
                data, manifest_path = download_btcusdt_monthly_klines(
                    "2025-01",
                    "2025-01",
                    Path(directory),
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 2)
            self.assertEqual(manifest["archives"][0]["sha256"], sha256(archive).hexdigest())
            self.assertTrue(manifest_path.with_name(filename).exists())
            self.assertTrue(manifest_path.with_name(f"{filename}.CHECKSUM").exists())

            with patch("factor_lab.sources.binance.urlopen", side_effect=AssertionError("cache not used")):
                cached_data, _ = download_btcusdt_monthly_klines(
                    "2025-01",
                    "2025-01",
                    Path(directory),
                )
            self.assertEqual(len(cached_data), 2)

    def test_downloads_paginated_funding_rates_and_reuses_cache(self) -> None:
        first_page = json.dumps(
            [
                {"symbol": "BTCUSDT", "fundingTime": 1609459200002, "fundingRate": "0.00010000"},
                {"symbol": "BTCUSDT", "fundingTime": 1609488000000, "fundingRate": "-0.00005000"},
            ]
        ).encode("utf-8")
        second_page = json.dumps(
            [{"symbol": "BTCUSDT", "fundingTime": 1609516800000, "fundingRate": "0.00020000"}]
        ).encode("utf-8")

        with TemporaryDirectory() as directory:
            with patch(
                "factor_lab.sources.binance.urlopen",
                side_effect=[FakeResponse(first_page), FakeResponse(second_page)],
            ):
                data, manifest_path = download_btcusdt_funding_rates(
                    "2021-01-01",
                    "2021-01-01",
                    Path(directory),
                    page_limit=2,
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 3)
            self.assertEqual(manifest["records"], 3)
            self.assertEqual(data["funding_rate"].tolist(), [0.0001, -0.00005, 0.0002])
            funding = load_funding_rates(manifest_path.with_name(manifest["normalized_csv"]))
            self.assertEqual(len(funding), 3)
            self.assertEqual(funding.index[0].isoformat(), "2021-01-01T00:00:00+00:00")

            with patch("factor_lab.sources.binance.urlopen", side_effect=AssertionError("cache not used")):
                cached_data, _ = download_btcusdt_funding_rates(
                    "2021-01-01",
                    "2021-01-01",
                    Path(directory),
                    page_limit=2,
                )
            self.assertEqual(len(cached_data), 3)
