"""Reproducible imports from Binance Public Data USD-M Futures archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from factor_lab.data import validate_ohlcv


BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
BINANCE_VISION_DAILY_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_USDM_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


@dataclass(frozen=True)
class ArchiveRecord:
    month: str
    url: str
    sha256: str
    bytes: int


def monthly_kline_url(symbol: str, interval: str, month: str) -> str:
    """Build a Binance Public Data USD-M Futures monthly-kline URL."""
    if not symbol or symbol != symbol.upper() or not symbol.isalnum():
        raise ValueError("symbol must be uppercase alphanumeric")
    try:
        datetime.strptime(month, "%Y-%m")
    except ValueError as error:
        raise ValueError("month must have YYYY-MM format") from error
    filename = f"{symbol}-{interval}-{month}.zip"
    return f"{BINANCE_VISION_BASE_URL}/{symbol}/{interval}/{filename}"


def daily_kline_url(symbol: str, interval: str, date: str) -> str:
    """Build a Binance Public Data USD-M Futures daily-kline URL."""
    if not symbol or symbol != symbol.upper() or not symbol.isalnum():
        raise ValueError("symbol must be uppercase alphanumeric")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("date must have YYYY-MM-DD format") from error
    filename = f"{symbol}-{interval}-{date}.zip"
    return f"{BINANCE_VISION_DAILY_BASE_URL}/{symbol}/{interval}/{filename}"


def months_between(start_month: str, end_month: str) -> list[str]:
    """Return inclusive YYYY-MM strings in chronological order."""
    try:
        current = datetime.strptime(start_month, "%Y-%m").replace(day=1)
        end = datetime.strptime(end_month, "%Y-%m").replace(day=1)
    except ValueError as error:
        raise ValueError("months must have YYYY-MM format") from error
    if current > end:
        raise ValueError("start_month must not be after end_month")

    months = []
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
    return months


def dates_between(start_date: str, end_date: str) -> list[str]:
    try:
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except ValueError as error:
        raise ValueError("dates must have YYYY-MM-DD format") from error
    if current > end:
        raise ValueError("start_date must not be after end_date")
    return [date.strftime("%Y-%m-%d") for date in pd.date_range(current, end, freq="D")]


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    """Normalize a Binance archive CSV to the project OHLCV contract."""
    raw = pd.read_csv(BytesIO(content), header=None)
    if raw.shape[1] != len(KLINE_COLUMNS):
        raise ValueError(f"expected {len(KLINE_COLUMNS)} kline columns, got {raw.shape[1]}")
    raw.columns = KLINE_COLUMNS
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna(subset=["open_time"])
    if raw.empty:
        raise ValueError("archive does not contain any kline records")

    timestamp_unit = "us" if raw["open_time"].median() >= 10**14 else "ms"
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["open_time"], unit=timestamp_unit, utc=True),
            "open": pd.to_numeric(raw["open"], errors="raise"),
            "high": pd.to_numeric(raw["high"], errors="raise"),
            "low": pd.to_numeric(raw["low"], errors="raise"),
            "close": pd.to_numeric(raw["close"], errors="raise"),
            "volume": pd.to_numeric(raw["volume"], errors="raise"),
            "quote_volume": pd.to_numeric(raw["quote_volume"], errors="raise"),
            "trades": pd.to_numeric(raw["trades"], errors="raise"),
        }
    )
    return validate_ohlcv(data).reset_index(names="timestamp")


def _expected_checksum(checksum_text: str, filename: str) -> str:
    for line in checksum_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == filename:
            return parts[0]
    raise ValueError(f"checksum file does not contain {filename}")


def download_btcusdt_monthly_klines(
    start_month: str,
    end_month: str,
    output_root: Path,
    interval: str = "1h",
) -> tuple[pd.DataFrame, Path]:
    """Download, verify, normalize, and manifest Binance USD-M BTCUSDT archives."""
    symbol = "BTCUSDT"
    destination = output_root / "binance_um" / symbol / interval
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    records = []

    for month in months_between(start_month, end_month):
        url = monthly_kline_url(symbol, interval, month)
        filename = url.rsplit("/", maxsplit=1)[-1]
        archive_path = destination / filename
        checksum_path = destination / f"{filename}.CHECKSUM"
        if archive_path.exists():
            archive = archive_path.read_bytes()
        else:
            archive = urlopen(url, timeout=60).read()
            archive_path.write_bytes(archive)
        if checksum_path.exists():
            checksum_text = checksum_path.read_text(encoding="utf-8")
        else:
            checksum_text = urlopen(f"{url}.CHECKSUM", timeout=60).read().decode("utf-8")
            checksum_path.write_text(checksum_text, encoding="utf-8")
        digest = sha256(archive).hexdigest()
        if digest != _expected_checksum(checksum_text, filename):
            raise ValueError(f"checksum mismatch for {filename}")

        with ZipFile(BytesIO(archive)) as zip_file:
            csv_names = [name for name in zip_file.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"expected one CSV in {filename}, got {csv_names}")
            frames.append(parse_kline_csv(zip_file.read(csv_names[0])))
        records.append(ArchiveRecord(month=month, url=url, sha256=digest, bytes=len(archive)))

    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    data = validate_ohlcv(data, expected_frequency=interval).reset_index(names="timestamp")
    normalized_path = destination / f"{symbol}-{interval}-{start_month}_to_{end_month}.csv"
    data.to_csv(normalized_path, index=False)
    manifest_path = destination / f"{symbol}-{interval}-{start_month}_to_{end_month}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": "Binance Public Data USD-M Futures monthly klines",
                "symbol": symbol,
                "interval": interval,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "normalized_csv": normalized_path.name,
                "archives": [asdict(record) for record in records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return data, manifest_path


def download_btcusdt_daily_klines(
    start_date: str,
    end_date: str,
    output_root: Path,
    interval: str = "1h",
) -> tuple[pd.DataFrame, Path]:
    """Download consecutive daily archives, stopping at the first unpublished day."""
    symbol = "BTCUSDT"
    destination = output_root / "binance_um" / symbol / interval / "daily"
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for date in dates_between(start_date, end_date):
        url = daily_kline_url(symbol, interval, date)
        filename = url.rsplit("/", maxsplit=1)[-1]
        archive_path = destination / filename
        checksum_path = destination / f"{filename}.CHECKSUM"
        try:
            archive = archive_path.read_bytes() if archive_path.exists() else urlopen(url, timeout=60).read()
        except HTTPError as error:
            if error.code == 404:
                break
            raise
        if not archive_path.exists():
            archive_path.write_bytes(archive)
        checksum_text = (
            checksum_path.read_text(encoding="utf-8")
            if checksum_path.exists()
            else urlopen(f"{url}.CHECKSUM", timeout=60).read().decode("utf-8")
        )
        if not checksum_path.exists():
            checksum_path.write_text(checksum_text, encoding="utf-8")
        digest = sha256(archive).hexdigest()
        if digest != _expected_checksum(checksum_text, filename):
            raise ValueError(f"checksum mismatch for {filename}")
        with ZipFile(BytesIO(archive)) as zip_file:
            csv_names = [name for name in zip_file.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"expected one CSV in {filename}, got {csv_names}")
            frames.append(parse_kline_csv(zip_file.read(csv_names[0])))
        records.append({"date": date, "url": url, "sha256": digest, "bytes": len(archive)})
    if not frames:
        raise ValueError("no published daily kline archives found")
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    data = validate_ohlcv(data, expected_frequency=interval).reset_index(names="timestamp")
    actual_end = records[-1]["date"]
    stem = f"{symbol}-{interval}-{start_date}_to_{actual_end}"
    normalized_path = destination / f"{stem}.csv"
    manifest_path = destination / f"{stem}.manifest.json"
    data.to_csv(normalized_path, index=False)
    manifest_path.write_text(json.dumps({
        "source": "Binance Public Data USD-M Futures daily klines",
        "symbol": symbol, "interval": interval,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "normalized_csv": normalized_path.name,
        "archives": records,
    }, indent=2) + "\n", encoding="utf-8")
    return data, manifest_path


def load_normalized_ohlcv(path: Path, expected_frequency: str | None = None) -> pd.DataFrame:
    """Load an archived normalized dataset under the stage-one data contract."""
    return validate_ohlcv(pd.read_csv(path), expected_frequency=expected_frequency)


def download_btcusdt_funding_rates(
    start_date: str,
    end_date: str,
    output_root: Path,
    page_limit: int = 1000,
) -> tuple[pd.DataFrame, Path]:
    """Download and version BTCUSDT USD-M funding-rate history."""
    if not 1 <= page_limit <= 1000:
        raise ValueError("page_limit must be between 1 and 1000")
    try:
        start = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"), tz="UTC")
        end = pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d"), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    except ValueError as error:
        raise ValueError("dates must have YYYY-MM-DD format") from error
    if start > end:
        raise ValueError("start_date must not be after end_date")

    destination = output_root / "binance_um" / "BTCUSDT" / "funding"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"BTCUSDT-funding-{start_date}_to_{end_date}"
    raw_path = destination / f"{stem}.json"
    normalized_path = destination / f"{stem}.csv"
    manifest_path = destination / f"{stem}.manifest.json"

    if raw_path.exists() and normalized_path.exists() and manifest_path.exists():
        cached = pd.read_csv(normalized_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, format="mixed")
        return cached, manifest_path

    records: list[dict[str, object]] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor <= end_ms:
        query = urlencode(
            {
                "symbol": "BTCUSDT",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": page_limit,
            }
        )
        page = json.loads(urlopen(f"{BINANCE_USDM_FUNDING_URL}?{query}", timeout=60).read())
        if not page:
            break
        records.extend(page)
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            raise ValueError("funding-rate pagination did not advance")
        cursor = next_cursor
        if len(page) < page_limit:
            break

    if not records:
        raise ValueError("funding-rate request returned no records")

    raw_bytes = (json.dumps(records, indent=2) + "\n").encode("utf-8")
    raw_path.write_bytes(raw_bytes)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([record["fundingTime"] for record in records], unit="ms", utc=True),
            "funding_rate": pd.to_numeric([record["fundingRate"] for record in records], errors="raise"),
        }
    ).drop_duplicates("timestamp").sort_values("timestamp")
    data = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)]
    data.to_csv(normalized_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "source": "Binance USD-M Futures funding-rate REST API",
                "endpoint": BINANCE_USDM_FUNDING_URL,
                "symbol": "BTCUSDT",
                "requested_start": start_date,
                "requested_end": end_date,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "records": len(data),
                "raw_json": raw_path.name,
                "raw_sha256": sha256(raw_bytes).hexdigest(),
                "normalized_csv": normalized_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return data.reset_index(drop=True), manifest_path


def load_funding_rates(path: Path, settlement_frequency: str = "1h") -> pd.Series:
    """Load sparse funding payments indexed by their UTC settlement time."""
    data = pd.read_csv(path)
    required = {"timestamp", "funding_rate"}
    if not required.issubset(data.columns):
        raise ValueError(f"missing funding columns: {sorted(required.difference(data.columns))}")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, format="mixed")
    aligned = data["timestamp"].dt.round(settlement_frequency)
    if (data["timestamp"].sub(aligned).abs() > pd.Timedelta(minutes=1)).any():
        raise ValueError("funding timestamps are too far from a settlement boundary")
    data["timestamp"] = aligned
    if data["timestamp"].duplicated().any():
        raise ValueError("duplicate funding timestamps are not allowed")
    return pd.Series(
        pd.to_numeric(data["funding_rate"], errors="raise").to_numpy(),
        index=data["timestamp"],
        name="funding_rate",
    ).sort_index()
