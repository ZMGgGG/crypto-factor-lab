"""Build the latest complete observation snapshot from monthly and daily archives."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from factor_lab.data import validate_ohlcv
from factor_lab.sources.binance import (
    download_btcusdt_daily_klines,
    download_btcusdt_funding_rates,
    download_btcusdt_monthly_klines,
)


def _latest_ohlcv_snapshot(
    project_root: Path,
    *,
    start_month: str,
    as_of: datetime,
) -> dict[str, object]:
    """Download a complete monthly + current daily snapshot without changing config."""
    today = pd.Timestamp(as_of).tz_convert("UTC")
    current_month_start = today.normalize().replace(day=1)
    previous_month_end = current_month_start - pd.Timedelta(days=1)
    monthly_end = previous_month_end.strftime("%Y-%m")
    output_root = project_root / "data/raw"
    monthly, monthly_manifest = download_btcusdt_monthly_klines(
        start_month, monthly_end, output_root
    )
    frames = [monthly]
    daily_manifest = None
    yesterday = today.normalize() - pd.Timedelta(days=1)
    if yesterday >= current_month_start:
        try:
            daily, daily_manifest = download_btcusdt_daily_klines(
                current_month_start.strftime("%Y-%m-%d"),
                yesterday.strftime("%Y-%m-%d"),
                output_root,
            )
            frames.append(daily)
        except ValueError as error:
            if "no published daily" not in str(error):
                raise
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        "timestamp", keep="last"
    )
    combined = validate_ohlcv(
        combined.sort_values("timestamp"), expected_frequency="1h"
    ).reset_index(names="timestamp")
    actual_end_date = combined["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    destination = output_root / "binance_um/BTCUSDT/1h"
    stem = f"BTCUSDT-1h-{start_month}_to_{actual_end_date}"
    csv_path = destination / f"{stem}.csv"
    manifest_path = destination / f"{stem}.manifest.json"
    combined.to_csv(csv_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "source": "Binance monthly plus current-month daily archives",
                "generated_at": datetime.now(UTC).isoformat(),
                "rows": len(combined),
                "first_timestamp": combined["timestamp"].iloc[0].isoformat(),
                "last_timestamp": combined["timestamp"].iloc[-1].isoformat(),
                "csv_sha256": sha256(csv_path.read_bytes()).hexdigest(),
                "monthly_manifest": str(monthly_manifest.relative_to(project_root)),
                "daily_manifest": str(daily_manifest.relative_to(project_root))
                if daily_manifest
                else None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    funding, funding_manifest = download_btcusdt_funding_rates(
        f"{start_month}-01", actual_end_date, output_root
    )
    funding_csv = funding_manifest.with_name(
        funding_manifest.name.replace(".manifest.json", ".csv")
    )
    return {
        "dataset_id": f"btcusdt_1h_{start_month}_to_{actual_end_date}",
        "bars": len(combined),
        "funding_records": len(funding),
        "first_timestamp": combined["timestamp"].iloc[0].isoformat(),
        "last_timestamp": combined["timestamp"].iloc[-1].isoformat(),
        "ohlcv": str(csv_path.relative_to(project_root)),
        "manifest": str(manifest_path.relative_to(project_root)),
        "funding": str(funding_csv.relative_to(project_root)),
    }


def _write_dataset_config(
    project_root: Path,
    *,
    research: dict[str, object],
    observation: dict[str, object] | None,
) -> None:
    config_path = project_root / "data/state/datasets.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "research": research}
    if observation is not None:
        payload["observation"] = observation
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def update_research_snapshot(
    project_root: Path,
    *,
    start_month: str = "2021-01",
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Build the current research snapshot while preserving previous files."""
    result = _latest_ohlcv_snapshot(
        project_root.resolve(), start_month=start_month, as_of=as_of or datetime.now(UTC)
    )
    from factor_lab.datasets import load_dataset_config

    current = load_dataset_config(project_root.resolve())
    observation = None
    if current.observation:
        observation = {
            "dataset_id": current.observation.dataset_id,
            "ohlcv": str(current.observation.ohlcv.relative_to(project_root.resolve())),
            "manifest": str(current.observation.manifest.relative_to(project_root.resolve())),
            "funding": str(current.observation.funding.relative_to(project_root.resolve())),
        }
    _write_dataset_config(
        project_root.resolve(),
        research={key: result[key] for key in ("dataset_id", "ohlcv", "manifest", "funding")},
        observation=observation,
    )
    return result


def update_observation_snapshot(
    project_root: Path,
    *,
    start_month: str = "2026-01",
    as_of: datetime | None = None,
) -> dict[str, object]:
    as_of = as_of or datetime.now(UTC)
    result = _latest_ohlcv_snapshot(
        project_root.resolve(), start_month=start_month, as_of=as_of or datetime.now(UTC)
    )
    from factor_lab.datasets import load_dataset_config

    current = load_dataset_config(project_root.resolve())
    research = {
        "dataset_id": current.research.dataset_id,
        "ohlcv": str(current.research.ohlcv.relative_to(project_root.resolve())),
        "manifest": str(current.research.manifest.relative_to(project_root.resolve())),
        "funding": str(current.research.funding.relative_to(project_root.resolve())),
    }
    _write_dataset_config(
        project_root.resolve(),
        research=research,
        observation={key: result[key] for key in ("dataset_id", "ohlcv", "manifest", "funding")},
    )
    result["path"] = result["ohlcv"]
    return result
