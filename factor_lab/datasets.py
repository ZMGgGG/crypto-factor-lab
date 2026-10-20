"""Small, explicit dataset configuration for research and observation snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    ohlcv: Path
    manifest: Path
    funding: Path


@dataclass(frozen=True)
class DatasetConfig:
    research: DatasetSnapshot
    observation: DatasetSnapshot | None = None


def load_dataset_config(project_root: Path, path: Path | None = None) -> DatasetConfig:
    config_path = path or project_root / "data/state/datasets.json"
    if not config_path.exists():
        research = Path("data/raw/binance_um/BTCUSDT/1h/BTCUSDT-1h-2021-01_to_2025-12.csv")
        funding = Path("data/raw/binance_um/BTCUSDT/funding/BTCUSDT-funding-2021-01-01_to_2025-12-31.csv")
        return DatasetConfig(
            research=DatasetSnapshot(
                "btcusdt_1h_2021_2025",
                project_root / research,
                project_root / research.with_suffix(".manifest.json"),
                project_root / funding,
            )
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    def snapshot(name: str) -> DatasetSnapshot:
        item = payload[name]
        return DatasetSnapshot(
            str(item["dataset_id"]),
            project_root / str(item["ohlcv"]),
            project_root / str(item["manifest"]),
            project_root / str(item["funding"]),
        )

    return DatasetConfig(
        research=snapshot("research"),
        observation=snapshot("observation") if payload.get("observation") else None,
    )
