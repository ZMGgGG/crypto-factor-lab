"""Download a complete-month BTC observation snapshot and update datasets.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from factor_lab.observation_data import update_observation_snapshot, update_research_snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01", help="First observation UTC month")
    parser.add_argument("--research-start", default="2021-01", help="First research UTC month")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()

    research = update_research_snapshot(
        arguments.project_root.resolve(), start_month=arguments.research_start
    )
    result = update_observation_snapshot(
        arguments.project_root.resolve(), start_month=arguments.start
    )
    print(f"research dataset: {research['dataset_id']}")
    print(f"research bars: {research['bars']}")
    print(f"research last timestamp: {research['last_timestamp']}")
    print(f"observation bars: {result['bars']}")
    print(f"funding records: {result['funding_records']}")
    print(f"last timestamp: {result['last_timestamp']}")
