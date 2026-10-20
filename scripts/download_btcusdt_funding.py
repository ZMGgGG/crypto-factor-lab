"""Download reproducible Binance USD-M BTCUSDT funding rates."""

from __future__ import annotations

import argparse
from pathlib import Path

from factor_lab.sources.binance import download_btcusdt_funding_rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="First UTC date, for example 2021-01-01")
    parser.add_argument("--end", required=True, help="Last UTC date, for example 2025-12-31")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()

    data, manifest_path = download_btcusdt_funding_rates(
        arguments.start,
        arguments.end,
        arguments.output_root,
    )
    print(f"downloaded funding records: {len(data)}")
    print(f"first timestamp: {data['timestamp'].iloc[0].isoformat()}")
    print(f"last timestamp: {data['timestamp'].iloc[-1].isoformat()}")
    print(f"manifest: {manifest_path}")
