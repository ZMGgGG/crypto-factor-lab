"""Run and persist the registered BTC factor suite."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

from factor_lab.sources.binance import load_funding_rates, load_normalized_ohlcv
from factor_lab.execution import CostModel
from factor_lab.suite import run_btc_factor_suite


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--funding", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--momentum-window", type=int, default=24)
    parser.add_argument("--forward-horizon", type=int, default=4)
    parser.add_argument("--train-bars", type=int, default=24 * 365)
    parser.add_argument("--test-bars", type=int, default=24 * 90)
    parser.add_argument("--frequency", default="1h")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    arguments = parser.parse_args()

    manifest_path = arguments.manifest or arguments.dataset.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"data manifest not found: {manifest_path}")
    data = load_normalized_ohlcv(arguments.dataset, expected_frequency=arguments.frequency)
    funding = load_funding_rates(arguments.funding) if arguments.funding else None
    experiment = run_btc_factor_suite(
        data,
        momentum_window=arguments.momentum_window,
        forward_horizon=arguments.forward_horizon,
        train_bars=arguments.train_bars,
        test_bars=arguments.test_bars,
        funding_rate=funding,
        expected_frequency=arguments.frequency,
    )

    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"btcusdt_{arguments.frequency}_factor_suite_{arguments.momentum_window}"
    summary_csv = arguments.report_dir / f"{stem}.csv"
    correlations_csv = arguments.report_dir / f"{stem}_correlations.csv"
    combination_csv = arguments.report_dir / f"{stem}_combination.csv"
    incremental_csv = arguments.report_dir / f"{stem}_incremental.csv"
    summary_json = arguments.report_dir / f"{stem}.json"
    experiment.summary.to_csv(summary_csv, index=True)
    experiment.correlations.to_csv(correlations_csv, index=True)
    experiment.combination_summary.to_csv(combination_csv, index=True)
    experiment.incremental_contribution.to_csv(incremental_csv, index=True)
    summary_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset": str(arguments.dataset),
                "dataset_sha256": file_sha256(arguments.dataset),
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "funding_dataset": str(arguments.funding) if arguments.funding else None,
                "funding_dataset_sha256": file_sha256(arguments.funding) if arguments.funding else None,
                "momentum_window": arguments.momentum_window,
                "forward_horizon": arguments.forward_horizon,
                "train_bars": arguments.train_bars,
                "test_bars": arguments.test_bars,
                "purge_bars": arguments.forward_horizon,
                "factors": list(experiment.factors),
                "cost_model": {
                    "fee_bps": 4.0,
                    "half_spread_bps": 1.0,
                    "slippage_bps": 1.0,
                    "funding_included": funding is not None,
                },
                "correlations": experiment.correlations.to_dict(),
                "correlations_table": str(correlations_csv.relative_to(arguments.report_dir.parent.parent)),
                "combination": experiment.combination_summary.reset_index().to_dict(orient="records"),
                "incremental_contribution": experiment.incremental_contribution.reset_index().to_dict(orient="records"),
                "combination_table": str(combination_csv.relative_to(arguments.report_dir.parent.parent)),
                "incremental_table": str(incremental_csv.relative_to(arguments.report_dir.parent.parent)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(experiment.summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"summary: {summary_json}")
    print(f"table: {summary_csv}")
    print(f"correlations: {correlations_csv}")
    print(f"combination: {combination_csv}")
    print(f"incremental contribution: {incremental_csv}")
