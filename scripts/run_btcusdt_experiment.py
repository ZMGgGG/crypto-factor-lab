"""Evaluate BTCUSDT momentum from a normalized historical CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from factor_lab.pipeline import run_momentum_experiment
from factor_lab.sources.binance import load_normalized_ohlcv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--momentum-window", type=int, default=24)
    parser.add_argument("--forward-horizon", type=int, default=4)
    parser.add_argument("--frequency", default="1h")
    arguments = parser.parse_args()

    data = load_normalized_ohlcv(arguments.dataset, expected_frequency=arguments.frequency)
    experiment = run_momentum_experiment(
        data,
        momentum_window=arguments.momentum_window,
        forward_horizon=arguments.forward_horizon,
        expected_frequency=arguments.frequency,
    )
    report = experiment.report
    print(f"observations: {report.observations}")
    print(f"linear IC: {report.linear_ic:.4f}")
    print(f"rank IC: {report.rank_ic:.4f}")
    print("quantile mean forward returns:")
    print(report.quantile_returns.to_string())
