"""Run the stage-one pipeline with deterministic synthetic BTC-like OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_lab.pipeline import run_momentum_experiment


def synthetic_btc_ohlcv(periods: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    log_returns = rng.normal(loc=0.0001, scale=0.01, size=periods)
    close = 60_000 * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = rng.uniform(0.0002, 0.003, size=periods)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.lognormal(mean=10, sigma=0.4, size=periods),
        }
    )


if __name__ == "__main__":
    experiment = run_momentum_experiment(
        synthetic_btc_ohlcv(),
        momentum_window=24,
        forward_horizon=4,
        expected_frequency="1h",
    )
    report = experiment.report
    print(f"observations: {report.observations}")
    print(f"linear IC: {report.linear_ic:.4f}")
    print(f"rank IC: {report.rank_ic:.4f}")
    print("quantile mean forward returns:")
    print(report.quantile_returns.to_string())
