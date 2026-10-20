"""Run a deterministic multi-asset cross-sectional factor experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_lab.pipeline import run_cross_sectional_momentum_experiment


def synthetic_crypto_panel(periods: int = 720, symbols: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed=19)
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    rows = []
    for number in range(symbols):
        name = f"ASSET{number:02d}-USDT"
        close = (20 + number * 5) * np.exp(np.cumsum(rng.normal(0.0001, 0.015, periods)))
        open_ = np.concatenate(([close[0]], close[:-1]))
        spread = rng.uniform(0.0002, 0.003, periods)
        rows.extend(
            {
                "timestamp": timestamp,
                "symbol": name,
                "open": opening,
                "high": max(opening, price) * (1 + width),
                "low": min(opening, price) * (1 - width),
                "close": price,
                "volume": volume,
            }
            for timestamp, opening, price, width, volume in zip(
                timestamps,
                open_,
                close,
                spread,
                rng.lognormal(mean=9, sigma=0.5, size=periods),
                strict=True,
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiment = run_cross_sectional_momentum_experiment(
        synthetic_crypto_panel(),
        momentum_window=24,
        forward_horizon=4,
        liquidity_lookback=24,
        minimum_dollar_volume=100_000,
        quantiles=5,
        expected_frequency="1h",
    )
    report = experiment.report
    print(f"eligible observations: {report.observations}")
    print(f"evaluated timestamps: {report.periods}")
    print(f"mean IC: {report.mean_ic:.4f}")
    print(f"mean rank IC: {report.mean_rank_ic:.4f}")
    print(f"rank ICIR: {report.rank_ic_ir:.4f}")
    print("mean forward return by factor quantile:")
    print(report.quantile_returns.to_string())
