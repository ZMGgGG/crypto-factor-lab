"""Deterministic null benchmarks for factor research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RandomIcBenchmark:
    trials: int
    mean_rank_ic: float
    std_rank_ic: float
    absolute_rank_ic_95: float


def random_rank_ic_benchmark(
    target: pd.Series,
    *,
    trials: int = 100,
    seed: int = 0,
) -> RandomIcBenchmark:
    """Estimate the null Rank-IC range from deterministic random permutations."""
    if trials < 10:
        raise ValueError("trials must be at least ten")
    complete = target.dropna()
    if len(complete) < 20 or complete.nunique() < 2:
        raise ValueError("target needs at least twenty varying observations")
    target_rank = complete.rank().to_numpy(dtype=float)
    target_centered = target_rank - target_rank.mean()
    target_scale = np.sqrt(np.sum(target_centered * target_centered))
    base_rank = np.arange(1, len(complete) + 1, dtype=float)
    base_centered = base_rank - base_rank.mean()
    base_scale = np.sqrt(np.sum(base_centered * base_centered))
    rng = np.random.default_rng(seed)
    values = np.empty(trials, dtype=float)
    for index in range(trials):
        random_rank = rng.permutation(base_centered)
        values[index] = np.sum(random_rank * target_centered) / (base_scale * target_scale)
    return RandomIcBenchmark(
        trials=trials,
        mean_rank_ic=float(values.mean()),
        std_rank_ic=float(values.std(ddof=1)),
        absolute_rank_ic_95=float(np.quantile(np.abs(values), 0.95)),
    )
