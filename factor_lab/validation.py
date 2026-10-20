"""Time-ordered validation utilities for factor research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardReport:
    folds: pd.DataFrame
    mean_test_ic: float
    mean_test_rank_ic: float
    positive_test_rank_ic_ratio: float


def walk_forward_factor_evaluation(
    factor: pd.Series,
    target: pd.Series,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    purge_bars: int = 0,
) -> WalkForwardReport:
    """Measure factor stability across rolling, strictly ordered test windows."""
    if min(train_bars, test_bars) < 5:
        raise ValueError("train_bars and test_bars must each be at least five")
    step_bars = step_bars or test_bars
    if step_bars < 1:
        raise ValueError("step_bars must be positive")
    if purge_bars < 0:
        raise ValueError("purge_bars cannot be negative")

    data = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna().sort_index()
    if data.index.has_duplicates:
        raise ValueError("factor observations must have a unique time index")
    if len(data) < train_bars + purge_bars + test_bars:
        raise ValueError("not enough observations for one walk-forward fold")

    rows = []
    test_start = train_bars + purge_bars
    fold = 1
    while test_start + test_bars <= len(data):
        train_end = test_start - purge_bars
        train = data.iloc[train_end - train_bars:train_end]
        test = data.iloc[test_start:test_start + test_bars]
        rows.append(
            {
                "fold": fold,
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "purge_bars": purge_bars,
                "train_ic": train["factor"].corr(train["target"]),
                "train_rank_ic": train["factor"].rank().corr(train["target"].rank()),
                "test_ic": test["factor"].corr(test["target"]),
                "test_rank_ic": test["factor"].rank().corr(test["target"].rank()),
                "test_observations": len(test),
            }
        )
        fold += 1
        test_start += step_bars

    folds = pd.DataFrame(rows).set_index("fold")
    return WalkForwardReport(
        folds=folds,
        mean_test_ic=folds["test_ic"].mean(),
        mean_test_rank_ic=folds["test_rank_ic"].mean(),
        positive_test_rank_ic_ratio=folds["test_rank_ic"].gt(0).mean(),
    )
