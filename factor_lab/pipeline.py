"""Composable stage-one experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factor_lab.data import validate_ohlcv, validate_ohlcv_panel
from factor_lab.evaluation import (
    CrossSectionalFactorReport,
    FactorReport,
    evaluate_cross_sectional_factor,
    evaluate_factor,
)
from factor_lab.factors import cross_sectional_rank, panel_price_momentum, price_momentum
from factor_lab.labels import forward_return, panel_forward_return
from factor_lab.execution import CostModel, StrategyBacktest, backtest_positions
from factor_lab.universe import liquidity_universe


@dataclass(frozen=True)
class SingleAssetExperiment:
    data: pd.DataFrame
    factor: pd.Series
    target: pd.Series
    report: FactorReport


@dataclass(frozen=True)
class CrossSectionalExperiment:
    data: pd.DataFrame
    universe: pd.Series
    factor: pd.Series
    target: pd.Series
    report: CrossSectionalFactorReport


@dataclass(frozen=True)
class CostedMomentumExperiment:
    data: pd.DataFrame
    factor: pd.Series
    desired_position: pd.Series
    backtest: StrategyBacktest


def run_momentum_experiment(
    ohlcv: pd.DataFrame,
    momentum_window: int,
    forward_horizon: int,
    expected_frequency: str | None = None,
) -> SingleAssetExperiment:
    """Run the first end-to-end research check without any execution assumptions."""
    data = validate_ohlcv(ohlcv, expected_frequency=expected_frequency)
    factor = price_momentum(data["close"], momentum_window)
    target = forward_return(data["close"], forward_horizon)
    return SingleAssetExperiment(
        data=data,
        factor=factor,
        target=target,
        report=evaluate_factor(factor, target),
    )


def run_cross_sectional_momentum_experiment(
    ohlcv_panel: pd.DataFrame,
    momentum_window: int,
    forward_horizon: int,
    liquidity_lookback: int,
    minimum_dollar_volume: float,
    quantiles: int = 5,
    expected_frequency: str | None = None,
) -> CrossSectionalExperiment:
    """Evaluate ranked momentum in a point-in-time liquidity universe."""
    data = validate_ohlcv_panel(ohlcv_panel, expected_frequency=expected_frequency)
    universe = liquidity_universe(data, liquidity_lookback, minimum_dollar_volume)
    raw_factor = panel_price_momentum(data, momentum_window)
    factor = cross_sectional_rank(raw_factor).where(universe)
    target = panel_forward_return(data, forward_horizon).where(universe)
    return CrossSectionalExperiment(
        data=data,
        universe=universe,
        factor=factor,
        target=target,
        report=evaluate_cross_sectional_factor(factor, target, quantiles=quantiles),
    )


def run_costed_momentum_backtest(
    ohlcv: pd.DataFrame,
    momentum_window: int,
    cost_model: CostModel,
    expected_frequency: str | None = None,
    periods_per_year: int = 24 * 365,
    funding_rate: pd.Series | None = None,
) -> CostedMomentumExperiment:
    """Backtest long/short momentum with a one-bar execution delay."""
    data = validate_ohlcv(ohlcv, expected_frequency=expected_frequency)
    factor = price_momentum(data["close"], momentum_window)
    desired_position = factor.gt(0).astype(float).sub(factor.lt(0).astype(float))
    desired_position = desired_position.where(factor.notna(), 0.0).rename("desired_position")
    return CostedMomentumExperiment(
        data=data,
        factor=factor,
        desired_position=desired_position,
        backtest=backtest_positions(
            data["close"],
            desired_position,
            cost_model,
            periods_per_year,
            funding_rate=funding_rate,
        ),
    )
