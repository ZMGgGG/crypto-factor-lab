"""Unified evaluation of registered factors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factor_lab.data import validate_ohlcv
from factor_lab.evaluation import FactorReport, evaluate_factor
from factor_lab.execution import CostModel, backtest_positions
from factor_lab.labels import forward_return
from factor_lab.registry import FactorContext, default_registry
from factor_lab.validation import WalkForwardReport, walk_forward_factor_evaluation


@dataclass(frozen=True)
class FactorSuiteExperiment:
    data: pd.DataFrame
    target: pd.Series
    factors: dict[str, pd.Series]
    reports: dict[str, FactorReport]
    walk_forward: dict[str, WalkForwardReport]
    summary: pd.DataFrame
    correlations: pd.DataFrame
    combination_summary: pd.DataFrame
    incremental_contribution: pd.DataFrame


DEFAULT_SIGNAL_SEMANTICS = {
    "price_momentum": "direct",
    "price_reversal": "direct",
    "realized_volatility": "neutral",
    "relative_volume": "neutral",
    "trend_efficiency": "neutral",
    "funding_rate_zscore": "neutral",
}


def run_btc_factor_suite(
    ohlcv: pd.DataFrame,
    momentum_window: int,
    forward_horizon: int,
    train_bars: int,
    test_bars: int,
    funding_rate: pd.Series | None = None,
    expected_frequency: str | None = None,
    cost_model: CostModel | None = None,
    periods_per_year: int = 24 * 365,
    additional_factors: dict[str, pd.Series] | None = None,
    additional_factor_roles: dict[str, str] | None = None,
    additional_signal_semantics: dict[str, str] | None = None,
) -> FactorSuiteExperiment:
    """Evaluate every registered BTC factor with one shared target and split protocol."""
    data = validate_ohlcv(ohlcv, expected_frequency=expected_frequency)
    target = forward_return(data["close"], forward_horizon)
    registry = default_registry(momentum_window, include_funding=funding_rate is not None)
    factors = registry.compute_all(FactorContext(data["close"], data["volume"], funding_rate))
    factor_roles = {definition.factor_id: definition.factor_type for definition in registry.definitions()}
    signal_semantics = dict(DEFAULT_SIGNAL_SEMANTICS)
    for name, factor in (additional_factors or {}).items():
        if name in factors:
            raise ValueError(f"additional factor conflicts with registered factor: {name}")
        if not factor.index.equals(data.index):
            raise ValueError(f"additional factor must match data index: {name}")
        factors[name] = factor.rename(name)
    factor_roles.update(additional_factor_roles or {})
    signal_semantics.update(additional_signal_semantics or {})
    reports: dict[str, FactorReport] = {}
    walk_forward: dict[str, WalkForwardReport] = {}
    rows = []
    diagnostic_positions: dict[str, pd.Series] = {}
    cost_model = cost_model or CostModel()

    for name, factor in factors.items():
        report = evaluate_factor(factor, target)
        oos = walk_forward_factor_evaluation(
            factor,
            target,
            train_bars=train_bars,
            test_bars=test_bars,
            purge_bars=forward_horizon,
        )
        semantics = signal_semantics.get(name, "neutral")
        strategy = None
        if semantics in {"direct", "inverse"}:
            desired_position = factor.gt(0).astype(float).sub(factor.lt(0).astype(float))
            desired_position = desired_position.where(factor.notna(), 0.0)
            if semantics == "inverse":
                desired_position = desired_position.mul(-1)
            diagnostic_positions[name] = desired_position
            strategy = backtest_positions(
                data["close"], desired_position, cost_model, periods_per_year, funding_rate
            )
        reports[name] = report
        walk_forward[name] = oos
        rows.append(
            {
                "factor": name,
                "factor_role": factor_roles.get(name, "unknown"),
                "signal_semantics": semantics,
                "observations": report.observations,
                "full_linear_ic": report.linear_ic,
                "full_rank_ic": report.rank_ic,
                "oos_folds": len(oos.folds),
                "mean_test_ic": oos.mean_test_ic,
                "mean_test_rank_ic": oos.mean_test_rank_ic,
                "positive_test_rank_ic_ratio": oos.positive_test_rank_ic_ratio,
                "gross_return": strategy.gross_return if strategy else float("nan"),
                "net_return": strategy.net_return if strategy else float("nan"),
                "annualized_sharpe": strategy.annualized_sharpe if strategy else float("nan"),
                "max_drawdown": strategy.max_drawdown if strategy else float("nan"),
                "total_turnover": strategy.total_turnover if strategy else 0.0,
            }
        )

    correlation_data = pd.DataFrame(factors).corr(method="spearman")
    position_frame = pd.DataFrame(diagnostic_positions, index=data.index)

    def combination_row(label: str, names: list[str]) -> dict[str, object]:
        positions = position_frame[names].mean(axis=1) if names else pd.Series(0.0, index=data.index)
        strategy = backtest_positions(
            data["close"], positions, cost_model, periods_per_year, funding_rate
        )
        return {
            "combination": label,
            "factor_count": len(names),
            "gross_return": strategy.gross_return,
            "net_return": strategy.net_return,
            "annualized_sharpe": strategy.annualized_sharpe,
            "max_drawdown": strategy.max_drawdown,
            "total_turnover": strategy.total_turnover,
        }

    all_names = list(diagnostic_positions)
    combination_rows = [combination_row("equal_weight_all", all_names)]
    all_strategy = combination_rows[0]
    contribution_rows = []
    for name in all_names:
        remaining = [candidate for candidate in all_names if candidate != name]
        without = combination_row(f"without_{name}", remaining)
        combination_rows.append(without)
        contribution_rows.append(
            {
                "factor": name,
                "all_net_return": all_strategy["net_return"],
                "without_net_return": without["net_return"],
                "incremental_net_return": all_strategy["net_return"] - without["net_return"],
                "all_annualized_sharpe": all_strategy["annualized_sharpe"],
                "without_annualized_sharpe": without["annualized_sharpe"],
                "incremental_annualized_sharpe": all_strategy["annualized_sharpe"] - without["annualized_sharpe"],
            }
        )

    return FactorSuiteExperiment(
        data=data,
        target=target,
        factors=factors,
        reports=reports,
        walk_forward=walk_forward,
        summary=pd.DataFrame(rows).set_index("factor").sort_values("mean_test_rank_ic", ascending=False),
        correlations=correlation_data,
        combination_summary=pd.DataFrame(combination_rows).set_index("combination"),
        incremental_contribution=pd.DataFrame(contribution_rows).set_index("factor"),
    )
