"""Run the auditable BTCUSDT factor, cost, and walk-forward baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

from factor_lab.evaluation import evaluate_factor
from factor_lab.execution import CostModel
from factor_lab.factors import price_momentum
from factor_lab.labels import forward_return
from factor_lab.pipeline import run_costed_momentum_backtest
from factor_lab.sources.binance import load_funding_rates, load_normalized_ohlcv
from factor_lab.validation import walk_forward_factor_evaluation


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--momentum-window", type=int, default=24)
    parser.add_argument("--forward-horizon", type=int, default=4)
    parser.add_argument("--train-bars", type=int, default=24 * 365)
    parser.add_argument("--test-bars", type=int, default=24 * 90)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--frequency", default="1h")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--funding", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    arguments = parser.parse_args()

    manifest_path = arguments.manifest or arguments.dataset.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"data manifest not found: {manifest_path}")

    data = load_normalized_ohlcv(arguments.dataset, expected_frequency=arguments.frequency)
    funding_rate = load_funding_rates(arguments.funding) if arguments.funding else None
    factor = price_momentum(data["close"], arguments.momentum_window)
    target = forward_return(data["close"], arguments.forward_horizon)
    factor_report = evaluate_factor(factor, target)
    walk_forward = walk_forward_factor_evaluation(
        factor,
        target,
        train_bars=arguments.train_bars,
        test_bars=arguments.test_bars,
        purge_bars=arguments.forward_horizon,
    )
    cost_model = CostModel(
        fee_bps=arguments.fee_bps,
        half_spread_bps=arguments.half_spread_bps,
        slippage_bps=arguments.slippage_bps,
    )
    strategy = run_costed_momentum_backtest(
        data,
        momentum_window=arguments.momentum_window,
        cost_model=cost_model,
        expected_frequency=arguments.frequency,
        funding_rate=funding_rate,
    ).backtest

    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"btcusdt_{arguments.frequency}_momentum_{arguments.momentum_window}"
    folds_path = arguments.report_dir / f"{stem}_walk_forward.csv"
    summary_path = arguments.report_dir / f"{stem}_summary.json"
    walk_forward.folds.to_csv(folds_path, index=True)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(arguments.dataset),
        "dataset_sha256": file_sha256(arguments.dataset),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "first_timestamp": data.index[0].isoformat(),
        "last_timestamp": data.index[-1].isoformat(),
        "bars": len(data),
        "factor": f"price_momentum_{arguments.momentum_window}",
        "label": f"forward_return_{arguments.forward_horizon}",
        "cost_model": asdict(cost_model),
        "factor_evaluation": {
            "observations": factor_report.observations,
            "linear_ic": factor_report.linear_ic,
            "rank_ic": factor_report.rank_ic,
        },
        "walk_forward": {
            "train_bars": arguments.train_bars,
            "test_bars": arguments.test_bars,
            "purge_bars": arguments.forward_horizon,
            "folds": len(walk_forward.folds),
            "mean_test_ic": walk_forward.mean_test_ic,
            "mean_test_rank_ic": walk_forward.mean_test_rank_ic,
            "positive_test_rank_ic_ratio": walk_forward.positive_test_rank_ic_ratio,
        },
        "strategy": {
            "gross_return": strategy.gross_return,
            "net_return": strategy.net_return,
            "annualized_sharpe": strategy.annualized_sharpe,
            "max_drawdown": strategy.max_drawdown,
            "total_turnover": strategy.total_turnover,
            "rebalance_events": strategy.rebalance_events,
            "funding_included": arguments.funding is not None,
            "funding_dataset": str(arguments.funding) if arguments.funding else None,
            "funding_dataset_sha256": file_sha256(arguments.funding) if arguments.funding else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"bars: {len(data)}")
    print(f"rank IC: {factor_report.rank_ic:.4f}")
    print(f"walk-forward folds: {len(walk_forward.folds)}")
    print(f"mean test rank IC: {walk_forward.mean_test_rank_ic:.4f}")
    print(f"positive test rank IC ratio: {walk_forward.positive_test_rank_ic_ratio:.2%}")
    print(f"gross strategy return: {strategy.gross_return:.2%}")
    print(f"net strategy return: {strategy.net_return:.2%}")
    print(f"annualized Sharpe: {strategy.annualized_sharpe:.3f}")
    print(f"max drawdown: {strategy.max_drawdown:.2%}")
    print(f"summary: {summary_path}")
    print(f"folds: {folds_path}")
