"""Generate, evaluate, deduplicate, and catalog BTC factor candidates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from factor_lab.catalog import CandidateCatalog
from factor_lab.execution import CostModel
from factor_lab.expression import Expression
from factor_lab.generator import GeneratorConfig, generate_expressions
from factor_lab.labels import forward_return
from factor_lab.mining import SelectionPolicy, TimeSplit, mine_candidates, screen_candidates
from factor_lab.sources.binance import load_funding_rates, load_normalized_ohlcv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine bounded factor expressions on one time-series dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--momentum-horizon", "--forward-horizon", dest="forward_horizon", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--purge-bars", type=int, default=None)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--funding", type=Path)
    parser.add_argument("--stability-windows", type=int, default=8)
    parser.add_argument("--hac-lags", type=int, default=1)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--entry-quantile", type=float, default=0.8)
    parser.add_argument("--screen-min-validation-abs-rank-ic", "--screen-min-validation-rank-ic", dest="screen_min_validation_abs_rank_ic", type=float)
    parser.add_argument("--screen-min-stability-directional-ratio", "--screen-min-stability-positive-ratio", dest="screen_min_stability_directional_ratio", type=float)
    parser.add_argument("--screen-min-validation-sharpe", type=float)
    parser.add_argument("--screen-min-validation-net-return", type=float)
    parser.add_argument("--screen-min-validation-drawdown", type=float)
    parser.add_argument("--screen-max-validation-turnover", type=float)
    parser.add_argument("--screen-require-fdr-pass", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--catalog", type=Path, default=Path("data/state/factor_catalog.sqlite3"))
    parser.add_argument("--frequency", default="1h")
    arguments = parser.parse_args()

    data = load_normalized_ohlcv(arguments.dataset, expected_frequency=arguments.frequency)
    fields = tuple(field for field in ("close", "open", "high", "low", "volume", "funding_rate") if field in data)
    config = GeneratorConfig(fields=fields, max_candidates=arguments.max_candidates)
    expressions: tuple[Expression, ...] = generate_expressions(config)
    horizon = arguments.forward_horizon
    target = forward_return(data["close"], horizon)
    funding = load_funding_rates(arguments.funding) if arguments.funding else None
    cost_model = CostModel(
        fee_bps=arguments.fee_bps,
        half_spread_bps=arguments.half_spread_bps,
        slippage_bps=arguments.slippage_bps,
    )
    split = TimeSplit(arguments.train_ratio, arguments.validation_ratio, arguments.holdout_ratio)
    result = mine_candidates(
        expressions,
        data,
        target,
        split=split,
        purge_bars=horizon if arguments.purge_bars is None else arguments.purge_bars,
        min_observations=arguments.min_observations,
        stability_windows=arguments.stability_windows,
        hac_lags=arguments.hac_lags,
        fdr_q=arguments.fdr_q,
    )
    policy = SelectionPolicy(
        min_validation_abs_rank_ic=arguments.screen_min_validation_abs_rank_ic,
        min_stability_directional_ratio=arguments.screen_min_stability_directional_ratio,
        min_validation_sharpe=arguments.screen_min_validation_sharpe,
        min_validation_net_return=arguments.screen_min_validation_net_return,
        min_validation_drawdown=arguments.screen_min_validation_drawdown,
        max_validation_turnover=arguments.screen_max_validation_turnover,
        require_fdr_pass=True,
    )
    result = result.__class__(
        table=screen_candidates(result.table, policy),
        train_end=result.train_end,
        validation_end=result.validation_end,
        purge_bars=result.purge_bars,
    )
    run_id = CandidateCatalog(arguments.catalog).save_result(
        result,
        dataset_id=str(arguments.dataset),
        config={
            "protocol_version": "2.3",
            "fields": fields,
            "forward_horizon": horizon,
            "generator": config.__dict__,
            "split": split.__dict__,
            "min_observations": arguments.min_observations,
            "stability_windows": arguments.stability_windows,
            "hac_lags": arguments.hac_lags,
            "fdr_q": arguments.fdr_q,
            "entry_quantile": arguments.entry_quantile,
            "selection_policy": policy.__dict__,
            "cost_model": {
                "fee_bps": cost_model.fee_bps,
                "half_spread_bps": cost_model.half_spread_bps,
                "slippage_bps": cost_model.slippage_bps,
                "funding_included": funding is not None,
            },
        },
    )

    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"btc_factor_mining_{run_id}"
    table_path = arguments.report_dir / f"{stem}.csv"
    metadata_path = arguments.report_dir / f"{stem}.json"
    result.table.to_csv(table_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "dataset": str(arguments.dataset),
                "forward_horizon": horizon,
                "candidate_count": len(result.table),
                "valid_count": int(result.table["status"].eq("valid").sum()),
                "duplicate_count": int(result.table["status"].eq("duplicate").sum()),
                "rejected_count": int(result.table["status"].eq("rejected").sum()),
                "error_count": int(result.table["status"].eq("error").sum()),
                "fdr_rejected_count": int(result.table["status"].eq("fdr_rejected").sum()),
                "selected_count": int(result.table["screen_status"].eq("selected").sum()),
                "cost_model": {
                    "fee_bps": cost_model.fee_bps,
                    "half_spread_bps": cost_model.half_spread_bps,
                    "slippage_bps": cost_model.slippage_bps,
                    "funding_included": funding is not None,
                },
                "table": str(table_path),
                "catalog": str(arguments.catalog),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result.table.loc[result.table["screen_status"].eq("selected"), ["formula_zh", "formula", "status", "screen_status", "selection_score", "validation_rank_ic", "validation_stability_directional_ratio"]].head(20).to_string(index=False))
    print(f"run: {run_id}")
    print(f"table: {table_path}")
    print(f"metadata: {metadata_path}")
    print(f"catalog: {arguments.catalog}")
