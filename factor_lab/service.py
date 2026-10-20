"""Application services shared by the HTTP API and future workers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from threading import Lock
from uuid import uuid4

import pandas as pd

from factor_lab.ai.client import AIClientError, OpenAIProposalClient, openai_availability, openai_status, resolve_ai_config
from factor_lab.ai.proposals import merge_candidate_sources, validate_proposal_batch
from factor_lab.benchmarks import random_rank_ic_benchmark
from factor_lab.datasets import load_dataset_config
from factor_lab.registry import default_registry
from factor_lab.sources.binance import load_funding_rates, load_normalized_ohlcv
from factor_lab.suite import run_btc_factor_suite
from factor_lab.execution import CostModel, quantile_positions, rebalance_every
from factor_lab.catalog import CandidateCatalog
from factor_lab.expression import Expression, evaluate_expression
from factor_lab.generator import GeneratorConfig, generate_expressions
from factor_lab.labels import forward_return
from factor_lab.mining import SelectionPolicy, TimeSplit, mine_candidates, screen_candidates
from factor_lab.observation_data import update_observation_snapshot, update_research_snapshot
from factor_lab.task_store import TaskStore
from factor_lab.evaluation import evaluate_factor
from factor_lab.validation import walk_forward_factor_evaluation
from factor_lab.execution import backtest_positions
from factor_lab.tracking import evaluate_tracking


@dataclass
class TaskRecord:
    task_id: str
    payload: dict[str, object] = field(default_factory=dict)
    status: str = "queued"
    submitted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class ResearchService:
    """Long-lived research service with bounded background execution."""

    def __init__(
        self,
        project_root: Path,
        max_workers: int = 1,
        task_store_path: Path | None = None,
        candidate_catalog_path: Path | None = None,
    ) -> None:
        self.project_root = project_root
        datasets = load_dataset_config(project_root)
        self.dataset_path = datasets.research.ohlcv
        self.manifest_path = datasets.research.manifest
        self.funding_path = datasets.research.funding
        self.research_dataset_id = datasets.research.dataset_id
        self.report_dir = project_root / "reports/generated"
        self.catalog = CandidateCatalog(candidate_catalog_path or project_root / "data/state/factor_catalog.sqlite3")
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="factor-research")
        self.mining_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="factor-mining")
        self.tasks: dict[str, TaskRecord] = {}
        self.lock = Lock()
        self.task_store = TaskStore(task_store_path or project_root / "data/state/research_tasks.sqlite3")
        self.task_store.fail_interrupted(datetime.now(UTC).isoformat())

    @staticmethod
    def _research_window(
        data: pd.DataFrame, config: dict[str, object]
    ) -> pd.DataFrame:
        start = pd.Timestamp(str(config.get("research_start", data.index[0])))
        end = pd.Timestamp(str(config.get("research_end", data.index[-1])))
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        result = data.loc[(data.index >= start) & (data.index <= end)]
        if len(result) < 24 * 30:
            raise ValueError("research interval must contain at least 30 days of hourly data")
        return result

    def factors(self) -> list[dict[str, object]]:
        factors = [
            {
                "factor_id": definition.factor_id,
                "factor_type": definition.factor_type,
                "description": definition.description,
                "required_fields": definition.required_fields,
                "source": "manual",
            }
            for definition in default_registry(24, include_funding=True).definitions()
        ]
        for registered in self.catalog.list_library():
            expression = Expression.from_dict(json.loads(str(registered["canonical"])))
            factors.append(
                {
                    "factor_id": registered["factor_id"],
                    "factor_type": registered["factor_role"],
                    "description": expression.formula_zh(),
                    "formula": registered["formula"],
                    "formula_zh": expression.formula_zh(),
                    "required_fields": sorted(expression.fields),
                    "source": "mined",
                    "signal_semantics": registered["signal_semantics"],
                }
            )
        return factors

    def factor_library(self) -> list[dict[str, object]]:
        factors = []
        for factor in self.catalog.list_library():
            item = dict(factor)
            expression = Expression.from_dict(json.loads(str(item["canonical"])))
            item["formula_zh"] = expression.formula_zh()
            factors.append(item)
        return factors

    def observation_tracks(self) -> list[dict[str, object]]:
        tracks = []
        for track in self.catalog.list_tracks():
            candidate = self.catalog.get_candidate(
                str(track["run_id"]), str(track["expression_id"])
            )
            item = dict(track)
            if candidate:
                item["formula"] = candidate["formula"]
                item["formula_zh"] = Expression.from_dict(
                    json.loads(str(candidate["canonical"]))
                ).formula_zh()
            tracks.append(item)
        return tracks

    def create_observation_track(
        self, run_id: str, expression_id: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        payload = payload or {}
        candidate = self.catalog.get_candidate(run_id, expression_id)
        run = self.catalog.get_run(run_id)
        if not candidate or not run:
            raise KeyError("candidate or mining run not found")
        if candidate.get("status") != "valid" or candidate.get("screen_status") != "selected":
            raise ValueError("only validation-selected candidates can be observed")
        config = dict(run.get("config", {}))
        if config.get("protocol_version") != "2.3":
            raise ValueError("observation tracking requires a protocol 2.3 mining run")
        datasets = load_dataset_config(self.project_root)
        if datasets.observation is None or not datasets.observation.ohlcv.exists():
            raise ValueError("observation dataset is not configured; update monthly data first")
        expression = Expression.from_dict(json.loads(str(candidate["canonical"])))
        all_research = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        research = self._research_window(all_research, config)
        factor = evaluate_expression(expression, research)
        semantics = str(candidate.get("suggested_signal_semantics", ""))
        aligned = factor.mul(-1.0 if semantics == "inverse" else 1.0)
        development_end = pd.Timestamp(str(run["validation_end"]))
        development = aligned.loc[:development_end].dropna()
        _, _, holdout_slice = TimeSplit(
            float(dict(config.get("split", {})).get("train_ratio", 0.6)),
            float(dict(config.get("split", {})).get("validation_ratio", 0.2)),
            float(dict(config.get("split", {})).get("holdout_ratio", 0.2)),
        ).segments(len(research), int(config.get("forward_horizon", 4)))
        automatic_start = research.index[holdout_slice.start]
        requested_start = payload.get("observation_start")
        if requested_start:
            tracking_start = pd.Timestamp(str(requested_start), tz="UTC")
            if tracking_start < automatic_start:
                raise ValueError(
                    f"observation_start cannot be before {automatic_start.isoformat()}"
                )
        else:
            tracking_start = automatic_start
        requested_end = payload.get("observation_end")
        if requested_end:
            tracking_end = pd.Timestamp(str(requested_end), tz="UTC") + pd.Timedelta(hours=23)
            if tracking_end < tracking_start:
                raise ValueError("observation_end must not be before observation_start")
            observation_bars = int(
                (tracking_end - tracking_start) / pd.Timedelta(hours=1)
            ) + 1
        else:
            observation_days = int(payload.get("observation_days", 90))
            if observation_days < 1 or observation_days > 365:
                raise ValueError("observation_days must be between 1 and 365")
            observation_bars = 24 * observation_days
            tracking_end = tracking_start + pd.Timedelta(hours=observation_bars - 1)
        entry_quantile = float(config.get("entry_quantile", 0.8))
        lower = float(development.quantile(1.0 - entry_quantile))
        upper = float(development.quantile(entry_quantile))
        track = self.catalog.create_track(
            run_id=run_id,
            expression_id=expression_id,
            signal_semantics=semantics,
            forward_horizon=int(config.get("forward_horizon", 4)),
            entry_quantile=entry_quantile,
            entry_lower=lower,
            entry_upper=upper,
            cost_model=dict(config.get("cost_model", {})),
            observation_bars=observation_bars,
            tracking_start=tracking_start.isoformat(),
            tracking_end=tracking_end.isoformat(),
        )
        return self.update_observation_track(str(track["track_id"]))

    def update_observation_track(self, track_id: str) -> dict[str, object]:
        track = self.catalog.get_track(track_id)
        if not track:
            raise KeyError("observation track not found")
        candidate = self.catalog.get_candidate(
            str(track["run_id"]), str(track["expression_id"])
        )
        run = self.catalog.get_run(str(track["run_id"]))
        if not candidate or not run:
            raise KeyError("tracked candidate or mining run not found")
        datasets = load_dataset_config(self.project_root)
        if datasets.observation is None:
            raise ValueError("observation dataset is not configured")
        all_research = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        run_config = dict(run.get("config", {}))
        research = self._research_window(all_research, run_config)
        later_data = load_normalized_ohlcv(
            datasets.observation.ohlcv, expected_frequency="1h"
        )
        available_data = pd.concat([all_research, later_data])
        available_data = available_data.loc[
            ~available_data.index.duplicated(keep="last")
        ].sort_index()
        research_funding = load_funding_rates(self.funding_path)
        later_funding = load_funding_rates(datasets.observation.funding)
        funding = pd.concat([research_funding, later_funding]).sort_index()
        funding = funding.loc[~funding.index.duplicated(keep="last")]
        split_config = dict(run_config.get("split", {}))
        _, _, holdout_slice = TimeSplit(
            float(split_config.get("train_ratio", 0.6)),
            float(split_config.get("validation_ratio", 0.2)),
            float(split_config.get("holdout_ratio", 0.2)),
        ).segments(len(research), int(track["forward_horizon"]))
        tracking_start = (
            pd.Timestamp(str(track["tracking_start"]))
            if track.get("tracking_start")
            else research.index[holdout_slice.start]
        )
        tracking_end = (
            pd.Timestamp(str(track["tracking_end"]))
            if track.get("tracking_end")
            else tracking_start + pd.Timedelta(hours=int(track["observation_bars"]) - 1)
        )
        costs = dict(track["cost_model"])
        report = evaluate_tracking(
            Expression.from_dict(json.loads(str(candidate["canonical"]))),
            research,
            available_data.loc[available_data.index > pd.Timestamp(str(run["validation_end"]))],
            funding,
            signal_semantics=str(track["signal_semantics"]),
            forward_horizon=int(track["forward_horizon"]),
            entry_quantile=float(track["entry_quantile"]),
            development_end=pd.Timestamp(str(run["validation_end"])),
            tracking_start=tracking_start,
            tracking_end=tracking_end,
            observation_bars=int(track["observation_bars"]),
            cost_model=CostModel(
                fee_bps=float(costs.get("fee_bps", 4.0)),
                half_spread_bps=float(costs.get("half_spread_bps", 1.0)),
                slippage_bps=float(costs.get("slippage_bps", 1.0)),
            ),
            frozen_thresholds=(float(track["entry_lower"]), float(track["entry_upper"])),
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = self.report_dir / f"observation_{track_id}.csv"
        report.timeline.to_csv(timeline_path)
        result = {
            **report.summary,
            "source_dataset_ids": [
                str(run.get("dataset_id", "")),
                datasets.observation.dataset_id,
            ],
            "timeline": str(timeline_path.relative_to(self.project_root)),
        }
        updated = self.catalog.update_track(track_id, result)
        updated["formula"] = candidate["formula"]
        updated["formula_zh"] = Expression.from_dict(
            json.loads(str(candidate["canonical"]))
        ).formula_zh()
        return updated

    def register_candidate(
        self,
        run_id: str,
        expression_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        factor_id = str(payload.get("factor_id", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", factor_id):
            raise ValueError("factor_id must use 3-64 lowercase letters, digits, or underscores")
        return self.catalog.add_to_library(
            factor_id=factor_id,
            run_id=run_id,
            expression_id=expression_id,
        )

    @staticmethod
    def _validate_factor_semantics(factor_role: str, signal_semantics: str) -> None:
        if factor_role not in {"direction", "risk", "condition"}:
            raise ValueError("factor_role must be direction, risk, or condition")
        if signal_semantics not in {"direct", "inverse", "neutral"}:
            raise ValueError("signal_semantics must be direct, inverse, or neutral")
        if factor_role == "direction" and signal_semantics == "neutral":
            raise ValueError("direction factors must use direct or inverse signal semantics")
        if factor_role != "direction" and signal_semantics != "neutral":
            raise ValueError("risk and condition factors must use neutral signal semantics")

    def status(self) -> dict[str, object]:
        data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        datasets = load_dataset_config(self.project_root)
        observation_status = None
        if datasets.observation and datasets.observation.ohlcv.exists():
            observation = load_normalized_ohlcv(
                datasets.observation.ohlcv, expected_frequency="1h"
            )
            observation_status = {
                "id": datasets.observation.dataset_id,
                "bars": len(observation),
                "first_timestamp": observation.index[0].isoformat(),
                "last_timestamp": observation.index[-1].isoformat(),
                "path": str(datasets.observation.ohlcv.relative_to(self.project_root)),
            }
        latest_report = self._latest_report_rows()
        return {
            "dataset": {
                "id": self.research_dataset_id,
                "symbol": "BTCUSDT",
                "market": "Binance USD-M",
                "frequency": "1h",
                "bars": len(data),
                "first_timestamp": data.index[0].isoformat(),
                "last_timestamp": data.index[-1].isoformat(),
                "path": str(self.dataset_path.relative_to(self.project_root)),
                "manifest": str(self.manifest_path.relative_to(self.project_root)),
                "funding_path": str(self.funding_path.relative_to(self.project_root)),
            },
            "latest_report": latest_report,
            "observation_dataset": observation_status,
            "ai": openai_status(),
            "tasks": self.task_history(),
        }

    def chart(self, window: int = 24, points: int = 180) -> dict[str, object]:
        if window < 2 or points < 10:
            raise ValueError("window must be at least 2 and points at least 10")
        data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        close = data["close"]
        momentum = close.pct_change(window, fill_method=None)
        step = max(1, len(data) // points)
        sampled = pd.DataFrame({"timestamp": data.index, "close": close, "momentum": momentum}).iloc[::step].tail(points)
        return {
            "window": window,
            "points": [
                {"timestamp": row.timestamp.isoformat(), "close": float(row.close), "momentum": None if pd.isna(row.momentum) else float(row.momentum)}
                for row in sampled.itertuples(index=False)
            ],
        }

    def submit_experiment(self, payload: dict[str, object]) -> TaskRecord:
        task = TaskRecord(task_id=uuid4().hex, payload={**payload, "task_type": "experiment"})
        with self.lock:
            self.tasks[task.task_id] = task
        self.task_store.save(self._task_dict(task))
        self.executor.submit(self._run_task, task.task_id, payload)
        return task

    def submit_mining(self, payload: dict[str, object]) -> TaskRecord:
        task = TaskRecord(task_id=uuid4().hex, payload={**payload, "task_type": "mining"})
        with self.lock:
            self.tasks[task.task_id] = task
        self.task_store.save(self._task_dict(task))
        self.mining_executor.submit(self._run_task, task.task_id, task.payload)
        return task

    def submit_data_update(self) -> TaskRecord:
        task = TaskRecord(task_id=uuid4().hex, payload={"task_type": "data_update"})
        with self.lock:
            self.tasks[task.task_id] = task
        self.task_store.save(self._task_dict(task))
        self.executor.submit(self._run_task, task.task_id, task.payload)
        return task

    def submit_promotion(
        self,
        run_id: str,
        expression_id: str,
        payload: dict[str, object],
    ) -> TaskRecord:
        candidate = self.catalog.get_candidate(run_id, expression_id)
        if not candidate:
            raise KeyError("candidate not found")
        mining_run = self.catalog.get_run(run_id)
        if not mining_run:
            raise KeyError("mining run not found")
        confirmation_available, confirmation_reason = self._final_confirmation_availability(
            mining_run
        )
        if not confirmation_available:
            raise ValueError(confirmation_reason)
        if candidate.get("status") != "valid" or candidate.get("screen_status") != "selected":
            raise ValueError("only selected valid candidates can be promoted")
        if candidate.get("promotion_status") == "promoted" and candidate.get("promotion_role"):
            raise ValueError("candidate has already been promoted")
        factor_role = str(payload.get("factor_role", "")).strip()
        signal_semantics = str(payload.get("signal_semantics", "")).strip()
        self._validate_factor_semantics(factor_role, signal_semantics)
        if factor_role != "direction":
            raise ValueError("risk and condition factors require a specialized confirmation protocol")
        task = TaskRecord(
            task_id=uuid4().hex,
            payload={"task_type": "promotion", "run_id": run_id, "expression_id": expression_id, "formula": candidate.get("formula"), "factor_role": factor_role, "signal_semantics": signal_semantics},
        )
        task.payload["task_id"] = task.task_id
        with self.lock:
            self.tasks[task.task_id] = task
        self.task_store.save(self._task_dict(task))
        self.executor.submit(self._run_task, task.task_id, task.payload)
        return task

    def task_history(self) -> list[dict[str, object]]:
        return self.task_store.list()

    def mining_runs(self) -> list[dict[str, object]]:
        return self.catalog.list_runs()

    def mining_run(self, run_id: str) -> dict[str, object]:
        run = self.catalog.get_run(run_id)
        if not run:
            raise KeyError("mining run not found")
        available, reason = self._final_confirmation_availability(run)
        tracks = {
            str(track["expression_id"]): track
            for track in self.catalog.list_tracks()
            if str(track["run_id"]) == run_id
        }
        run["final_confirmation_available"] = available
        run["final_confirmation_reason"] = reason
        config = dict(run.get("config", {}))
        run["mining_mode_requested"] = config.get("mining_mode_requested", "formula")
        run["mining_mode_effective"] = config.get("mining_mode_effective", "formula")
        run["ai_used"] = bool(config.get("ai_used", False))
        run["ai_fallback_reason"] = config.get("ai_fallback_reason")
        run["ai_audit"] = config.get("ai_audit", {})
        run["candidates"] = [
            {
                **self._candidate_for_api(candidate),
                "final_confirmation_available": available,
                "final_confirmation_reason": reason,
                "observation_track": tracks.get(str(candidate["expression_id"])),
            }
            for candidate in run["candidates"]
        ]
        return run

    def mining_candidate(self, run_id: str, expression_id: str) -> dict[str, object]:
        candidate = self.catalog.get_candidate(run_id, expression_id)
        if not candidate:
            raise KeyError("candidate not found")
        run = self.catalog.get_run(run_id)
        available, reason = self._final_confirmation_availability(run or {})
        earliest_observation_start = None
        if run:
            config = dict(run.get("config", {}))
            split_config = dict(config.get("split", {}))
            all_research = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
            research = self._research_window(all_research, config)
            _, _, holdout_slice = TimeSplit(
                float(split_config.get("train_ratio", 0.6)),
                float(split_config.get("validation_ratio", 0.2)),
                float(split_config.get("holdout_ratio", 0.2)),
            ).segments(len(research), int(config.get("forward_horizon", 4)))
            earliest_observation_start = research.index[holdout_slice.start].ceil("D").strftime("%Y-%m-%d")
        return {
            **self._candidate_for_api(candidate),
            "final_confirmation_available": available,
            "final_confirmation_reason": reason,
            "observation_track": self.catalog.get_track_for_candidate(run_id, expression_id),
            "earliest_observation_start": earliest_observation_start,
        }

    def _final_confirmation_availability(
        self, run: dict[str, object]
    ) -> tuple[bool, str]:
        config = dict(run.get("config", {}))
        if config.get("protocol_version") != "2.3":
            return False, "只有协议 2.3 运行可以执行最终确认"
        all_data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        data = self._research_window(all_data, config)
        split_config = dict(config.get("split", {}))
        split = TimeSplit(
            float(split_config.get("train_ratio", 0.6)),
            float(split_config.get("validation_ratio", 0.2)),
            float(split_config.get("holdout_ratio", 0.2)),
        )
        _, _, holdout_slice = split.segments(
            len(data), int(config.get("forward_horizon", 4))
        )
        holdout_start = data.index[holdout_slice.start].isoformat()
        latest_revealed = self.catalog.latest_revealed_holdout_end(
            str(run.get("dataset_id", ""))
        )
        if latest_revealed is not None and latest_revealed >= holdout_start:
            return False, "当前留出区间与已揭示数据重叠，需要新增未见数据"
        return True, "可执行一次最终留出确认"

    def task(self, task_id: str) -> TaskRecord:
        with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id]
        stored = self.task_store.get(task_id)
        if not stored:
            raise KeyError(f"unknown task: {task_id}")
        return TaskRecord(
            task_id=str(stored["task_id"]),
            payload=dict(stored["payload"]),
            status=str(stored["status"]),
            submitted_at=str(stored["submitted_at"]),
            started_at=stored["started_at"],
            finished_at=stored["finished_at"],
            result=stored["result"],
            error=stored["error"],
        )

    def _run_task(self, task_id: str, payload: dict[str, object]) -> None:
        task = self.tasks[task_id]
        task.status = "running"
        task.started_at = datetime.now(UTC).isoformat()
        self.task_store.save(self._task_dict(task))
        try:
            if payload.get("task_type") == "mining":
                task.result = self.run_mining(payload)
            elif payload.get("task_type") == "promotion":
                task.result = self.run_promotion(payload)
            elif payload.get("task_type") == "data_update":
                research = update_research_snapshot(self.project_root)
                dataset = update_observation_snapshot(self.project_root)
                snapshots = load_dataset_config(self.project_root)
                self.dataset_path = snapshots.research.ohlcv
                self.manifest_path = snapshots.research.manifest
                self.funding_path = snapshots.research.funding
                self.research_dataset_id = snapshots.research.dataset_id
                tracks = [
                    self.update_observation_track(str(track["track_id"]))
                    for track in self.catalog.list_tracks()
                ]
                task.result = {
                    "research_dataset": research,
                    "observation_dataset": dataset,
                    "dataset": dataset,
                    "updated_tracks": len(tracks),
                    "restart_required": True,
                }
            else:
                task.result = self.run_experiment(payload)
            task.status = "succeeded"
        except Exception as error:  # task status must expose errors to the client
            task.status = "failed"
            task.error = str(error)
        finally:
            task.finished_at = datetime.now(UTC).isoformat()
            self.task_store.save(self._task_dict(task))

    def run_experiment(self, payload: dict[str, object]) -> dict[str, object]:
        momentum_window = int(payload.get("momentum_window", 24))
        forward_horizon = int(payload.get("forward_horizon", 4))
        train_bars = int(payload.get("train_bars", 24 * 365))
        test_bars = int(payload.get("test_bars", 24 * 90))
        if not 2 <= momentum_window <= 24 * 365:
            raise ValueError("momentum_window must be between 2 and 8760")
        if not 1 <= forward_horizon <= 24 * 30:
            raise ValueError("forward_horizon must be between 1 and 720")
        if train_bars < 5 or test_bars < 5:
            raise ValueError("train_bars and test_bars must be at least 5")
        data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        funding = load_funding_rates(self.funding_path)
        cost_model = CostModel(
            fee_bps=float(payload.get("fee_bps", 4.0)),
            half_spread_bps=float(payload.get("half_spread_bps", 1.0)),
            slippage_bps=float(payload.get("slippage_bps", 1.0)),
        )
        formal_factors: dict[str, pd.Series] = {}
        formal_roles: dict[str, str] = {}
        formal_semantics: dict[str, str] = {}
        for registered in self.catalog.list_library():
            if registered["status"] != "active":
                continue
            expression = Expression.from_dict(json.loads(str(registered["canonical"])))
            formal_factors[str(registered["factor_id"])] = evaluate_expression(expression, data)
            formal_roles[str(registered["factor_id"])] = str(registered["factor_role"])
            formal_semantics[str(registered["factor_id"])] = str(registered["signal_semantics"])
        experiment = run_btc_factor_suite(
            data,
            momentum_window,
            forward_horizon,
            train_bars,
            test_bars,
            funding,
            "1h",
            cost_model,
            additional_factors=formal_factors,
            additional_factor_roles=formal_roles,
            additional_signal_semantics=formal_semantics,
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = f"api_btcusdt_1h_factor_suite_{momentum_window}_{forward_horizon}"
        csv_path = self.report_dir / f"{stem}.csv"
        json_path = self.report_dir / f"{stem}.json"
        experiment.summary.to_csv(csv_path, index=True)
        combination_csv = self.report_dir / f"{stem}_combination.csv"
        incremental_csv = self.report_dir / f"{stem}_incremental.csv"
        experiment.combination_summary.to_csv(combination_csv, index=True)
        experiment.incremental_contribution.to_csv(incremental_csv, index=True)
        result = {
            "dataset": self.research_dataset_id,
            "momentum_window": momentum_window,
            "forward_horizon": forward_horizon,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "purge_bars": forward_horizon,
            "cost_model": {"fee_bps": cost_model.fee_bps, "half_spread_bps": cost_model.half_spread_bps, "slippage_bps": cost_model.slippage_bps, "funding_included": True},
            "factors": experiment.summary.reset_index().to_dict(orient="records"),
            "correlations": experiment.correlations.to_dict(),
            "combination": experiment.combination_summary.reset_index().to_dict(orient="records"),
            "incremental_contribution": experiment.incremental_contribution.reset_index().to_dict(orient="records"),
            "report": str(json_path.relative_to(self.project_root)),
            "table": str(csv_path.relative_to(self.project_root)),
            "combination_table": str(combination_csv.relative_to(self.project_root)),
            "incremental_table": str(incremental_csv.relative_to(self.project_root)),
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def run_mining(self, payload: dict[str, object]) -> dict[str, object]:
        max_candidates = int(payload.get("max_candidates", 200))
        mining_mode = str(payload.get("mining_mode", "formula"))
        if mining_mode not in {"formula", "hybrid"}:
            raise ValueError("mining_mode must be formula or hybrid")
        forward_horizon = int(payload.get("forward_horizon", 4))
        train_ratio = float(payload.get("train_ratio", 0.6))
        validation_ratio = float(payload.get("validation_ratio", 0.2))
        holdout_ratio = float(payload.get("holdout_ratio", 0.2))
        fdr_q = float(payload.get("fdr_q", 0.05))
        entry_quantile = float(payload.get("entry_quantile", 0.8))
        if not 1 <= max_candidates <= 5000:
            raise ValueError("max_candidates must be between 1 and 5000")
        ai_candidate_count = int(payload.get("ai_candidate_count", min(20, max_candidates)))
        if mining_mode == "hybrid" and not 1 <= ai_candidate_count <= min(100, max_candidates):
            raise ValueError("ai_candidate_count must be between 1 and max_candidates, capped at 100")
        if not 1 <= forward_horizon <= 24 * 30:
            raise ValueError("forward_horizon must be between 1 and 720")
        if not 0 < fdr_q <= 1:
            raise ValueError("fdr_q must be in (0, 1]")
        if not 0.5 < entry_quantile < 1:
            raise ValueError("entry_quantile must be between 0.5 and 1")
        all_data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        requested_start = pd.Timestamp(str(payload.get("research_start", all_data.index[0].date())), tz="UTC")
        requested_end = pd.Timestamp(str(payload.get("research_end", all_data.index[-1].date())), tz="UTC") + pd.Timedelta(hours=23)
        if requested_end < requested_start:
            raise ValueError("research_end must not be before research_start")
        data = self._research_window(
            all_data,
            {"research_start": requested_start.isoformat(), "research_end": requested_end.isoformat()},
        )
        funding = load_funding_rates(self.funding_path)
        config = GeneratorConfig(fields=("close", "open", "high", "low", "volume"), max_candidates=max_candidates)
        ai_config = resolve_ai_config(
            str(payload.get("ai_model", "")) or None,
            provider=str(payload.get("ai_provider", "")) or None,
        )
        ai_availability = openai_availability(
            ai_config.model, provider=ai_config.provider
        )
        ai_used = False
        ai_fallback_reason = None
        ai_audit = {
            "requested": mining_mode == "hybrid",
            "available": ai_availability.available,
            "provider": ai_availability.provider,
            "model": ai_availability.model,
            "base_url": ai_config.base_url,
            "config_source": ai_config.source,
            "proposed": 0,
            "valid": 0,
            "invalid": 0,
            "duplicate": 0,
            "rounds": 0,
            "prompt_version": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        if mining_mode == "hybrid" and not ai_availability.available:
            ai_fallback_reason = ai_availability.reason
        validated_ai = ()
        if mining_mode == "hybrid" and ai_availability.available:
            try:
                batch = OpenAIProposalClient(
                    str(ai_config.api_key), str(ai_availability.model),
                    base_url=ai_config.base_url,
                    provider=ai_config.provider,
                ).propose(
                    {
                        "proposal_count": ai_candidate_count,
                        "research_context": "single-asset BTCUSDT 1-hour factor discovery",
                        "available_fields": list(config.fields),
                        "allowed_windows": list(config.windows),
                        "allowed_unary_operators": list(config.unary_operators),
                        "allowed_window_operators": list(config.window_operators),
                        "allowed_binary_operators": list(config.binary_operators),
                        "max_depth": config.max_depth,
                        "max_complexity": config.max_complexity,
                    }
                )
                validated_ai, audit = validate_proposal_batch(
                    batch,
                    allowed_fields=config.fields,
                    allowed_windows=config.windows,
                    max_depth=config.max_depth,
                    max_complexity=config.max_complexity,
                    allowed_operators=(
                        *config.unary_operators,
                        *config.window_operators,
                        *config.binary_operators,
                    ),
                )
                ai_audit.update(
                    {
                        "proposed": audit.proposed,
                        "valid": audit.valid,
                        "invalid": audit.invalid,
                        "duplicate": audit.duplicate,
                        "rounds": 1,
                        "prompt_version": batch.prompt_version,
                        "input_tokens": batch.input_tokens,
                        "output_tokens": batch.output_tokens,
                        "errors": list(audit.errors),
                    }
                )
                if validated_ai:
                    ai_used = True
                else:
                    ai_fallback_reason = "OpenAI returned no locally valid factor proposals"
            except AIClientError as error:
                ai_fallback_reason = str(error)

        deterministic = generate_expressions(config)
        if validated_ai:
            deterministic_budget = max(0, max_candidates - len(validated_ai))
            expressions, source_metadata = merge_candidate_sources(
                deterministic[:deterministic_budget], validated_ai
            )
            if len(expressions) < max_candidates:
                existing = {expression.expression_id for expression in expressions}
                expressions_list = list(expressions)
                for expression in deterministic[deterministic_budget:]:
                    if expression.expression_id in existing:
                        metadata = source_metadata[expression.expression_id]
                        if metadata.get("source") == "ai":
                            metadata["source"] = "deterministic"
                            metadata["also_proposed_by_ai"] = True
                        continue
                    expressions_list.append(expression)
                    source_metadata[expression.expression_id] = {"source": "deterministic"}
                    existing.add(expression.expression_id)
                    if len(expressions_list) == max_candidates:
                        break
                expressions = tuple(expressions_list)
        else:
            expressions, source_metadata = merge_candidate_sources(deterministic, ())
        split = TimeSplit(train_ratio, validation_ratio, holdout_ratio)
        cost_model = CostModel(
            fee_bps=float(payload.get("fee_bps", 4.0)),
            half_spread_bps=float(payload.get("half_spread_bps", 1.0)),
            slippage_bps=float(payload.get("slippage_bps", 1.0)),
        )
        result = mine_candidates(
            expressions,
            data,
            forward_return(data["close"], forward_horizon),
            split=split,
            purge_bars=forward_horizon,
            min_observations=int(payload.get("min_observations", 20)),
            stability_windows=int(payload.get("stability_windows", 8)),
            hac_lags=int(payload.get("hac_lags", 1)),
            fdr_q=fdr_q,
        )
        raw_min_abs_ic = payload.get("min_validation_abs_rank_ic", payload.get("min_validation_rank_ic"))
        raw_min_directional = payload.get(
            "min_stability_directional_ratio",
            payload.get("min_stability_positive_ratio"),
        )
        policy = SelectionPolicy(
            min_validation_abs_rank_ic=max(0.0, float(raw_min_abs_ic)) if raw_min_abs_ic not in (None, "") else None,
            min_stability_directional_ratio=float(raw_min_directional) if raw_min_directional not in (None, "") else None,
            min_validation_sharpe=float(payload["min_validation_sharpe"]) if payload.get("min_validation_sharpe") not in (None, "") else None,
            max_validation_turnover=float(payload["max_validation_turnover"]) if payload.get("max_validation_turnover") not in (None, "") else None,
            require_fdr_pass=True,
        )
        table = screen_candidates(result.table, policy)
        for column in (
            "source", "also_proposed_by_ai", "ai_hypothesis", "ai_family", "ai_round"
        ):
            table[column] = table["expression_id"].map(
                lambda expression_id: source_metadata.get(str(expression_id), {}).get(column)
            )
        result = result.__class__(table=table, train_end=result.train_end, validation_end=result.validation_end, purge_bars=result.purge_bars)
        run_id = self.catalog.save_result(
            result,
            dataset_id=self.research_dataset_id,
            config={
                "protocol_version": "2.3",
                "mining_mode_requested": mining_mode,
                "mining_mode_effective": "hybrid" if ai_used else "formula",
                "ai_used": ai_used,
                "ai_candidate_count": ai_candidate_count,
                "ai_fallback_reason": ai_fallback_reason,
                "ai_audit": ai_audit,
                "dataset_sha256": self._file_sha256(self.dataset_path),
                "manifest_sha256": self._file_sha256(self.manifest_path),
                "funding_sha256": self._file_sha256(self.funding_path),
                "runtime": {
                    "python": sys.version.split()[0],
                    "pandas": pd.__version__,
                    "project_version": "0.1.0",
                },
                "forward_horizon": forward_horizon,
                "research_start": data.index[0].isoformat(),
                "research_end": data.index[-1].isoformat(),
                "research_bars": len(data),
                "min_observations": int(payload.get("min_observations", 20)),
                "stability_windows": int(payload.get("stability_windows", 8)),
                "hac_lags": int(payload.get("hac_lags", 1)),
                "fdr_q": fdr_q,
                "entry_quantile": entry_quantile,
                "generator": config.__dict__,
                "split": split.__dict__,
                "cost_model": {"fee_bps": cost_model.fee_bps, "half_spread_bps": cost_model.half_spread_bps, "slippage_bps": cost_model.slippage_bps},
                "selection_policy": policy.__dict__,
            },
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = f"api_btc_factor_mining_{run_id}"
        table_path = self.report_dir / f"{stem}.csv"
        json_path = self.report_dir / f"{stem}.json"
        table.to_csv(table_path, index=False)
        safe_rows = [
            self._candidate_for_api(row)
            for row in self._safe_rows(table.head(50).to_dict(orient="records"))
        ]
        summary = {"run_id": run_id, "protocol_version": "2.3", "mining_mode_requested": mining_mode, "mining_mode_effective": "hybrid" if ai_used else "formula", "ai_used": ai_used, "ai_fallback_reason": ai_fallback_reason, "ai_audit": ai_audit, "candidate_count": len(table), "selected_count": int(table["screen_status"].eq("selected").sum()), "eligible_count": int(table["screen_status"].eq("eligible").sum()), "duplicate_count": int(table["status"].eq("duplicate").sum()), "rejected_count": int(table["status"].isin(["rejected", "fdr_rejected"]).sum()), "candidates": safe_rows, "table": str(table_path.relative_to(self.project_root)), "catalog": "data/state/factor_catalog.sqlite3"}
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["report"] = str(json_path.relative_to(self.project_root))
        return summary

    def run_promotion(self, payload: dict[str, object]) -> dict[str, object]:
        run_id = str(payload["run_id"])
        expression_id = str(payload["expression_id"])
        factor_role = str(payload["factor_role"])
        signal_semantics = str(payload["signal_semantics"])
        self._validate_factor_semantics(factor_role, signal_semantics)
        if factor_role != "direction":
            raise ValueError("risk and condition factors require a specialized confirmation protocol")
        candidate = self.catalog.get_candidate(run_id, expression_id)
        if not candidate:
            raise KeyError("candidate not found")
        mining_run = self.catalog.get_run(run_id)
        if not mining_run:
            raise KeyError("mining run not found")
        mining_config = dict(mining_run.get("config", {}))
        if mining_config.get("protocol_version") != "2.3":
            raise ValueError("final holdout confirmation requires a protocol 2.3 mining run")
        forward_horizon = int(mining_config.get("forward_horizon", 4))
        entry_quantile = float(mining_config.get("entry_quantile", 0.8))
        cost_config = dict(mining_config.get("cost_model", {}))
        expression = Expression.from_dict(json.loads(str(candidate["canonical"])))
        all_data = load_normalized_ohlcv(self.dataset_path, expected_frequency="1h")
        data = self._research_window(all_data, mining_config)
        funding = load_funding_rates(self.funding_path)
        target = forward_return(data["close"], forward_horizon)
        factor = evaluate_expression(expression, data)
        split_config = dict(mining_config.get("split", {}))
        split = TimeSplit(
            float(split_config.get("train_ratio", 0.6)),
            float(split_config.get("validation_ratio", 0.2)),
            float(split_config.get("holdout_ratio", 0.2)),
        )
        _, validation_slice, holdout_slice = split.segments(len(data), forward_horizon)
        holdout_start = data.index[holdout_slice.start].isoformat()
        holdout_end = data.index[holdout_slice.stop - 1].isoformat()
        latest_revealed = self.catalog.latest_revealed_holdout_end(
            str(mining_run.get("dataset_id", ""))
        )
        if latest_revealed is not None and latest_revealed >= holdout_start:
            raise ValueError(
                "holdout period overlaps previously revealed data; add a non-overlapping unseen period before final confirmation"
            )
        direction_multiplier = -1.0 if signal_semantics == "inverse" else 1.0
        aligned_factor = factor.mul(direction_multiplier) if factor_role == "direction" else factor
        holdout_report = evaluate_factor(
            factor.iloc[holdout_slice], target.iloc[holdout_slice]
        )
        aligned_holdout_report = evaluate_factor(
            aligned_factor.iloc[holdout_slice], target.iloc[holdout_slice]
        )
        random_benchmark = random_rank_ic_benchmark(
            target.iloc[holdout_slice],
            trials=100,
            seed=int(expression_id[:8], 16),
        )
        walk_forward = walk_forward_factor_evaluation(
            aligned_factor.iloc[:validation_slice.stop],
            target.iloc[:validation_slice.stop],
            train_bars=24 * 365,
            test_bars=24 * 90,
            purge_bars=forward_horizon,
        )
        cost_model = CostModel(
            fee_bps=float(cost_config.get("fee_bps", 4.0)),
            half_spread_bps=float(cost_config.get("half_spread_bps", 1.0)),
            slippage_bps=float(cost_config.get("slippage_bps", 1.0)),
        )
        strategy = None
        buy_hold = None
        entry_lower = None
        entry_upper = None
        active_position_ratio = None
        if factor_role == "direction":
            development_factor = aligned_factor.iloc[:validation_slice.stop]
            holdout_factor = aligned_factor.iloc[holdout_slice]
            position, entry_lower, entry_upper = quantile_positions(
                development_factor, holdout_factor, entry_quantile
            )
            position = rebalance_every(position, forward_horizon)
            active_position_ratio = float(position.ne(0).mean())
            holdout_close = data["close"].iloc[holdout_slice]
            holdout_funding = funding.reindex(holdout_close.index)
            strategy = backtest_positions(holdout_close, position, cost_model, 24 * 365, holdout_funding)
            buy_hold = backtest_positions(
                holdout_close,
                pd.Series(1.0, index=holdout_close.index),
                cost_model,
                24 * 365,
                holdout_funding,
            )
        exceeds_random = bool(
            aligned_holdout_report.rank_ic > random_benchmark.absolute_rank_ic_95
        )
        if factor_role == "direction":
            confirmation_status = (
                "confirmed"
                if exceeds_random
                and strategy is not None
                and strategy.net_return > 0
                and strategy.net_return - buy_hold.net_return > 0
                else "not_confirmed"
            )
        else:
            confirmation_status = "awaiting_specialized_protocol"
        stem = f"api_promoted_factor_{run_id}_{expression_id}"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        table_path = self.report_dir / f"{stem}.csv"
        json_path = self.report_dir / f"{stem}.json"
        pd.DataFrame(
            [{"factor": expression.formula(), "formula_zh": expression.formula_zh(), "factor_role": factor_role, "signal_semantics": signal_semantics, "confirmation_status": confirmation_status, "holdout_start": data.index[holdout_slice.start].isoformat(), "holdout_end": data.index[holdout_slice.stop - 1].isoformat(), "holdout_observations": holdout_report.observations, "holdout_raw_linear_ic": holdout_report.linear_ic, "holdout_raw_rank_ic": holdout_report.rank_ic, "holdout_aligned_rank_ic": aligned_holdout_report.rank_ic, "random_null_trials": random_benchmark.trials, "random_null_mean_rank_ic": random_benchmark.mean_rank_ic, "random_null_std_rank_ic": random_benchmark.std_rank_ic, "random_null_abs_rank_ic_95": random_benchmark.absolute_rank_ic_95, "rank_ic_exceeds_random_95": exceeds_random, "development_mean_test_rank_ic": walk_forward.mean_test_rank_ic, "development_positive_test_rank_ic_ratio": walk_forward.positive_test_rank_ic_ratio, "rebalance_bars": forward_horizon, "entry_quantile": entry_quantile if strategy else None, "entry_lower": entry_lower, "entry_upper": entry_upper, "active_position_ratio": active_position_ratio, "gross_return": strategy.gross_return if strategy else None, "net_return": strategy.net_return if strategy else None, "annualized_sharpe": strategy.annualized_sharpe if strategy else None, "max_drawdown": strategy.max_drawdown if strategy else None, "total_turnover": strategy.total_turnover if strategy else None, "buy_hold_gross_return": buy_hold.gross_return if buy_hold else None, "buy_hold_net_return": buy_hold.net_return if buy_hold else None, "buy_hold_annualized_sharpe": buy_hold.annualized_sharpe if buy_hold else None, "buy_hold_max_drawdown": buy_hold.max_drawdown if buy_hold else None, "excess_net_return_vs_buy_hold": strategy.net_return - buy_hold.net_return if strategy and buy_hold else None}]
        ).to_csv(table_path, index=False)
        result = {
            "run_id": run_id,
            "expression_id": expression_id,
            "formula": expression.formula(),
            "formula_zh": expression.formula_zh(),
            "factor_role": factor_role,
            "signal_semantics": signal_semantics,
            "evaluation_mode": "direction_backtest" if strategy else "statistical_only",
            "confirmation_status": confirmation_status,
            "forward_horizon": forward_horizon,
            "rebalance_bars": forward_horizon if strategy else None,
            "entry_quantile": entry_quantile if strategy else None,
            "entry_lower": self._safe_number(entry_lower),
            "entry_upper": self._safe_number(entry_upper),
            "active_position_ratio": self._safe_number(active_position_ratio),
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "holdout_observations": holdout_report.observations,
            "holdout_raw_linear_ic": self._safe_number(holdout_report.linear_ic),
            "holdout_raw_rank_ic": self._safe_number(holdout_report.rank_ic),
            "holdout_aligned_rank_ic": self._safe_number(aligned_holdout_report.rank_ic),
            "random_null_trials": random_benchmark.trials,
            "random_null_mean_rank_ic": self._safe_number(random_benchmark.mean_rank_ic),
            "random_null_std_rank_ic": self._safe_number(random_benchmark.std_rank_ic),
            "random_null_abs_rank_ic_95": self._safe_number(random_benchmark.absolute_rank_ic_95),
            "rank_ic_exceeds_random_95": exceeds_random,
            "development_mean_test_rank_ic": self._safe_number(walk_forward.mean_test_rank_ic),
            "development_positive_test_rank_ic_ratio": self._safe_number(walk_forward.positive_test_rank_ic_ratio),
            "gross_return": self._safe_number(strategy.gross_return) if strategy else None,
            "net_return": self._safe_number(strategy.net_return) if strategy else None,
            "annualized_sharpe": self._safe_number(strategy.annualized_sharpe) if strategy else None,
            "max_drawdown": self._safe_number(strategy.max_drawdown) if strategy else None,
            "total_turnover": self._safe_number(strategy.total_turnover) if strategy else None,
            "buy_hold_gross_return": self._safe_number(buy_hold.gross_return) if buy_hold else None,
            "buy_hold_net_return": self._safe_number(buy_hold.net_return) if buy_hold else None,
            "buy_hold_annualized_sharpe": self._safe_number(buy_hold.annualized_sharpe) if buy_hold else None,
            "buy_hold_max_drawdown": self._safe_number(buy_hold.max_drawdown) if buy_hold else None,
            "excess_net_return_vs_buy_hold": self._safe_number(strategy.net_return - buy_hold.net_return) if strategy and buy_hold else None,
            "report": str(json_path.relative_to(self.project_root)),
            "table": str(table_path.relative_to(self.project_root)),
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.catalog.mark_promoted(
            run_id,
            expression_id,
            str(payload.get("task_id", "")),
            factor_role,
            signal_semantics,
            result,
        )
        return result

    @staticmethod
    def _safe_number(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _candidate_for_api(candidate: dict[str, object]) -> dict[str, object]:
        result = dict(candidate)
        try:
            expression = Expression.from_dict(json.loads(str(result["canonical"])))
            result["formula_zh"] = expression.formula_zh()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            result["formula_zh"] = result.get("formula")
        if pd.isna(result.get("validation_stability_directional_ratio")):
            positive_ratio = result.get("validation_stability_positive_ratio")
            rank_ic = result.get("validation_rank_ic")
            if pd.notna(positive_ratio) and pd.notna(rank_ic):
                result["validation_stability_directional_ratio"] = (
                    float(positive_ratio) if float(rank_ic) >= 0 else 1.0 - float(positive_ratio)
                )
        rank_ic = result.get("validation_rank_ic")
        if pd.notna(rank_ic):
            result.setdefault("validation_abs_rank_ic", abs(float(rank_ic)))
            result.setdefault("suggested_signal_semantics", "direct" if float(rank_ic) >= 0 else "inverse")
        if result.get("promotion_status") == "promoted":
            return result
        return {
            key: value
            for key, value in result.items()
            if not key.startswith("holdout_")
        }

    @staticmethod
    def _safe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        safe: list[dict[str, object]] = []
        for row in rows:
            item: dict[str, object] = {}
            for key, value in row.items():
                if pd.isna(value) if not isinstance(value, (list, dict)) else False:
                    item[key] = None
                elif hasattr(value, "item"):
                    item[key] = value.item()
                else:
                    item[key] = value
            safe.append(item)
        return safe

    def _latest_report_rows(self) -> list[dict[str, object]]:
        path = self.report_dir / "btcusdt_1h_factor_suite_24.csv"
        if not path.exists():
            return []
        rows = pd.read_csv(path).to_dict(orient="records")
        return self._safe_rows(rows)

    @staticmethod
    def _task_dict(task: TaskRecord) -> dict[str, object]:
        return {
            "task_id": task.task_id,
            "source": "web_task",
            "status": task.status,
            "submitted_at": task.submitted_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "payload": task.payload,
            "result": task.result,
            "error": task.error,
        }
