"""Persistent catalog for factor-mining runs and candidate decisions."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from factor_lab.mining import MiningResult


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class CandidateCatalog:
    """SQLite-backed audit log for every candidate tested by a mining run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mining_runs (
                        run_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        dataset_id TEXT,
                        train_end TEXT NOT NULL,
                        validation_end TEXT NOT NULL,
                        purge_bars INTEGER NOT NULL,
                        config_json TEXT NOT NULL,
                        candidate_count INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mining_candidates (
                        run_id TEXT NOT NULL,
                        expression_id TEXT NOT NULL,
                        rank_order INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        formula TEXT NOT NULL,
                        canonical TEXT NOT NULL,
                        duplicate_of TEXT,
                        duplicate_correlation REAL,
                        metrics_json TEXT NOT NULL,
                        PRIMARY KEY (run_id, expression_id),
                        FOREIGN KEY (run_id) REFERENCES mining_runs(run_id)
                    )
                    """
                )
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(mining_candidates)")}
                for column, definition in {
                    "promotion_status": "TEXT NOT NULL DEFAULT 'candidate'",
                    "promotion_task_id": "TEXT",
                    "promoted_at": "TEXT",
                    "promotion_role": "TEXT",
                    "promotion_signal_semantics": "TEXT",
                    "promotion_result_json": "TEXT",
                }.items():
                    if column not in columns:
                        connection.execute(f"ALTER TABLE mining_candidates ADD COLUMN {column} {definition}")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS factor_library (
                        factor_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        expression_id TEXT NOT NULL,
                        formula TEXT NOT NULL,
                        canonical TEXT NOT NULL,
                        factor_role TEXT NOT NULL,
                        signal_semantics TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        promotion_task_id TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE (run_id, expression_id),
                        FOREIGN KEY (run_id, expression_id)
                            REFERENCES mining_candidates(run_id, expression_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS observation_tracks (
                        track_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        expression_id TEXT NOT NULL,
                        signal_semantics TEXT NOT NULL,
                        forward_horizon INTEGER NOT NULL,
                        entry_quantile REAL NOT NULL,
                        entry_lower REAL NOT NULL,
                        entry_upper REAL NOT NULL,
                        observation_bars INTEGER NOT NULL DEFAULT 2160,
                        tracking_start TEXT,
                        tracking_end TEXT,
                        cost_model_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (run_id, expression_id),
                        FOREIGN KEY (run_id, expression_id)
                            REFERENCES mining_candidates(run_id, expression_id)
                    )
                    """
                )
                track_columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(observation_tracks)")
                }
                if "observation_bars" not in track_columns:
                    connection.execute(
                        "ALTER TABLE observation_tracks ADD COLUMN observation_bars INTEGER NOT NULL DEFAULT 2160"
                    )
                if "tracking_start" not in track_columns:
                    connection.execute(
                        "ALTER TABLE observation_tracks ADD COLUMN tracking_start TEXT"
                    )
                if "tracking_end" not in track_columns:
                    connection.execute(
                        "ALTER TABLE observation_tracks ADD COLUMN tracking_end TEXT"
                    )

    def save_result(
        self,
        result: MiningResult,
        *,
        run_id: str | None = None,
        dataset_id: str | None = None,
        config: Mapping[str, object] | None = None,
        created_at: str | None = None,
    ) -> str:
        run_id = run_id or uuid4().hex
        created_at = created_at or datetime.now(UTC).isoformat()
        config_payload = _json_safe(config or {})
        rows = result.table.to_dict(orient="records")
        with self.lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO mining_runs (
                        run_id, created_at, dataset_id, train_end, validation_end,
                        purge_bars, config_json, candidate_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        dataset_id=excluded.dataset_id,
                        train_end=excluded.train_end,
                        validation_end=excluded.validation_end,
                        purge_bars=excluded.purge_bars,
                        config_json=excluded.config_json,
                        candidate_count=excluded.candidate_count
                    """,
                    (
                        run_id,
                        created_at,
                        dataset_id,
                        str(result.train_end),
                        str(result.validation_end),
                        result.purge_bars,
                        json.dumps(config_payload, ensure_ascii=False, allow_nan=False),
                        len(rows),
                    ),
                )
                connection.execute("DELETE FROM mining_candidates WHERE run_id = ?", (run_id,))
                connection.executemany(
                    """
                    INSERT INTO mining_candidates (
                        run_id, expression_id, rank_order, status, formula, canonical,
                        duplicate_of, duplicate_correlation, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            str(row.get("expression_id", "")),
                            index,
                            str(row.get("status", "")),
                            str(row.get("formula", "")),
                            str(row.get("canonical", "")),
                            row.get("duplicate_of"),
                            _json_safe(row.get("duplicate_correlation")),
                            json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False),
                        )
                        for index, row in enumerate(rows)
                    ],
                )
        return run_id

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            run = connection.execute("SELECT * FROM mining_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            candidates = connection.execute(
                """
                SELECT c.*, f.factor_id AS library_factor_id,
                       f.factor_role AS library_factor_role,
                       f.signal_semantics AS library_signal_semantics
                FROM mining_candidates c
                LEFT JOIN factor_library f
                  ON f.run_id = c.run_id AND f.expression_id = c.expression_id
                WHERE c.run_id = ? ORDER BY c.rank_order
                """,
                (run_id,),
            ).fetchall()
        return {
            "run_id": run["run_id"],
            "created_at": run["created_at"],
            "dataset_id": run["dataset_id"],
            "train_end": run["train_end"],
            "validation_end": run["validation_end"],
            "purge_bars": run["purge_bars"],
            "config": json.loads(run["config_json"]),
            "candidate_count": run["candidate_count"],
            "candidates": [self._candidate_payload(candidate) for candidate in candidates],
        }

    @staticmethod
    def _candidate_payload(row: sqlite3.Row) -> dict[str, object]:
        payload = json.loads(row["metrics_json"])
        payload["promotion_status"] = row["promotion_status"]
        payload["promotion_task_id"] = row["promotion_task_id"]
        payload["promoted_at"] = row["promoted_at"]
        payload["promotion_role"] = row["promotion_role"]
        payload["promotion_signal_semantics"] = row["promotion_signal_semantics"]
        payload["promotion_result"] = json.loads(row["promotion_result_json"]) if row["promotion_result_json"] else None
        if "library_factor_id" in row.keys():
            payload["library_factor_id"] = row["library_factor_id"]
            payload["library_factor_role"] = row["library_factor_role"]
            payload["library_signal_semantics"] = row["library_signal_semantics"]
        return payload

    def get_candidate(self, run_id: str, expression_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT c.*, f.factor_id AS library_factor_id,
                       f.factor_role AS library_factor_role,
                       f.signal_semantics AS library_signal_semantics
                FROM mining_candidates c
                LEFT JOIN factor_library f
                  ON f.run_id = c.run_id AND f.expression_id = c.expression_id
                WHERE c.run_id = ? AND c.expression_id = ?
                """,
                (run_id, expression_id),
            ).fetchone()
        return self._candidate_payload(row) if row else None

    def latest_revealed_holdout_end(self, dataset_id: str) -> str | None:
        """Return the latest holdout end ever written for this dataset."""
        with self.lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT c.promotion_result_json
                FROM mining_candidates c
                JOIN mining_runs r ON r.run_id = c.run_id
                WHERE r.dataset_id = ? AND c.promotion_result_json IS NOT NULL
                """,
                (dataset_id,),
            ).fetchall()
        ends = []
        for row in rows:
            result = json.loads(row["promotion_result_json"])
            if result.get("holdout_end"):
                ends.append(str(result["holdout_end"]))
        return max(ends) if ends else None

    def create_track(
        self,
        *,
        run_id: str,
        expression_id: str,
        signal_semantics: str,
        forward_horizon: int,
        entry_quantile: float,
        entry_lower: float,
        entry_upper: float,
        cost_model: Mapping[str, object],
        observation_bars: int = 2160,
        tracking_start: str | None = None,
        tracking_end: str | None = None,
    ) -> dict[str, object]:
        track_id = uuid4().hex
        now = datetime.now(UTC).isoformat()
        with self.lock, closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO observation_tracks (
                            track_id, run_id, expression_id, signal_semantics,
                            forward_horizon, entry_quantile, entry_lower, entry_upper,
                            observation_bars, tracking_start, tracking_end, cost_model_json, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'observing', ?, ?)
                        """,
                        (
                            track_id, run_id, expression_id, signal_semantics,
                            forward_horizon, entry_quantile, entry_lower, entry_upper,
                            observation_bars,
                            tracking_start,
                            tracking_end,
                            json.dumps(_json_safe(cost_model), ensure_ascii=False), now, now,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError("candidate is already being observed") from error
        return self.get_track(track_id) or {}

    def update_track(self, track_id: str, result: Mapping[str, object]) -> dict[str, object]:
        now = datetime.now(UTC).isoformat()
        with self.lock, closing(self._connect()) as connection:
            with connection:
                updated = connection.execute(
                    "UPDATE observation_tracks SET result_json = ?, status = ?, updated_at = ? WHERE track_id = ?",
                    (json.dumps(_json_safe(result), ensure_ascii=False, allow_nan=False), str(result.get("tracking_status", "observing")), now, track_id),
                )
                if updated.rowcount != 1:
                    raise KeyError("observation track not found")
        return self.get_track(track_id) or {}

    @staticmethod
    def _track_payload(row: sqlite3.Row) -> dict[str, object]:
        item = dict(row)
        item["cost_model"] = json.loads(str(item.pop("cost_model_json")))
        item["result"] = json.loads(item.pop("result_json")) if item["result_json"] else None
        return item

    def get_track(self, track_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM observation_tracks WHERE track_id = ?", (track_id,)
            ).fetchone()
        return self._track_payload(row) if row else None

    def get_track_for_candidate(self, run_id: str, expression_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM observation_tracks WHERE run_id = ? AND expression_id = ?",
                (run_id, expression_id),
            ).fetchone()
        return self._track_payload(row) if row else None

    def list_tracks(self) -> list[dict[str, object]]:
        with self.lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM observation_tracks ORDER BY created_at DESC"
            ).fetchall()
        return [self._track_payload(row) for row in rows]

    def mark_promoted(
        self,
        run_id: str,
        expression_id: str,
        task_id: str,
        factor_role: str,
        signal_semantics: str,
        result: Mapping[str, object] | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self.lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "UPDATE mining_candidates SET promotion_status = 'promoted', promotion_task_id = ?, promoted_at = ?, promotion_role = ?, promotion_signal_semantics = ?, promotion_result_json = ? WHERE run_id = ? AND expression_id = ?",
                    (task_id, now, factor_role, signal_semantics, json.dumps(_json_safe(result or {}), ensure_ascii=False, allow_nan=False), run_id, expression_id),
                )

    def add_to_library(
        self,
        *,
        factor_id: str,
        run_id: str,
        expression_id: str,
    ) -> dict[str, object]:
        candidate = self.get_candidate(run_id, expression_id)
        if not candidate:
            raise KeyError("candidate not found")
        if candidate.get("promotion_status") != "promoted":
            raise ValueError("candidate must complete promotion before library registration")
        promotion_result = candidate.get("promotion_result") or {}
        if promotion_result.get("confirmation_status") != "confirmed":
            raise ValueError("candidate must pass final holdout confirmation before library registration")
        factor_role = candidate.get("promotion_role")
        signal_semantics = candidate.get("promotion_signal_semantics")
        if not factor_role or not signal_semantics:
            raise ValueError("promoted candidate is missing factor role or signal semantics")
        created_at = datetime.now(UTC).isoformat()
        with self.lock, closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO factor_library (
                            factor_id, run_id, expression_id, formula, canonical,
                            factor_role, signal_semantics, status,
                            promotion_task_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                        """,
                        (
                            factor_id,
                            run_id,
                            expression_id,
                            candidate["formula"],
                            candidate["canonical"],
                            factor_role,
                            signal_semantics,
                            candidate.get("promotion_task_id"),
                            created_at,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError("factor_id or candidate is already registered") from error
        return self.get_library_factor(factor_id)

    def get_library_factor(self, factor_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM factor_library WHERE factor_id = ?", (factor_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_library(self) -> list[dict[str, object]]:
        with self.lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM factor_library ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self) -> list[dict[str, object]]:
        with self.lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT r.run_id, r.created_at, r.dataset_id, r.train_end,
                       r.validation_end, r.purge_bars, r.candidate_count,
                       r.config_json,
                       COALESCE(SUM(CASE WHEN c.status = 'valid' THEN 1 ELSE 0 END), 0) AS valid_count,
                       COALESCE(SUM(CASE WHEN c.status = 'duplicate' THEN 1 ELSE 0 END), 0) AS duplicate_count,
                       COALESCE(SUM(CASE WHEN c.promotion_status = 'promoted' THEN 1 ELSE 0 END), 0) AS promoted_count
                FROM mining_runs r
                LEFT JOIN mining_candidates c ON c.run_id = r.run_id
                GROUP BY r.run_id
                ORDER BY r.created_at DESC
                """
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            config = json.loads(str(item.pop("config_json")))
            item["protocol_version"] = config.get("protocol_version", "legacy")
            result.append(item)
        return result
