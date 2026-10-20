"""SQLite persistence for background research tasks."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from threading import Lock


class TaskStore:
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
                    CREATE TABLE IF NOT EXISTS research_tasks (
                        task_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        status TEXT NOT NULL,
                        submitted_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        payload_json TEXT NOT NULL,
                        result_json TEXT,
                        error TEXT
                    )
                    """
                )

    def save(self, record: dict[str, object]) -> None:
        with self.lock, closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO research_tasks (
                        task_id, source, status, submitted_at, started_at,
                        finished_at, payload_json, result_json, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        status=excluded.status,
                        started_at=excluded.started_at,
                        finished_at=excluded.finished_at,
                        payload_json=excluded.payload_json,
                        result_json=excluded.result_json,
                        error=excluded.error
                    """,
                    (
                        record["task_id"],
                        record.get("source", "web_task"),
                        record["status"],
                        record["submitted_at"],
                        record.get("started_at"),
                        record.get("finished_at"),
                        json.dumps(record.get("payload", {}), ensure_ascii=False),
                        json.dumps(record["result"], ensure_ascii=False) if record.get("result") is not None else None,
                        record.get("error"),
                    ),
                )

    def get(self, task_id: str) -> dict[str, object] | None:
        with self.lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM research_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def list(self) -> list[dict[str, object]]:
        with self.lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM research_tasks ORDER BY submitted_at DESC"
            ).fetchall()
        return [self._decode(row) for row in rows]

    def fail_interrupted(self, finished_at: str) -> int:
        with self.lock, closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE research_tasks
                    SET status = 'failed', finished_at = ?, error = ?
                    WHERE status IN ('queued', 'running')
                    """,
                    (finished_at, "service restarted before task completion"),
                )
        return cursor.rowcount

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, object]:
        return {
            "task_id": row["task_id"],
            "source": row["source"],
            "status": row["status"],
            "submitted_at": row["submitted_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
        }
