from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from factor_lab.task_store import TaskStore


class TaskStoreTest(unittest.TestCase):
    def test_task_survives_store_recreation(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite3"
            first = TaskStore(path)
            first.save(
                {
                    "task_id": "task-1",
                    "source": "web_task",
                    "status": "succeeded",
                    "submitted_at": "2026-09-21T00:00:00+00:00",
                    "started_at": "2026-09-21T00:00:01+00:00",
                    "finished_at": "2026-09-21T00:00:02+00:00",
                    "payload": {"momentum_window": 24},
                    "result": {"factors": []},
                    "error": None,
                }
            )

            restored = TaskStore(path).get("task-1")

            self.assertEqual(restored["status"], "succeeded")
            self.assertEqual(restored["payload"]["momentum_window"], 24)

    def test_marks_interrupted_tasks_failed(self) -> None:
        with TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.sqlite3")
            store.save(
                {
                    "task_id": "task-2",
                    "status": "running",
                    "submitted_at": "2026-09-21T00:00:00+00:00",
                    "payload": {},
                }
            )

            changed = store.fail_interrupted("2026-09-21T01:00:00+00:00")
            restored = store.get("task-2")

            self.assertEqual(changed, 1)
            self.assertEqual(restored["status"], "failed")
            self.assertIn("restarted", restored["error"])
