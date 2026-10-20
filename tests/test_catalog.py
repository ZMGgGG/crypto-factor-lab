from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from factor_lab.catalog import CandidateCatalog
from factor_lab.expression import call, field
from factor_lab.mining import TimeSplit, mine_candidates


class CandidateCatalogTest(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[pd.DataFrame, pd.Series]:
        index = pd.date_range("2025-01-01", periods=120, freq="h", tz="UTC")
        close = pd.Series(100.0 + np.arange(120, dtype=float), index=index)
        return (
            pd.DataFrame({"close": close, "volume": np.linspace(10, 20, 120)}, index=index),
            close.shift(-1).div(close).sub(1).rename("forward_return_1"),
        )

    def test_mining_result_survives_catalog_recreation(self) -> None:
        data, target = self.fixture()
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            data,
            target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            run_id = CandidateCatalog(path).save_result(
                result,
                run_id="run-1",
                dataset_id="btc-test",
                config={"windows": [1], "penalty": 0.001},
            )
            restored = CandidateCatalog(path).get_run(run_id)

        self.assertEqual(restored["run_id"], "run-1")
        self.assertEqual(restored["dataset_id"], "btc-test")
        self.assertEqual(restored["config"]["windows"], [1])
        self.assertEqual(restored["candidate_count"], 1)
        self.assertEqual(restored["candidates"][0]["status"], "valid")

    def test_catalog_keeps_rejection_reason_and_run_list(self) -> None:
        data, target = self.fixture()
        result = mine_candidates(
            (field("close"),),
            data,
            target,
            min_observations=10,
            correlation_threshold=None,
        )

        with TemporaryDirectory() as directory:
            catalog = CandidateCatalog(Path(directory) / "catalog.sqlite3")
            run_id = catalog.save_result(result, dataset_id="btc-test")
            run = catalog.get_run(run_id)

            self.assertEqual(run["candidates"][0]["status"], "rejected")
            self.assertIn("temporal operator", run["candidates"][0]["error"])
            runs = catalog.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["candidate_count"], 1)
            self.assertEqual(runs[0]["promoted_count"], 0)
            self.assertEqual(runs[0]["protocol_version"], "legacy")

    def test_only_promoted_candidate_can_enter_factor_library(self) -> None:
        data, target = self.fixture()
        result = mine_candidates(
            (call("return", field("close"), window=1),),
            data,
            target,
            split=TimeSplit(0.5, 0.25, 0.25),
            min_observations=10,
            correlation_threshold=None,
        )
        with TemporaryDirectory() as directory:
            catalog = CandidateCatalog(Path(directory) / "catalog.sqlite3")
            run_id = catalog.save_result(result, run_id="run-library")
            expression_id = result.table.iloc[0]["expression_id"]
            with self.assertRaisesRegex(ValueError, "complete promotion"):
                catalog.add_to_library(
                    factor_id="mined_return_1",
                    run_id=run_id,
                    expression_id=expression_id,
                )
            catalog.mark_promoted(
                run_id,
                expression_id,
                "task-1",
                "direction",
                "direct",
                {"confirmation_status": "not_confirmed"},
            )
            with self.assertRaisesRegex(ValueError, "final holdout confirmation"):
                catalog.add_to_library(
                    factor_id="mined_return_1",
                    run_id=run_id,
                    expression_id=expression_id,
                )
            catalog.mark_promoted(
                run_id,
                expression_id,
                "task-2",
                "direction",
                "direct",
                {"confirmation_status": "confirmed"},
            )
            registered = catalog.add_to_library(
                factor_id="mined_return_1",
                run_id=run_id,
                expression_id=expression_id,
            )
            self.assertEqual(registered["factor_id"], "mined_return_1")
            self.assertEqual(len(catalog.list_library()), 1)
            with self.assertRaisesRegex(ValueError, "already registered"):
                catalog.add_to_library(
                    factor_id="mined_return_1_copy",
                    run_id=run_id,
                    expression_id=expression_id,
                )

    def test_observation_track_survives_catalog_recreation(self) -> None:
        data, target = self.fixture()
        result = mine_candidates(
            (call("return", field("close"), window=1),), data, target,
            split=TimeSplit(0.5, 0.25, 0.25), min_observations=10,
            correlation_threshold=None,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.sqlite3"
            catalog = CandidateCatalog(path)
            run_id = catalog.save_result(result, run_id="track-run")
            expression_id = result.table.iloc[0]["expression_id"]
            track = catalog.create_track(
                run_id=run_id, expression_id=expression_id,
                signal_semantics="inverse", forward_horizon=4,
                entry_quantile=0.8, entry_lower=-0.1, entry_upper=0.1,
                cost_model={"fee_bps": 4.0},
            )
            catalog.update_track(track["track_id"], {"net_return": 0.1})
            restored = CandidateCatalog(path).get_track(track["track_id"])
        self.assertEqual(restored["signal_semantics"], "inverse")
        self.assertEqual(restored["result"]["net_return"], 0.1)
