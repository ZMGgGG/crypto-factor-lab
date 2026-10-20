from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from factor_lab.ai.proposals import AIProposal, AIProposalBatch
from factor_lab.expression import call, field
from factor_lab.service import ResearchService


class ResearchServiceTest(unittest.TestCase):
    def test_status_exposes_real_dataset(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"FACTOR_AI_ENV_FILE": ""}, clear=False
        ):
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
            )
            status = service.status()

            self.assertEqual(status["dataset"]["symbol"], "BTCUSDT")
            self.assertEqual(status["dataset"]["id"], service.research_dataset_id)
            self.assertGreaterEqual(status["dataset"]["bars"], 43824)
            self.assertIn("latest_report", status)
            self.assertIn("ai", status)
            self.assertFalse(status["ai"]["configured"])
            self.assertGreater(len(status["latest_report"]), 0)
            self.assertEqual(status["tasks"], [])
            json.dumps(status, allow_nan=False)
            service.executor.shutdown(wait=True)

    def test_chart_returns_real_points(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
            )
            chart = service.chart(window=24, points=20)

            self.assertEqual(len(chart["points"]), 20)
            self.assertIn("close", chart["points"][0])
            service.executor.shutdown(wait=True)

    def test_selected_candidate_can_be_observed_and_updated(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            mining = service.run_mining(
                {"max_candidates": 10, "min_validation_abs_rank_ic": 0.0,
                 "research_start": "2021-01-01", "research_end": "2025-12-31"}
            )
            candidate = next(
                row for row in mining["candidates"] if row.get("screen_status") == "selected"
            )

            with self.assertRaisesRegex(ValueError, "cannot be before"):
                service.create_observation_track(
                    mining["run_id"], candidate["expression_id"],
                    {"observation_start": "2024-01-01", "observation_end": "2024-01-30"},
                )

            track = service.create_observation_track(
                mining["run_id"], candidate["expression_id"],
                {"observation_start": "2025-02-01", "observation_end": "2025-02-10"},
            )
            updated = service.update_observation_track(str(track["track_id"]))

            self.assertEqual(track["result"]["tracking_status"], "completed")
            self.assertEqual(track["observation_bars"], 24 * 10)
            self.assertEqual(track["tracking_start"], "2025-02-01T00:00:00+00:00")
            self.assertEqual(track["tracking_end"], "2025-02-10T23:00:00+00:00")
            self.assertEqual(track["result"]["observations"], 24 * 10)
            self.assertEqual(track["result"]["observation_start"], "2025-02-01T00:00:00+00:00")
            self.assertIsNotNone(track["result"]["net_return"])
            self.assertIn(service.research_dataset_id, track["result"]["source_dataset_ids"])
            self.assertEqual(track["entry_lower"], updated["entry_lower"])
            self.assertEqual(track["entry_upper"], updated["entry_upper"])
            self.assertEqual(len(service.observation_tracks()), 1)
            observed_candidate = next(
                row
                for row in service.mining_run(mining["run_id"])["candidates"]
                if row["expression_id"] == candidate["expression_id"]
            )
            self.assertGreaterEqual(
                service.mining_candidate(mining["run_id"], candidate["expression_id"])[
                    "earliest_observation_start"
                ],
                "2025-01-01",
            )
            self.assertEqual(
                observed_candidate["observation_track"]["track_id"], track["track_id"]
            )
            with self.assertRaisesRegex(ValueError, "already being observed"):
                service.create_observation_track(
                    mining["run_id"], candidate["expression_id"]
                )
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_mining_returns_candidate_rows(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            result = service.run_mining({"max_candidates": 10, "forward_horizon": 4, "stability_windows": 2})

            self.assertEqual(result["candidate_count"], 10)
            self.assertIn("selected_count", result)
            self.assertGreater(len(result["candidates"]), 0)
            self.assertEqual(result["protocol_version"], "2.3")
            self.assertIn("formula_zh", result["candidates"][0])
            self.assertNotIn("holdout_rank_ic", result["candidates"][0])
            ineligible = next(row for row in result["candidates"] if row.get("screen_status") != "selected")
            with self.assertRaisesRegex(ValueError, "only selected"):
                service.submit_promotion(
                    result["run_id"],
                    ineligible["expression_id"],
                    {"factor_role": "direction", "signal_semantics": "direct"},
                )
            stored_run = service.catalog.get_run(result["run_id"])
            self.assertEqual(len(stored_run["config"]["dataset_sha256"]), 64)
            self.assertIn("python", stored_run["config"]["runtime"])
            self.assertEqual(stored_run["config"]["fdr_q"], 0.05)
            self.assertEqual(stored_run["config"]["entry_quantile"], 0.8)
            api_run = service.mining_run(result["run_id"])
            self.assertTrue(api_run["final_confirmation_available"])
            self.assertTrue(api_run["candidates"][0]["final_confirmation_available"])
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_mining_uses_selected_research_interval(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            result = service.run_mining(
                {
                    "max_candidates": 5,
                    "research_start": "2023-01-01",
                    "research_end": "2024-12-31",
                }
            )
            config = service.catalog.get_run(result["run_id"])["config"]
            self.assertEqual(config["research_start"], "2023-01-01T00:00:00+00:00")
            self.assertEqual(config["research_end"], "2024-12-31T23:00:00+00:00")
            self.assertEqual(config["research_bars"], 24 * 731)
            with self.assertRaisesRegex(ValueError, "research_end"):
                service.run_mining(
                    {
                        "max_candidates": 5,
                        "research_start": "2024-01-01",
                        "research_end": "2023-12-31",
                    }
                )
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_hybrid_mining_falls_back_without_api_key(self) -> None:
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"FACTOR_AI_ENV_FILE": ""}, clear=True):
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            result = service.run_mining(
                {"max_candidates": 5, "mining_mode": "hybrid", "ai_model": "test-model"}
            )
            self.assertEqual(result["mining_mode_requested"], "hybrid")
            self.assertEqual(result["mining_mode_effective"], "formula")
            self.assertFalse(result["ai_used"])
            self.assertIn("OPENAI_API_KEY", result["ai_fallback_reason"])
            self.assertTrue(all(row["source"] == "deterministic" for row in result["candidates"]))
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_hybrid_mining_uses_validated_ai_candidates_within_total_budget(self) -> None:
        ai_expression = call("neg", call("return", field("close"), window=4))
        batch = AIProposalBatch(
            model="test-model",
            prompt_version="test-v1",
            proposals=(AIProposal("短周期回撤反转", "reversal", ai_expression.to_dict()),),
            input_tokens=100,
            output_tokens=30,
        )
        with TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "secret", "OPENAI_FACTOR_MODEL": "test-model"},
            clear=True,
        ), patch("factor_lab.service.OpenAIProposalClient") as client_class:
            client_class.return_value.propose.return_value = batch
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            result = service.run_mining(
                {
                    "max_candidates": 10,
                    "ai_candidate_count": 2,
                    "mining_mode": "hybrid",
                }
            )

            self.assertEqual(result["candidate_count"], 10)
            self.assertTrue(result["ai_used"])
            self.assertEqual(result["mining_mode_effective"], "hybrid")
            self.assertEqual(result["ai_audit"]["valid"], 1)
            self.assertEqual(result["ai_audit"]["input_tokens"], 100)
            ai_row = next(
                row
                for row in result["candidates"]
                if row.get("source") == "ai" or row.get("also_proposed_by_ai")
            )
            self.assertEqual(ai_row["ai_hypothesis"], "短周期回撤反转")
            stored = service.catalog.get_run(result["run_id"])["config"]
            self.assertNotIn("secret", json.dumps(stored))
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_selected_candidate_can_be_promoted_to_formal_experiment(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            mining = service.run_mining({"max_candidates": 10, "forward_horizon": 2, "stability_windows": 2, "min_validation_abs_rank_ic": 0.0, "research_start": "2021-01-01", "research_end": "2025-12-31"})
            candidate = next(row for row in mining["candidates"] if row.get("screen_status") == "selected")
            promoted = service.run_promotion(
                {"run_id": mining["run_id"], "expression_id": candidate["expression_id"], "task_id": "promotion-test", "factor_role": "direction", "signal_semantics": "direct"}
            )

            self.assertEqual(promoted["expression_id"], candidate["expression_id"])
            self.assertEqual(promoted["formula"], candidate["formula"])
            self.assertEqual(promoted["forward_horizon"], 2)
            self.assertEqual(promoted["rebalance_bars"], 2)
            self.assertEqual(promoted["entry_quantile"], 0.8)
            self.assertIsNotNone(promoted["active_position_ratio"])
            self.assertIn(promoted["confirmation_status"], {"confirmed", "not_confirmed"})
            self.assertIn("holdout_start", promoted)
            self.assertIn("holdout_end", promoted)
            self.assertIn("holdout_aligned_rank_ic", promoted)
            self.assertIn("net_return", promoted)
            self.assertEqual(promoted["random_null_trials"], 100)
            self.assertGreater(promoted["random_null_abs_rank_ic_95"], 0)
            self.assertIsNotNone(promoted["buy_hold_net_return"])
            self.assertIsNotNone(promoted["excess_net_return_vs_buy_hold"])
            restored = service.catalog.get_candidate(mining["run_id"], candidate["expression_id"])
            self.assertEqual(restored["promotion_status"], "promoted")
            self.assertEqual(restored["promotion_task_id"], "promotion-test")
            self.assertEqual(restored["promotion_role"], "direction")
            self.assertEqual(restored["promotion_result"]["evaluation_mode"], "direction_backtest")
            self.assertEqual(service.mining_runs()[0]["promoted_count"], 1)
            self.assertEqual(service.mining_candidate(mining["run_id"], candidate["expression_id"])["formula"], candidate["formula"])
            self.assertFalse(
                service.mining_run(mining["run_id"])["final_confirmation_available"]
            )
            self.assertIn(
                "新增未见数据",
                service.mining_candidate(mining["run_id"], candidate["expression_id"])[
                    "final_confirmation_reason"
                ],
            )
            self.assertFalse(any(key.startswith("holdout_") for key in candidate))
            if promoted["confirmation_status"] == "confirmed":
                registered = service.register_candidate(
                    mining["run_id"],
                    candidate["expression_id"],
                    {"factor_id": "mined_test_factor"},
                )
                self.assertEqual(registered["factor_id"], "mined_test_factor")
                self.assertEqual(service.factor_library()[0]["signal_semantics"], "direct")
                self.assertIn("mined_test_factor", [factor["factor_id"] for factor in service.factors()])
            else:
                with self.assertRaisesRegex(ValueError, "final holdout confirmation"):
                    service.register_candidate(
                        mining["run_id"],
                        candidate["expression_id"],
                        {"factor_id": "mined_test_factor"},
                    )
            service._validate_factor_semantics("risk", "neutral")
            with self.assertRaisesRegex(ValueError, "neutral"):
                service._validate_factor_semantics("risk", "direct")
            with self.assertRaisesRegex(ValueError, "fdr_q"):
                service.run_mining({"max_candidates": 5, "fdr_q": 0})
            second = next(
                row
                for row in mining["candidates"]
                if row.get("screen_status") == "selected" and row["expression_id"] != candidate["expression_id"]
            )
            with self.assertRaisesRegex(ValueError, "新增未见数据"):
                service.submit_promotion(
                    mining["run_id"],
                    second["expression_id"],
                    {"factor_role": "direction", "signal_semantics": "direct"},
                )
            with self.assertRaisesRegex(ValueError, "specialized confirmation"):
                service.run_promotion(
                    {"run_id": mining["run_id"], "expression_id": second["expression_id"], "task_id": "risk-test", "factor_role": "risk", "signal_semantics": "neutral"}
                )
            with self.assertRaisesRegex(ValueError, "overlaps previously revealed"):
                service.run_promotion(
                    {"run_id": mining["run_id"], "expression_id": second["expression_id"], "task_id": "reuse-test", "factor_role": "direction", "signal_semantics": "direct"}
                )
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_legacy_mining_run_cannot_use_final_confirmation(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
                candidate_catalog_path=Path(directory) / "factor_catalog.sqlite3",
            )
            mining = service.run_mining(
                {"max_candidates": 5, "min_validation_abs_rank_ic": 0.0}
            )
            candidate = next(
                row for row in mining["candidates"] if row.get("screen_status") == "selected"
            )
            config = service.catalog.get_run(mining["run_id"])["config"]
            with closing(service.catalog._connect()) as connection:
                config["protocol_version"] = "2.2"
                connection.execute(
                    "UPDATE mining_runs SET config_json = ? WHERE run_id = ?",
                    (json.dumps(config), mining["run_id"]),
                )
                connection.commit()

            self.assertFalse(
                service.mining_run(mining["run_id"])["final_confirmation_available"]
            )

            with self.assertRaisesRegex(ValueError, "protocol 2.3"):
                service.run_promotion(
                    {
                        "run_id": mining["run_id"],
                        "expression_id": candidate["expression_id"],
                        "task_id": "legacy-test",
                        "factor_role": "direction",
                        "signal_semantics": "direct",
                    }
                )
            service.executor.shutdown(wait=True)
            service.mining_executor.shutdown(wait=True)

    def test_experiment_returns_combination_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            service = ResearchService(
                Path(__file__).parents[1],
                task_store_path=Path(directory) / "tasks.sqlite3",
            )
            result = service.run_experiment(
                {"momentum_window": 24, "forward_horizon": 4, "train_bars": 24 * 30, "test_bars": 24 * 7}
            )

            self.assertEqual(len(result["factors"]), 6)
            self.assertEqual(len(result["combination"]), 3)
            self.assertEqual(len(result["incremental_contribution"]), 2)
            self.assertEqual(result["combination"][0]["combination"], "equal_weight_all")
            service.executor.shutdown(wait=True)
