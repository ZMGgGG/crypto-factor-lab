from __future__ import annotations

import os
import json
import unittest
from unittest.mock import patch

from factor_lab.ai.client import OpenAIProposalClient, openai_availability, resolve_ai_config
from factor_lab.ai.proposals import (
    AIProposal,
    AIProposalBatch,
    merge_candidate_sources,
    validate_proposal_batch,
)
from factor_lab.expression import call, field


class AIProposalTest(unittest.TestCase):
    def test_validates_and_deduplicates_structured_proposals(self) -> None:
        expression = call("return", field("close"), window=4).to_dict()
        batch = AIProposalBatch(
            model="test-model",
            prompt_version="v1",
            proposals=(
                AIProposal("短周期反转", "reversal", expression),
                AIProposal("重复提议", "reversal", expression),
                AIProposal("非法字段", "other", call("return", field("future"), window=4).to_dict()),
            ),
        )

        valid, audit = validate_proposal_batch(
            batch,
            allowed_fields=("close", "volume"),
            allowed_windows=(2, 4, 8),
            max_depth=3,
            max_complexity=6,
        )

        self.assertEqual(len(valid), 1)
        self.assertEqual(audit.proposed, 3)
        self.assertEqual(audit.duplicate, 1)
        self.assertEqual(audit.invalid, 1)

    def test_merges_ai_and_deterministic_sources_with_provenance(self) -> None:
        deterministic = (call("return", field("close"), window=4),)
        batch = AIProposalBatch(
            "test-model", "v1",
            (AIProposal("反转", "reversal", deterministic[0].to_dict()),),
        )
        valid, _ = validate_proposal_batch(
            batch, allowed_fields=("close",), allowed_windows=(4,),
            max_depth=3, max_complexity=6,
        )

        expressions, metadata = merge_candidate_sources(deterministic, valid)

        self.assertEqual(len(expressions), 1)
        self.assertEqual(metadata[expressions[0].expression_id]["source"], "deterministic")
        self.assertTrue(metadata[expressions[0].expression_id]["also_proposed_by_ai"])

    def test_missing_key_reports_explicit_formula_fallback(self) -> None:
        with patch.dict(os.environ, {"FACTOR_AI_ENV_FILE": ""}, clear=True):
            availability = openai_availability("test-model")
        self.assertFalse(availability.available)
        self.assertIn("OPENAI_API_KEY", availability.reason)

    def test_resolves_project_qwen_cn_env_without_exposing_key(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_ai_config(
                env_files=(Path("/tmp/tradingagents-test.env"),),
            )
        self.assertEqual(config.provider, "openai")

        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "FACTOR_AI_PROVIDER=qwen-cn\n"
                "FACTOR_AI_MODEL=qwen3.6-plus\n"
                "DASHSCOPE_CN_API_KEY=test-secret\n",
                encoding="utf-8",
            )
            config = resolve_ai_config(env_files=(env_path,))
        self.assertEqual(config.provider, "qwen-cn")
        self.assertEqual(config.model, "qwen3.6-plus")
        self.assertEqual(config.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertTrue(config.api_key)

    def test_openai_client_requests_strict_schema_and_parses_ast(self) -> None:
        captured = {}

        def transport(url, headers, body, timeout):
            captured.update(
                url=url,
                headers=headers,
                payload=json.loads(body),
                timeout=timeout,
            )
            return {
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "proposals": [
                                            {
                                                "hypothesis": "短周期价格回撤可能反转",
                                                "family": "reversal",
                                                "expression": {
                                                    "op": "neg",
                                                    "args": [
                                                        {
                                                            "op": "return",
                                                            "args": [
                                                                {
                                                                    "op": "field",
                                                                    "args": [],
                                                                    "field": "close",
                                                                    "window": None,
                                                                    "value": None,
                                                                }
                                                            ],
                                                            "field": None,
                                                            "window": 4,
                                                            "value": None,
                                                        }
                                                    ],
                                                    "field": None,
                                                    "window": None,
                                                    "value": None,
                                                },
                                            }
                                        ]
                                    }
                                ),
                            }
                        ]
                    }
                ],
                "usage": {"input_tokens": 123, "output_tokens": 45},
            }

        batch = OpenAIProposalClient(
            "secret", "test-model", transport=transport
        ).propose({
            "proposal_count": 2,
            "available_fields": ["close"],
            "allowed_windows": [4],
            "allowed_unary_operators": ["neg"],
            "allowed_window_operators": ["return"],
            "allowed_binary_operators": [],
        })

        self.assertEqual(batch.input_tokens, 123)
        self.assertEqual(batch.output_tokens, 45)
        self.assertEqual(batch.proposals[0].expression["op"], "neg")
        self.assertNotIn("secret", json.dumps(captured["payload"]))
        output_format = captured["payload"]["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        expression_schema = output_format["schema"]["$defs"]["expression"]
        self.assertEqual(expression_schema["properties"]["field"]["anyOf"][0]["enum"], ["close"])
        self.assertEqual(expression_schema["properties"]["window"]["anyOf"][0]["enum"], [4])

    def test_compacts_flattened_compatible_provider_field_call(self) -> None:
        expression = OpenAIProposalClient._compact_expression(
            {"op": "zscore", "field": "close", "window": 4}
        )
        self.assertEqual(expression, {
            "op": "zscore",
            "args": [{"op": "field", "field": "close"}],
            "window": 4,
        })
