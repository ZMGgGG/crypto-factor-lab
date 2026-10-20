from __future__ import annotations

import json
import unittest

import numpy as np

from scripts.serve_frontend import json_response


class JsonResponseTest(unittest.TestCase):
    def test_non_finite_and_numpy_values_are_json_safe(self) -> None:
        status, body = json_response(
            {"nan": float("nan"), "infinity": float("inf"), "number": np.float64(1.5)}
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body),
            {"nan": None, "infinity": None, "number": 1.5},
        )
