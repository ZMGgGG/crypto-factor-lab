from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_lab.benchmarks import random_rank_ic_benchmark


class RandomIcBenchmarkTest(unittest.TestCase):
    def test_random_ic_benchmark_is_deterministic(self) -> None:
        target = pd.Series(np.linspace(-0.1, 0.1, 200))
        first = random_rank_ic_benchmark(target, trials=50, seed=41)
        second = random_rank_ic_benchmark(target, trials=50, seed=41)

        self.assertEqual(first, second)
        self.assertEqual(first.trials, 50)
        self.assertGreater(first.absolute_rank_ic_95, 0)

    def test_random_ic_benchmark_rejects_small_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "twenty"):
            random_rank_ic_benchmark(pd.Series([1.0, 2.0, 3.0]), trials=10)
