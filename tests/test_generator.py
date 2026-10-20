from __future__ import annotations

import unittest

from factor_lab.generator import GeneratorConfig, generate_expressions


class GeneratorTest(unittest.TestCase):
    def test_generation_is_deterministic_unique_and_bounded(self) -> None:
        config = GeneratorConfig(
            fields=("volume", "close", "close"),
            windows=(4, 2, 2),
            unary_operators=("abs",),
            window_operators=("return", "rolling_mean"),
            binary_operators=("add", "sub"),
            max_depth=3,
            max_complexity=5,
            max_candidates=80,
        )

        first = generate_expressions(config)
        second = generate_expressions(config)
        canonical = [expression.canonical() for expression in first]

        self.assertEqual(canonical, [expression.canonical() for expression in second])
        self.assertEqual(len(canonical), len(set(canonical)))
        self.assertLessEqual(len(first), config.max_candidates)
        self.assertTrue(all(expression.depth <= config.max_depth for expression in first))
        self.assertTrue(all(expression.complexity <= config.max_complexity for expression in first))
        self.assertTrue(all(expression.fields <= {"close", "volume"} for expression in first))
        self.assertTrue(any(expression.op in {"add", "sub", "mul", "div"} for expression in first))
        self.assertTrue(any(expression.depth >= 3 for expression in first))
        self.assertTrue(all(expression.operators.intersection(config.window_operators) for expression in first))

    def test_commutative_duplicates_and_trivial_ratios_are_not_generated(self) -> None:
        config = GeneratorConfig(
            fields=("close", "volume"),
            windows=(2,),
            unary_operators=(),
            window_operators=(),
            binary_operators=("add", "sub", "mul", "div"),
            max_depth=2,
            max_complexity=3,
            max_candidates=100,
        )
        expressions = generate_expressions(config)
        formulas = {expression.formula() for expression in expressions}

        self.assertNotIn("(close - close)", formulas)
        self.assertNotIn("(close / close)", formulas)
        self.assertFalse("(close + volume)" in formulas and "(volume + close)" in formulas)

    def test_configuration_rejects_unbounded_or_unknown_inputs(self) -> None:
        with self.assertRaises(ValueError):
            GeneratorConfig(max_candidates=0)
        with self.assertRaisesRegex(ValueError, "unsupported unary"):
            GeneratorConfig(unary_operators=("python",))
