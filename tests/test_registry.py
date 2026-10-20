from __future__ import annotations

import unittest

import pandas as pd

from factor_lab.registry import FactorContext, FactorDefinition, FactorRegistry, default_registry


class FactorRegistryTest(unittest.TestCase):
    def test_default_registry_has_stable_factor_metadata(self) -> None:
        registry = default_registry(24, include_funding=False)

        self.assertEqual([definition.factor_id for definition in registry.definitions()], [
            "price_momentum",
            "price_reversal",
            "realized_volatility",
            "relative_volume",
            "trend_efficiency",
        ])
        self.assertEqual(registry.get("realized_volatility").factor_type, "volatility")

    def test_registry_rejects_duplicates_and_missing_inputs(self) -> None:
        registry = FactorRegistry()
        definition = FactorDefinition("custom", "custom", "test", ("close",), lambda context: context.close)
        registry.register(definition)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(definition)

        context = FactorContext(pd.Series([1.0]), pd.Series([1.0]))
        with self.assertRaisesRegex(ValueError, "missing fields"):
            default_registry(2, include_funding=True).compute_all(context)
