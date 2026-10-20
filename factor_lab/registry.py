"""Extensible factor definitions shared by single- and multi-asset research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from factor_lab.factors import (
    funding_rate_zscore,
    price_momentum,
    realized_volatility,
    relative_volume,
    trend_efficiency,
)


@dataclass(frozen=True)
class FactorContext:
    """Data available at a research timestamp; future labels are excluded."""

    close: pd.Series
    volume: pd.Series
    funding_rate: pd.Series | None = None


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    factor_type: str
    description: str
    required_fields: tuple[str, ...]
    compute: Callable[[FactorContext], pd.Series]


class FactorRegistry:
    """Explicit registry preventing duplicate and undocumented factors."""

    def __init__(self) -> None:
        self._definitions: dict[str, FactorDefinition] = {}

    def register(self, definition: FactorDefinition) -> None:
        if definition.factor_id in self._definitions:
            raise ValueError(f"factor already registered: {definition.factor_id}")
        if not definition.factor_id or not definition.factor_type:
            raise ValueError("factor_id and factor_type are required")
        self._definitions[definition.factor_id] = definition

    def get(self, factor_id: str) -> FactorDefinition:
        try:
            return self._definitions[factor_id]
        except KeyError as error:
            raise KeyError(f"unknown factor: {factor_id}") from error

    def definitions(self) -> tuple[FactorDefinition, ...]:
        return tuple(self._definitions.values())

    def compute_all(self, context: FactorContext) -> dict[str, pd.Series]:
        factors: dict[str, pd.Series] = {}
        for definition in self.definitions():
            missing = [field for field in definition.required_fields if getattr(context, field, None) is None]
            if missing:
                raise ValueError(f"{definition.factor_id} requires missing fields: {missing}")
            factors[definition.factor_id] = definition.compute(context)
        return factors


def default_registry(window: int, include_funding: bool = False) -> FactorRegistry:
    """Build the shared starter registry for the current crypto data context."""
    registry = FactorRegistry()
    registry.register(
        FactorDefinition(
            "price_momentum",
            "time_series",
            "过去窗口的收盘价收益率。",
            ("close",),
            lambda context: price_momentum(context.close, window),
        )
    )
    registry.register(
        FactorDefinition(
            "price_reversal",
            "time_series",
            "过去窗口收盘价收益率的相反数。",
            ("close",),
            lambda context: price_momentum(context.close, window).mul(-1).rename(
                f"price_reversal_{window}"
            ),
        )
    )
    registry.register(
        FactorDefinition(
            "realized_volatility",
            "volatility",
            "过去窗口的对数收益率波动率。",
            ("close",),
            lambda context: realized_volatility(context.close, window),
        )
    )
    registry.register(
        FactorDefinition(
            "relative_volume",
            "liquidity",
            "当前成交量除以过去窗口平均成交量。",
            ("volume",),
            lambda context: relative_volume(context.volume, window),
        )
    )
    registry.register(
        FactorDefinition(
            "trend_efficiency",
            "trend",
            "价格净变化除以路径绝对变化。",
            ("close",),
            lambda context: trend_efficiency(context.close, window),
        )
    )
    if include_funding:
        registry.register(
            FactorDefinition(
                "funding_rate_zscore",
                "derivatives",
                "滞后一根 K 线后的资金费率滚动标准分。",
                ("funding_rate",),
                lambda context: funding_rate_zscore(
                    context.funding_rate, window, target_index=context.close.index
                ),
            )
        )
    return registry
