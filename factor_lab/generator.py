"""Deterministic generation of bounded factor-expression candidates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from factor_lab.expression import Expression, call, field


SUPPORTED_UNARY_OPERATORS = ("abs", "neg", "log", "sign")
SUPPORTED_WINDOW_OPERATORS = (
    "return",
    "log_return",
    "delta",
    "shift",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_min",
    "rolling_max",
    "zscore",
)
SUPPORTED_BINARY_OPERATORS = ("add", "sub", "mul", "div")
COMMUTATIVE_OPERATORS = {"add", "mul"}


@dataclass(frozen=True)
class GeneratorConfig:
    fields: tuple[str, ...] = ("close", "volume", "funding_rate")
    windows: tuple[int, ...] = (2, 4, 8, 12, 24, 48, 96)
    unary_operators: tuple[str, ...] = SUPPORTED_UNARY_OPERATORS
    window_operators: tuple[str, ...] = SUPPORTED_WINDOW_OPERATORS
    binary_operators: tuple[str, ...] = SUPPORTED_BINARY_OPERATORS
    max_depth: int = 3
    max_complexity: int = 6
    max_candidates: int = 5_000
    include_raw_fields: bool = False

    def __post_init__(self) -> None:
        if not self.fields or any(not isinstance(name, str) or not name.strip() for name in self.fields):
            raise ValueError("fields must contain non-empty names")
        if not self.windows or any(not isinstance(window, int) or isinstance(window, bool) or window < 1 for window in self.windows):
            raise ValueError("windows must contain positive integers")
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least two")
        if self.max_complexity < 2:
            raise ValueError("max_complexity must be at least two")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self._validate_operators("unary", self.unary_operators, SUPPORTED_UNARY_OPERATORS)
        self._validate_operators("window", self.window_operators, SUPPORTED_WINDOW_OPERATORS)
        self._validate_operators("binary", self.binary_operators, SUPPORTED_BINARY_OPERATORS)

    @staticmethod
    def _validate_operators(label: str, operators: tuple[str, ...], supported: tuple[str, ...]) -> None:
        unknown = set(operators).difference(supported)
        if unknown:
            raise ValueError(f"unsupported {label} operators: {sorted(unknown)}")


def generate_expressions(config: GeneratorConfig = GeneratorConfig()) -> tuple[Expression, ...]:
    """Generate a reproducible candidate set balanced across expression families."""
    roots = tuple(field(name) for name in sorted(set(config.fields)))
    seen: dict[str, Expression] = {expression.canonical(): expression for expression in roots}
    buckets: dict[str, list[Expression]] = {
        "raw": list(roots) if config.include_raw_fields else [],
        "unary": [],
        "window": [],
        "binary": [],
        "temporal_binary": [],
        "composed": [],
    }

    def add(bucket: str, expression: Expression) -> None:
        expression = expression.normalized()
        if expression.depth > config.max_depth or expression.complexity > config.max_complexity:
            return
        canonical = expression.canonical()
        if canonical in seen:
            return
        seen[canonical] = expression
        buckets[bucket].append(expression)

    windows = tuple(sorted(set(config.windows)))
    for expression in roots:
        for operator in config.unary_operators:
            add("unary", call(operator, expression))
        for operator, window in product(config.window_operators, windows):
            add("window", call(operator, expression, window=window))

    for left in roots:
        for right in roots:
            for operator in config.binary_operators:
                if left.canonical() == right.canonical() and operator in {"sub", "div"}:
                    continue
                if operator in COMMUTATIVE_OPERATORS and left.canonical() > right.canonical():
                    continue
                add("binary", call(operator, left, right))

    for expression in sorted(buckets["window"], key=lambda item: item.canonical()):
        if len(buckets["temporal_binary"]) >= config.max_candidates:
            break
        for root in roots:
            for operator in config.binary_operators:
                add("temporal_binary", call(operator, expression, root))
                if operator not in COMMUTATIVE_OPERATORS:
                    add("temporal_binary", call(operator, root, expression))
        if len(buckets["temporal_binary"]) >= config.max_candidates:
            buckets["temporal_binary"] = buckets["temporal_binary"][: config.max_candidates]

    depth_two = tuple(
        sorted(
            buckets["unary"] + buckets["window"] + buckets["binary"],
            key=lambda item: item.canonical(),
        )
    )
    for expression in depth_two:
        if len(buckets["composed"]) >= config.max_candidates:
            break
        for operator in config.unary_operators:
            add("composed", call(operator, expression))
        for operator, window in product(config.window_operators, windows):
            add("composed", call(operator, expression, window=window))
        for root in roots:
            for operator in config.binary_operators:
                add("composed", call(operator, expression, root))
                if operator not in COMMUTATIVE_OPERATORS:
                    add("composed", call(operator, root, expression))
        if len(buckets["composed"]) >= config.max_candidates:
            buckets["composed"] = buckets["composed"][: config.max_candidates]

    temporal_operators = set(config.window_operators)
    output_buckets = {
        "raw": buckets["raw"],
        "window": buckets["window"],
        "binary_temporal": buckets["temporal_binary"],
        "deep_window": [
            expression
            for expression in buckets["composed"]
            if expression.op in SUPPORTED_WINDOW_OPERATORS
            and expression.operators.intersection(temporal_operators)
        ],
        "deep_other": [
            expression
            for expression in buckets["composed"]
            if expression.op not in SUPPORTED_BINARY_OPERATORS
            and expression.op not in SUPPORTED_WINDOW_OPERATORS
            and expression.operators.intersection(temporal_operators)
        ],
    }
    family_order = ("raw", "window", "binary_temporal", "deep_window", "deep_other")
    offsets = {name: 0 for name in family_order}
    candidates: list[Expression] = []
    while len(candidates) < config.max_candidates:
        added = False
        for name in family_order:
            offset = offsets[name]
            if offset < len(output_buckets[name]):
                candidates.append(output_buckets[name][offset])
                offsets[name] += 1
                added = True
                if len(candidates) >= config.max_candidates:
                    break
        if not added:
            break
    return tuple(candidates)
