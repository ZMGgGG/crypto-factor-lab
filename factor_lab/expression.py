"""Safe, serializable expression trees for factor research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from typing import Mapping

import numpy as np
import pandas as pd


_COMMUTATIVE = {"add", "mul"}
_BINARY = {"add", "sub", "mul", "div"}
_UNARY = {"abs", "neg", "log", "sign"}
_WINDOWED = {"return", "log_return", "delta", "shift", "rolling_mean", "rolling_std", "rolling_sum", "rolling_min", "rolling_max", "zscore"}
_OPS = {"field", "constant", "range", *_BINARY, *_UNARY, *_WINDOWED}
_CHINESE_FIELDS = {
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "volume": "成交量",
    "funding_rate": "资金费率",
}
_CHINESE_OPERATORS = {
    "abs": "绝对值",
    "neg": "取负",
    "log": "对数",
    "sign": "符号",
    "return": "收益率",
    "log_return": "对数收益率",
    "delta": "差分",
    "shift": "滞后",
    "rolling_mean": "滚动均值",
    "rolling_std": "滚动标准差",
    "rolling_sum": "滚动求和",
    "rolling_min": "滚动最小值",
    "rolling_max": "滚动最大值",
    "zscore": "滚动标准分",
}


@dataclass(frozen=True)
class Expression:
    """An immutable factor expression; it never contains executable code."""

    op: str
    args: tuple["Expression", ...] = ()
    field: str | None = None
    window: int | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unsupported expression operation: {self.op}")
        if any(not isinstance(argument, Expression) for argument in self.args):
            raise TypeError("expression arguments must be Expression instances")
        if self.window is not None and (not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1):
            raise ValueError("window must be a positive integer")
        if self.op == "field":
            if not isinstance(self.field, str) or not self.field.strip() or self.args or self.window is not None or self.value is not None:
                raise ValueError("field expressions require only a non-empty field name")
        elif self.op == "constant":
            if self.args or self.field is not None or self.window is not None or self.value is None or not np.isfinite(self.value):
                raise ValueError("constant expressions require one finite numeric value")
        else:
            if self.field is not None or self.value is not None:
                raise ValueError(f"{self.op} cannot contain field or value metadata")
            self._validate_arity()

    def _validate_arity(self) -> None:
        expected = {"range": 2, **{operation: 2 for operation in _BINARY}}
        if self.op in expected and len(self.args) != expected[self.op]:
            raise ValueError(f"{self.op} requires {expected[self.op]} arguments")
        if self.op in _WINDOWED and len(self.args) != 1:
            raise ValueError(f"{self.op} requires one argument")
        if self.op in _UNARY and len(self.args) != 1:
            raise ValueError(f"{self.op} requires one argument")
        if self.op in _WINDOWED and self.window is None:
            raise ValueError(f"{self.op} requires a window")
        if (self.op == "range" or self.op in _UNARY) and self.window is not None:
            raise ValueError(f"{self.op} does not accept a window")

    @classmethod
    def field_ref(cls, name: str) -> "Expression":
        return cls("field", field=name)

    @classmethod
    def constant_value(cls, value: Real) -> "Expression":
        return cls("constant", value=float(value))

    @classmethod
    def call(cls, op: str, *args: "Expression", window: int | None = None) -> "Expression":
        return cls(op, tuple(args), window=window)

    @property
    def fields(self) -> frozenset[str]:
        if self.op == "field":
            return frozenset({self.field})
        fields: set[str] = set()
        for argument in self.args:
            fields.update(argument.fields)
        return frozenset(fields)

    @property
    def depth(self) -> int:
        return 1 if not self.args else 1 + max(argument.depth for argument in self.args)

    @property
    def complexity(self) -> int:
        return 1 + sum(argument.complexity for argument in self.args)

    @property
    def operators(self) -> frozenset[str]:
        operators = set() if self.op in {"field", "constant"} else {self.op}
        for argument in self.args:
            operators.update(argument.operators)
        return frozenset(operators)

    @property
    def lookback_windows(self) -> tuple[int, ...]:
        windows = [self.window] if self.window is not None else []
        for argument in self.args:
            windows.extend(argument.lookback_windows)
        return tuple(sorted(windows))

    def normalized(self) -> "Expression":
        args = tuple(argument.normalized() for argument in self.args)
        if self.op in _COMMUTATIVE:
            args = tuple(sorted(args, key=lambda argument: argument.canonical()))
        return Expression(self.op, args, field=self.field, window=self.window, value=self.value)

    def to_dict(self) -> dict[str, object]:
        expression = self.normalized()
        result: dict[str, object] = {"op": expression.op}
        if expression.args:
            result["args"] = [argument.to_dict() for argument in expression.args]
        if expression.field is not None:
            result["field"] = expression.field
        if expression.window is not None:
            result["window"] = expression.window
        if expression.value is not None:
            result["value"] = expression.value
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Expression":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("op"), str):
            raise ValueError("expression payload must contain a string op")
        unknown = set(payload).difference({"op", "args", "field", "window", "value"})
        if unknown:
            raise ValueError(f"unknown expression fields: {sorted(unknown)}")
        raw_args = payload.get("args", ())
        if not isinstance(raw_args, (list, tuple)):
            raise ValueError("expression args must be a list")
        return cls(
            payload["op"],
            tuple(cls.from_dict(argument) for argument in raw_args),
            field=payload.get("field"),
            window=payload.get("window"),
            value=payload.get("value"),
        )

    def canonical(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def expression_id(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()[:16]

    def formula(self) -> str:
        if self.op == "field":
            return self.field or ""
        if self.op == "constant":
            return format(self.value or 0.0, ".12g")
        if self.op in _BINARY or self.op == "range":
            symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/", "range": "-"}[self.op]
            return f"({self.args[0].formula()} {symbol} {self.args[1].formula()})"
        suffix = f", {self.window}" if self.window is not None else ""
        return f"{self.op}({self.args[0].formula()}{suffix})"

    def formula_zh(self) -> str:
        """Return a Chinese display label without changing the executable DSL."""
        if self.op == "field":
            return _CHINESE_FIELDS.get(self.field or "", self.field or "")
        if self.op == "constant":
            return format(self.value or 0.0, ".12g")
        if self.op in _BINARY or self.op == "range":
            symbol = {"add": "+", "sub": "-", "mul": "×", "div": "÷", "range": "-"}[self.op]
            return f"({self.args[0].formula_zh()} {symbol} {self.args[1].formula_zh()})"
        name = _CHINESE_OPERATORS[self.op]
        if self.window is not None:
            return f"{self.window}小时{name}({self.args[0].formula_zh()})"
        return f"{name}({self.args[0].formula_zh()})"


def _as_series(value: object, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"expression input {name!r} must be a pandas Series")
    return pd.to_numeric(value, errors="coerce")


def evaluate_expression(
    expression: Expression,
    data: pd.DataFrame | Mapping[str, pd.Series],
    group_level: str | None = None,
) -> pd.Series:
    """Evaluate an expression using only explicitly supplied, time-indexed fields."""
    expression = expression.normalized()
    context = {column: data[column] for column in data.columns} if isinstance(data, pd.DataFrame) else dict(data)
    if not context:
        raise ValueError("expression data cannot be empty")
    for field_name in expression.fields:
        if field_name not in context:
            raise KeyError(f"expression requires missing field: {field_name}")
    required = [_as_series(context[field_name], field_name) for field_name in sorted(expression.fields)]
    if required and any(not series.index.equals(required[0].index) for series in required[1:]):
        raise ValueError("expression inputs must use the same index")
    sample_index = next(iter(context.values())).index
    if group_level is not None and (
        not isinstance(sample_index, pd.MultiIndex) or group_level not in sample_index.names
    ):
        raise ValueError(f"group_level {group_level!r} is not present in the data index")

    def time_operation(series: pd.Series, operation) -> pd.Series:
        if group_level is None:
            return operation(series)
        return series.groupby(level=group_level, group_keys=False).transform(operation)

    def evaluate(node: Expression) -> pd.Series:
        if node.op == "field":
            return _as_series(context[node.field], node.field).copy()
        if node.op == "constant":
            index = next(iter(context.values())).index
            return pd.Series(node.value, index=index, dtype=float)
        values = [evaluate(argument) for argument in node.args]
        if node.op == "add":
            result = values[0] + values[1]
        elif node.op == "sub":
            result = values[0] - values[1]
        elif node.op == "mul":
            result = values[0] * values[1]
        elif node.op == "div":
            result = values[0].div(values[1].replace(0, np.nan))
        elif node.op == "range":
            result = values[0] - values[1]
        elif node.op == "abs":
            result = values[0].abs()
        elif node.op == "neg":
            result = values[0].mul(-1)
        elif node.op == "log":
            result = np.log(values[0].where(values[0] > 0))
        elif node.op == "sign":
            result = np.sign(values[0])
        elif node.op == "return":
            result = time_operation(
                values[0], lambda series: series.pct_change(node.window, fill_method=None)
            )
        elif node.op == "log_return":
            safe_log = np.log(values[0].where(values[0] > 0))
            result = time_operation(safe_log, lambda series: series.diff(node.window))
        elif node.op == "delta":
            result = time_operation(values[0], lambda series: series.diff(node.window))
        elif node.op == "shift":
            result = time_operation(values[0], lambda series: series.shift(node.window))
        elif node.op == "rolling_mean":
            result = time_operation(
                values[0], lambda series: series.rolling(node.window, min_periods=node.window).mean()
            )
        elif node.op == "rolling_std":
            result = time_operation(
                values[0], lambda series: series.rolling(node.window, min_periods=node.window).std()
            )
        elif node.op == "rolling_sum":
            result = time_operation(
                values[0], lambda series: series.rolling(node.window, min_periods=node.window).sum()
            )
        elif node.op == "rolling_min":
            result = time_operation(
                values[0], lambda series: series.rolling(node.window, min_periods=node.window).min()
            )
        elif node.op == "rolling_max":
            result = time_operation(
                values[0], lambda series: series.rolling(node.window, min_periods=node.window).max()
            )
        elif node.op == "zscore":
            def rolling_zscore(series: pd.Series) -> pd.Series:
                rolling = series.rolling(node.window, min_periods=node.window)
                return series.sub(rolling.mean()).div(rolling.std().replace(0, np.nan))

            result = time_operation(values[0], rolling_zscore)
        else:  # pragma: no cover - Expression validates operations before evaluation.
            raise ValueError(f"unsupported expression operation: {node.op}")
        return result.replace([np.inf, -np.inf], np.nan)

    result = evaluate(expression)
    result.name = expression.formula()
    return result


def field(name: str) -> Expression:
    return Expression.field_ref(name)


def constant(value: Real) -> Expression:
    return Expression.constant_value(value)


def call(op: str, *args: Expression, window: int | None = None) -> Expression:
    return Expression.call(op, *args, window=window)
