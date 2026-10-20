"""Provider boundary for structured AI factor proposals."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.request import Request, urlopen

from factor_lab.ai.proposals import AIProposal, AIProposalBatch


class AIProposalClient(Protocol):
    def propose(self, request: dict[str, object]) -> AIProposalBatch: ...


class AIClientError(RuntimeError):
    """A safe, user-visible provider failure without request credentials."""


Transport = Callable[[str, dict[str, str], bytes, float], dict[str, object]]


_PROVIDER_ALIASES = {"dashscope": "qwen-cn"}
_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_PROVIDER_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "qwen-cn": "DASHSCOPE_CN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@dataclass(frozen=True)
class AIConfig:
    provider: str
    model: str | None
    base_url: str | None
    api_key_env: str | None
    api_key: str | None
    source: str


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without writing or logging secrets."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _ai_env_files() -> tuple[Path, ...]:
    explicit = os.environ.get("FACTOR_AI_ENV_FILE")
    if explicit is not None:
        return (Path(explicit),) if explicit else ()
    return (Path.cwd() / ".env",)


def resolve_ai_config(
    model: str | None = None,
    *,
    provider: str | None = None,
    env_files: tuple[Path, ...] | None = None,
) -> AIConfig:
    """Resolve process settings first, then this project's private dotenv."""
    file_values: dict[str, str] = {}
    files = env_files if env_files is not None else _ai_env_files()
    for path in files:
        file_values.update(_read_env_file(path))
    merged: Mapping[str, str] = {**file_values, **os.environ}
    selected_provider = str(
        provider or merged.get("FACTOR_AI_PROVIDER") or merged.get("TRADINGAGENTS_LLM_PROVIDER") or "openai"
    ).lower()
    selected_provider = _PROVIDER_ALIASES.get(selected_provider, selected_provider)
    key_env = _PROVIDER_KEY_ENVS.get(selected_provider)
    selected_model = model or merged.get("FACTOR_AI_MODEL")
    if not selected_model:
        selected_model = merged.get("TRADINGAGENTS_DEEP_THINK_LLM") or merged.get("OPENAI_FACTOR_MODEL")
    if selected_provider == "openai" and not model:
        selected_model = selected_model or "gpt-4o-mini"
    api_key = merged.get(key_env) if key_env else None
    base_url = merged.get("FACTOR_AI_BASE_URL") or _PROVIDER_BASE_URLS.get(selected_provider)
    source = "process environment"
    if not any(os.environ.get(key) for key in ("FACTOR_AI_PROVIDER", "FACTOR_AI_MODEL", "FACTOR_AI_BASE_URL", key_env or "")):
        for path in files:
            if path.exists():
                source = f"project env ({path.name})"
                break
    return AIConfig(selected_provider, selected_model, base_url, key_env, api_key, source)


def _default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, object]:
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise AIClientError("OpenAI returned a non-object response")
    return payload


def _expression_schema(request: Mapping[str, object]) -> dict[str, object]:
    fields = [str(value) for value in request.get("available_fields", [])]
    windows = [int(value) for value in request.get("allowed_windows", [])]
    operators = [
        str(value)
        for key in (
            "allowed_unary_operators",
            "allowed_window_operators",
            "allowed_binary_operators",
        )
        for value in request.get(key, [])
    ]
    allowed_ops = list(dict.fromkeys(["field", *operators]))
    return {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": allowed_ops,
            },
            "args": {"type": "array", "items": {"$ref": "#/$defs/expression"}},
            "field": {"anyOf": [{"type": "string", "enum": fields}, {"type": "null"}]},
            "window": {"anyOf": [{"type": "integer", "enum": windows}, {"type": "null"}]},
            "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "required": ["op", "args", "field", "window", "value"],
        "additionalProperties": False,
    }


class OpenAIProposalClient:
    """Single-round structured adapter for OpenAI and TradingAgents providers."""

    endpoint = "https://api.openai.com/v1/responses"
    prompt_version = "factor-proposal-v1"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        if not api_key or not model:
            raise ValueError("api_key and model are required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or _PROVIDER_BASE_URLS.get(provider, self.endpoint.rsplit("/", 1)[0])
        self.provider = _PROVIDER_ALIASES.get(provider.lower(), provider.lower())
        self.timeout = timeout
        self.transport = transport or _default_transport

    def propose(self, request: dict[str, object]) -> AIProposalBatch:
        proposal_count = int(request.get("proposal_count", 1))
        schema = {
            "type": "object",
            "properties": {
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {"type": "string"},
                            "family": {"type": "string"},
                            "expression": {"$ref": "#/$defs/expression"},
                        },
                        "required": ["hypothesis", "family", "expression"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["proposals"],
            "additionalProperties": False,
            "$defs": {"expression": _expression_schema(request)},
        }
        instructions = (
            "You propose auditable quantitative factor hypotheses, not trading advice. "
            "Return only expressions allowed by the supplied DSL. Use current or historical "
            "fields only. Prefer economically distinct, simple hypotheses. Do not infer or "
            "claim performance, and do not use future, validation, holdout, or observation data."
        )
        api_payload = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(request, ensure_ascii=False, sort_keys=True),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "factor_proposals",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self.provider != "openai":
            api_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False, sort_keys=True)},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "factor_proposals",
                        "description": "Return constrained factor proposals.",
                        "parameters": schema,
                    },
                }],
                "tool_choice": {"type": "function", "function": {"name": "factor_proposals"}},
            }
        try:
            response = self.transport(
                self.endpoint if self.provider == "openai" else f"{self.base_url.rstrip('/')}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json.dumps(api_payload, ensure_ascii=False).encode("utf-8"),
                self.timeout,
            )
            if self.provider == "openai" and response.get("status") not in (None, "completed"):
                raise AIClientError(f"OpenAI response status is {response.get('status')}")
            output_text = response.get("output_text")
            if self.provider != "openai":
                output_text = self._extract_chat_output(response)
            if not isinstance(output_text, str):
                output_text = self._extract_output_text(response)
            parsed = json.loads(output_text)
            raw_proposals = parsed.get("proposals") if isinstance(parsed, dict) else None
            if not isinstance(raw_proposals, list):
                raise AIClientError("OpenAI response does not contain proposals")
            raw_proposals = raw_proposals[:proposal_count]
            proposals = tuple(
                AIProposal(
                    hypothesis=str(item["hypothesis"]),
                    family=str(item["family"]),
                    expression=self._compact_expression(item["expression"]),
                )
                for item in raw_proposals
            )
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            return AIProposalBatch(
                model=self.model,
                prompt_version=self.prompt_version,
                proposals=proposals,
                input_tokens=self._optional_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
                output_tokens=self._optional_int(usage.get("output_tokens", usage.get("completion_tokens"))),
            )
        except AIClientError:
            raise
        except Exception as error:
            raise AIClientError(f"OpenAI proposal request failed: {type(error).__name__}") from error

    @staticmethod
    def _extract_chat_output(response: dict[str, object]) -> str | None:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AIClientError("provider response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise AIClientError("provider response has no message")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict):
            function = tool_calls[0].get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                return function["arguments"]
        content = message.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _extract_output_text(response: dict[str, object]) -> str:
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
                if isinstance(content, dict) and content.get("type") == "refusal":
                    raise AIClientError("OpenAI refused the factor proposal request")
        raise AIClientError("OpenAI response has no structured output text")

    @classmethod
    def _compact_expression(cls, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise AIClientError("OpenAI returned a malformed expression")
        operation = payload.get("op")
        result = {"op": operation}
        args = payload.get("args")
        if isinstance(args, list) and args:
            result["args"] = [cls._compact_expression(argument) for argument in args]
        elif operation not in {"field", "constant"} and payload.get("field") is not None:
            # Some OpenAI-compatible providers flatten a one-field call, e.g.
            # {"op":"zscore","field":"close","window":4}.
            result["args"] = [{"op": "field", "field": payload["field"]}]
        for key in ("field", "window", "value"):
            if payload.get(key) is not None and not (
                key == "field" and operation not in {"field", "constant"}
            ):
                result[key] = payload[key]
        return result

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) else None


@dataclass(frozen=True)
class AIAvailability:
    available: bool
    provider: str
    model: str | None
    reason: str | None


def openai_availability(
    model: str | None = None, provider: str | None = None
) -> AIAvailability:
    """Backward-compatible name for the shared TradingAgents-compatible resolver."""
    config = resolve_ai_config(model, provider=provider)
    if not config.api_key:
        return AIAvailability(False, config.provider, config.model, f"{config.api_key_env or 'API key'} is not configured")
    if not config.model:
        return AIAvailability(False, config.provider, None, "AI model is not configured")
    return AIAvailability(True, config.provider, config.model, None)


def openai_status(
    model: str | None = None, provider: str | None = None
) -> dict[str, object]:
    """Expose configuration state without exposing the API key."""
    config = resolve_ai_config(model, provider=provider)
    availability = openai_availability(model, provider=config.provider)
    return {
        "provider": availability.provider,
        "configured": availability.available,
        "model": availability.model,
        "message": f"已配置真实 {config.provider} API" if availability.available else "未配置真实 API，将回退公式模式",
        "reason": availability.reason,
        "base_url": config.base_url,
        "source": config.source,
    }
