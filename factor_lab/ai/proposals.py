"""Structured AI proposals converted into the existing safe expression AST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from factor_lab.expression import Expression


@dataclass(frozen=True)
class AIProposal:
    hypothesis: str
    family: str
    expression: Mapping[str, object]
    round_number: int = 1


@dataclass(frozen=True)
class AIProposalBatch:
    model: str
    prompt_version: str
    proposals: tuple[AIProposal, ...]
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ValidatedProposal:
    expression: Expression
    hypothesis: str
    family: str
    round_number: int


@dataclass(frozen=True)
class ProposalAudit:
    proposed: int
    valid: int
    invalid: int
    duplicate: int
    errors: tuple[str, ...]


def validate_proposal_batch(
    batch: AIProposalBatch,
    *,
    allowed_fields: Sequence[str],
    allowed_windows: Sequence[int],
    max_depth: int,
    max_complexity: int,
    allowed_operators: Sequence[str] | None = None,
) -> tuple[tuple[ValidatedProposal, ...], ProposalAudit]:
    """Reject malformed, unsupported, over-complex, and duplicate AI proposals."""
    fields = set(allowed_fields)
    windows = set(allowed_windows)
    operators = set(allowed_operators) if allowed_operators is not None else None
    valid: list[ValidatedProposal] = []
    errors: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for index, proposal in enumerate(batch.proposals):
        try:
            if not proposal.hypothesis.strip() or not proposal.family.strip():
                raise ValueError("hypothesis and family are required")
            if proposal.round_number < 1:
                raise ValueError("round_number must be positive")
            expression = Expression.from_dict(proposal.expression).normalized()
            if not expression.fields.issubset(fields):
                raise ValueError(f"unsupported fields: {sorted(expression.fields.difference(fields))}")
            if any(window not in windows for window in expression.lookback_windows):
                raise ValueError("expression uses a window outside the configured set")
            if operators is not None and not expression.operators.issubset(operators):
                raise ValueError(
                    f"unsupported operators: {sorted(expression.operators.difference(operators))}"
                )
            if expression.depth > max_depth:
                raise ValueError("expression exceeds maximum depth")
            if expression.complexity > max_complexity:
                raise ValueError("expression exceeds maximum complexity")
            canonical = expression.canonical()
            if canonical in seen:
                duplicates += 1
                continue
            seen.add(canonical)
            valid.append(
                ValidatedProposal(
                    expression,
                    proposal.hypothesis.strip(),
                    proposal.family.strip(),
                    proposal.round_number,
                )
            )
        except (TypeError, ValueError) as error:
            errors.append(f"proposal {index}: {error}")
    return tuple(valid), ProposalAudit(
        proposed=len(batch.proposals),
        valid=len(valid),
        invalid=len(errors),
        duplicate=duplicates,
        errors=tuple(errors),
    )


def merge_candidate_sources(
    deterministic: Sequence[Expression],
    ai_proposals: Sequence[ValidatedProposal],
) -> tuple[tuple[Expression, ...], dict[str, dict[str, object]]]:
    """Merge canonical expressions while keeping deterministic and AI provenance."""
    expressions: list[Expression] = []
    metadata: dict[str, dict[str, object]] = {}
    for expression in deterministic:
        normalized = expression.normalized()
        if normalized.expression_id in metadata:
            continue
        expressions.append(normalized)
        metadata[normalized.expression_id] = {"source": "deterministic"}
    for proposal in ai_proposals:
        expression_id = proposal.expression.expression_id
        if expression_id in metadata:
            existing = metadata[expression_id]
            existing["also_proposed_by_ai"] = True
            existing["ai_hypothesis"] = proposal.hypothesis
            existing["ai_family"] = proposal.family
            existing["ai_round"] = proposal.round_number
            continue
        expressions.append(proposal.expression)
        metadata[expression_id] = {
            "source": "ai",
            "ai_hypothesis": proposal.hypothesis,
            "ai_family": proposal.family,
            "ai_round": proposal.round_number,
        }
    return tuple(expressions), metadata
