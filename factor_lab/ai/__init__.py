"""Auditable AI-assisted proposal boundaries for factor mining."""

from factor_lab.ai.proposals import (
    AIProposal,
    AIProposalBatch,
    ProposalAudit,
    merge_candidate_sources,
    validate_proposal_batch,
)
from factor_lab.ai.client import AIClientError, OpenAIProposalClient, openai_availability, openai_status, resolve_ai_config

__all__ = [
    "AIProposal",
    "AIProposalBatch",
    "ProposalAudit",
    "merge_candidate_sources",
    "validate_proposal_batch",
    "AIClientError",
    "OpenAIProposalClient",
    "openai_availability",
    "openai_status",
    "resolve_ai_config",
]
