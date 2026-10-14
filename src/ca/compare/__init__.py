"""Contract comparison package for Contract Autopsy Agent (Stage 5)."""

from __future__ import annotations

from src.ca.compare.matcher import match_contracts
from src.ca.compare.engine import (
    ALL_PRIORITY_CATEGORIES,
    has_clause_evidence,
    calculate_category_favourability,
    calculate_weighted_scores,
    find_dealbreakers,
)
from src.ca.compare.verdict import (
    build_deterministic_verdict,
    rewrite_verdict_with_ai,
    validate_ai_rewrite,
)
from src.ca.compare.borrow import generate_borrow_suggestions

__all__ = [
    "match_contracts",
    "ALL_PRIORITY_CATEGORIES",
    "has_clause_evidence",
    "calculate_category_favourability",
    "calculate_weighted_scores",
    "find_dealbreakers",
    "build_deterministic_verdict",
    "rewrite_verdict_with_ai",
    "validate_ai_rewrite",
    "generate_borrow_suggestions",
]
