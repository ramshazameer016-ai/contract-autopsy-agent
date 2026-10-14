"""Deterministic preference weighting and scoring engine for comparison.

Implements Architecture Section G.2:
- Six priorities: payment, ip, termination, scope, liability, confidentiality
- User weights: 0 (ignore), 1 (low), 2 (medium default), 3 (high)
- Favourability: fav_X(cat) = 100 - risk_score_X(cat) for categories with evidence
- Scored set: categories with clause evidence in BOTH contracts and weight > 0
- Score(X) = sum(w * fav_X) / sum(w_scored)
- Coverage = sum(w_scored) / sum(w_all)
- Dealbreaker guard: names RED findings in categories weighted 3
"""

from __future__ import annotations

from typing import Any
from config.settings import COVERAGE_THRESHOLD, MARGIN_THRESHOLD
from src.ca.schemas import (
    ContractAnalysis,
    PriorityWeights,
    VerdictStatus,
)

ALL_PRIORITY_CATEGORIES: list[str] = [
    "payment",
    "ip",
    "termination",
    "scope",
    "liability",
    "confidentiality",
]


def has_clause_evidence(analysis: ContractAnalysis, category: str) -> bool:
    """Return True if the contract contains actual clause evidence for this category."""
    has_clauses = any(c.category == category for c in analysis.clauses)
    findings = [f for f in analysis.findings if f.category == category]
    has_present_finding = any(f.evidence_status != "absent" for f in findings)
    return has_clauses and has_present_finding


def calculate_category_favourability(
    analysis: ContractAnalysis,
) -> dict[str, float | None]:
    """Calculate favourability (100 - risk_score) for each category with evidence."""
    fav_map: dict[str, float | None] = {}
    for cat in ALL_PRIORITY_CATEGORIES:
        if has_clause_evidence(analysis, cat):
            score = analysis.category_scores.get(cat, 0)
            fav_map[cat] = float(max(0, 100 - score))
        else:
            fav_map[cat] = None
    return fav_map


def calculate_weighted_scores(
    fav_a: dict[str, float | None],
    fav_b: dict[str, float | None],
    weights: dict[str, int],
) -> dict[str, Any]:
    """Calculate preference-weighted scores and coverage over the scored set.

    Formula from Architecture G.2:
    - Scored set: categories where both fav_a and fav_b are not None and weight > 0
    - Score(X) = sum(w * fav_X) / sum(w_scored)
    - coverage = sum(w_scored) / sum(w_all)
    """
    scored_categories: list[str] = []
    unscored_categories: list[str] = []

    sum_w_all = 0
    sum_w_scored = 0

    sum_weighted_a = 0.0
    sum_weighted_b = 0.0

    for cat in ALL_PRIORITY_CATEGORIES:
        w = weights.get(cat, 0)
        sum_w_all += w

        val_a = fav_a.get(cat)
        val_b = fav_b.get(cat)

        if w > 0 and val_a is not None and val_b is not None:
            scored_categories.append(cat)
            sum_w_scored += w
            sum_weighted_a += w * val_a
            sum_weighted_b += w * val_b
        else:
            unscored_categories.append(cat)

    # Coverage calculation
    coverage = round(sum_w_scored / sum_w_all, 3) if sum_w_all > 0 else 0.0

    # Scores calculation
    if sum_w_scored > 0:
        score_a = round(sum_weighted_a / sum_w_scored, 1)
        score_b = round(sum_weighted_b / sum_w_scored, 1)
    else:
        score_a = 0.0
        score_b = 0.0

    margin = round(score_a - score_b, 1)

    # Decision logic
    if coverage < COVERAGE_THRESHOLD:
        status: VerdictStatus = "insufficient_evidence"
    elif abs(margin) < MARGIN_THRESHOLD:
        status = "no_clear_winner"
    elif margin >= MARGIN_THRESHOLD:
        status = "leans_a"
    else:
        status = "leans_b"

    return {
        "score_a": score_a,
        "score_b": score_b,
        "margin": margin,
        "coverage": coverage,
        "status": status,
        "scored_categories": scored_categories,
        "unscored_categories": unscored_categories,
        "sum_w_scored": sum_w_scored,
        "sum_w_all": sum_w_all,
    }


def find_dealbreakers(
    analysis: ContractAnalysis,
    weights: dict[str, int],
    contract_label: str,
) -> list[str]:
    """Find RED findings in categories weighted 3 (dealbreakers)."""
    dealbreakers: list[str] = []
    for f in analysis.findings:
        if f.rating == "RED" and weights.get(f.category, 0) == 3:
            rule_ids = [tr.rule_id for tr in f.triggered_rules if tr.rule_id]
            rule_str = ", ".join(rule_ids) if rule_ids else "RED risk"
            reason = f.plain_reason or f.category.title()
            dealbreakers.append(
                f"[{contract_label} Dealbreaker] Category '{f.category.title()}' (High Priority) has a RED risk: {reason} ({rule_str})."
            )
    return dealbreakers
