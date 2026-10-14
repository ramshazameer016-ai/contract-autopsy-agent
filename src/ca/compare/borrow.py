"""Cross-contract borrow terms suggestions (Feature 7).

Implements Architecture Section G.4:
- Favourability gap check: fav(X) - fav(Y) >= 25 generates BorrowSuggestion from X to Y
- Missing clause check: Y lacks a category that X covers and weight > 0
- Exact required legal caveat:
  "This wording comes from another offer. It may not be legally transferable or acceptable to this party; use it only as a negotiation reference."
- Verbatim source quote and clause ID preservation
- Clean hand-off to counter-clause generator
"""

from __future__ import annotations

from config.settings import FAVOURABILITY_GAP_THRESHOLD
from src.ca.schemas import (
    BorrowSuggestion,
    ComparisonRow,
    ContractAnalysis,
    MANDATORY_BORROW_CAVEAT,
)
from src.ca.compare.engine import (
    ALL_PRIORITY_CATEGORIES,
    has_clause_evidence,
    calculate_category_favourability,
)


def generate_borrow_suggestions(
    analysis_a: ContractAnalysis,
    analysis_b: ContractAnalysis,
    rows: list[ComparisonRow],
    weights: dict[str, int],
    gap_threshold: int = FAVOURABILITY_GAP_THRESHOLD,
) -> list[BorrowSuggestion]:
    """Generate deterministic borrow suggestions from the stronger contract to the weaker one."""
    suggestions: list[BorrowSuggestion] = []
    fav_a = calculate_category_favourability(analysis_a)
    fav_b = calculate_category_favourability(analysis_b)

    seen_categories: set[str] = set()

    # 1. Favourability gap suggestions (where both contracts have evidence)
    for cat in ALL_PRIORITY_CATEGORIES:
        w = weights.get(cat, 0)
        if w == 0:
            continue

        val_fa = fav_a.get(cat)
        val_fb = fav_b.get(cat)

        if val_fa is not None and val_fb is not None:
            diff = val_fa - val_fb

            # Contract A is substantially more favourable than Contract B
            if diff >= gap_threshold:
                # Find strongest clause in A for this category with verbatim quote
                candidate_clause = None
                for c in analysis_a.clauses:
                    if c.category == cat and len(c.text.strip()) > 15:
                        candidate_clause = c
                        break

                if candidate_clause:
                    quote = candidate_clause.text.strip()
                    suggestions.append(
                        BorrowSuggestion(
                            source_contract="Contract A",
                            target_contract="Contract B",
                            category=cat,
                            clause_id=candidate_clause.clause_id,
                            page=candidate_clause.page_start,
                            verbatim_quote=quote,
                            reason=(
                                f"Contract A has a {diff:.0f}-point favourability advantage in {cat.title()}. "
                                "You can borrow this protective clause to propose better terms for Contract B."
                            ),
                            caveat=MANDATORY_BORROW_CAVEAT,
                            fav_diff=float(diff),
                            suggestion_type="favourability_gap",
                        )
                    )
                    seen_categories.add(cat)

            # Contract B is substantially more favourable than Contract A
            elif diff <= -gap_threshold:
                abs_diff = abs(diff)
                candidate_clause = None
                for c in analysis_b.clauses:
                    if c.category == cat and len(c.text.strip()) > 15:
                        candidate_clause = c
                        break

                if candidate_clause:
                    quote = candidate_clause.text.strip()
                    suggestions.append(
                        BorrowSuggestion(
                            source_contract="Contract B",
                            target_contract="Contract A",
                            category=cat,
                            clause_id=candidate_clause.clause_id,
                            page=candidate_clause.page_start,
                            verbatim_quote=quote,
                            reason=(
                                f"Contract B has a {abs_diff:.0f}-point favourability advantage in {cat.title()}. "
                                "You can borrow this protective clause to propose better terms for Contract A."
                            ),
                            caveat=MANDATORY_BORROW_CAVEAT,
                            fav_diff=float(abs_diff),
                            suggestion_type="favourability_gap",
                        )
                    )
                    seen_categories.add(cat)

    # 2. Missing clause suggestions (where one contract lacks category, other covers it, and w > 0)
    for cat in ALL_PRIORITY_CATEGORIES:
        if cat in seen_categories:
            continue

        w = weights.get(cat, 0)
        if w <= 0:
            continue

        has_a = has_clause_evidence(analysis_a, cat)
        has_b = has_clause_evidence(analysis_b, cat)

        # A has category, B completely lacks it
        if has_a and not has_b:
            candidate_clause = None
            for c in analysis_a.clauses:
                if c.category == cat and len(c.text.strip()) > 15:
                    candidate_clause = c
                    break

            if candidate_clause:
                quote = candidate_clause.text.strip()
                suggestions.append(
                    BorrowSuggestion(
                        source_contract="Contract A",
                        target_contract="Contract B",
                        category=cat,
                        clause_id=candidate_clause.clause_id,
                        page=candidate_clause.page_start,
                        verbatim_quote=quote,
                        reason=(
                            f"Contract B has no clause for {cat.title()}, leaving this priority unprotected. "
                            f"Borrow Contract A's clause as a starting point to insert this missing protection."
                        ),
                        caveat=MANDATORY_BORROW_CAVEAT,
                        fav_diff=None,
                        suggestion_type="missing_clause",
                    )
                )

        # B has category, A completely lacks it
        elif has_b and not has_a:
            candidate_clause = None
            for c in analysis_b.clauses:
                if c.category == cat and len(c.text.strip()) > 15:
                    candidate_clause = c
                    break

            if candidate_clause:
                quote = candidate_clause.text.strip()
                suggestions.append(
                    BorrowSuggestion(
                        source_contract="Contract B",
                        target_contract="Contract A",
                        category=cat,
                        clause_id=candidate_clause.clause_id,
                        page=candidate_clause.page_start,
                        verbatim_quote=quote,
                        reason=(
                            f"Contract A has no clause for {cat.title()}, leaving this priority unprotected. "
                            f"Borrow Contract B's clause as a starting point to insert this missing protection."
                        ),
                        caveat=MANDATORY_BORROW_CAVEAT,
                        fav_diff=None,
                        suggestion_type="missing_clause",
                    )
                )

    return suggestions
