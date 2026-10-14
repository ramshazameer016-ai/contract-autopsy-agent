"""Deterministic verdict generation and optional validated AI rewrite.

Implements Architecture Section G.3:
- Deterministic structured verdict with headline, category differences, trade-offs,
  dealbreakers, missing/uncertain evidence, and coverage/margin details.
- Optional validated AI rewrite for readability:
  - Bounded to readability only
  - Strictly preserves leaning
  - Must not add unsupported numbers or contract terms
  - Safe fallback to deterministic template on any failure
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from config.settings import is_offline, is_ai_available
from src.ca.schemas import (
    ComparisonVerdict,
    ComparisonRow,
    ContractAnalysis,
    VerdictStatus,
)
from src.ca.compare.engine import (
    ALL_PRIORITY_CATEGORIES,
    find_dealbreakers,
    calculate_weighted_scores,
    calculate_category_favourability,
)


class VerdictRewriteSchema(BaseModel):
    """Pydantic schema for AI verdict rewrite."""
    status: VerdictStatus
    headline: str = Field(min_length=5, max_length=200)
    rewritten_text: str = Field(min_length=20, max_length=2000)


def build_deterministic_verdict(
    scores_data: dict[str, Any],
    fav_a: dict[str, float | None],
    fav_b: dict[str, float | None],
    rows: list[ComparisonRow],
    dealbreakers: list[str],
    weights: dict[str, int],
) -> ComparisonVerdict:
    """Construct structured, deterministic verdict strictly from evidence."""
    status: VerdictStatus = scores_data["status"]
    score_a = scores_data["score_a"]
    score_b = scores_data["score_b"]
    margin = scores_data["margin"]
    coverage = scores_data["coverage"]
    scored_cats = scores_data["scored_categories"]
    unscored_cats = scores_data["unscored_categories"]

    # 1. Headline
    if status == "insufficient_evidence":
        headline = "Insufficient evidence to determine a winner on your priorities."
    elif status == "no_clear_winner":
        headline = f"No clear winner on your priorities (margin {abs(margin):.1f} pts is below threshold 8.0)."
    elif status == "leans_a":
        headline = f"Leans toward Contract A (Score {score_a:.1f} vs {score_b:.1f}, margin +{margin:.1f})."
    else:
        headline = f"Leans toward Contract B (Score {score_b:.1f} vs {score_a:.1f}, margin {margin:.1f})."

    # 2. Category differences
    category_summaries: list[str] = []
    leads_a: list[str] = []
    leads_b: list[str] = []

    for cat in scored_cats:
        fa = fav_a.get(cat, 0.0) or 0.0
        fb = fav_b.get(cat, 0.0) or 0.0
        diff = round(fa - fb, 1)

        # Find key slots for this category
        cat_rows = [r for r in rows if r.category == cat and r.status == "matched"]
        slot_details = []
        for r in cat_rows[:2]:
            slot_details.append(f"{r.label}: A has '{r.display_a}', B has '{r.display_b}'")
        detail_str = f" ({'; '.join(slot_details)})" if slot_details else ""

        if diff > 5.0:
            leads_a.append(cat.title())
            category_summaries.append(
                f"{cat.title()}: Contract A is more favourable ({fa:.0f} vs {fb:.0f} favourability){detail_str}."
            )
        elif diff < -5.0:
            leads_b.append(cat.title())
            category_summaries.append(
                f"{cat.title()}: Contract B is more favourable ({fb:.0f} vs {fa:.0f} favourability){detail_str}."
            )
        else:
            category_summaries.append(
                f"{cat.title()}: Both contracts offer comparable terms ({fa:.0f} vs {fb:.0f} favourability){detail_str}."
            )

    # 3. Trade-offs
    trade_offs: list[str] = []
    if leads_a and leads_b:
        trade_offs.append(
            f"Key Trade-off: Contract A is stronger in {', '.join(leads_a)}, whereas Contract B is stronger in {', '.join(leads_b)}."
        )

    # 4. Missing or uncertain evidence
    missing_evidence: list[str] = []
    for cat in unscored_cats:
        w = weights.get(cat, 0)
        reason = "assigned priority 0 (ignored)" if w == 0 else "missing clause evidence in one or both contracts"
        missing_evidence.append(f"{cat.title()} was not scored because it was {reason}.")

    uncertain_rows = [r for r in rows if r.status == "uncertain"]
    if uncertain_rows:
        missing_evidence.append(
            f"{len(uncertain_rows)} unslotted clauses were matched with uncertain similarity and excluded from scoring."
        )

    # Compose full deterministic text
    sections = [
        f"**Verdict:** {headline}",
        f"**Weighted Scores:** Contract A: {score_a:.1f} | Contract B: {score_b:.1f} | Margin: {margin:+.1f} | Priority Coverage: {int(coverage * 100)}%",
    ]
    if dealbreakers:
        sections.append("**Dealbreaker Warnings:**\n" + "\n".join(f"- ⚠️ {d}" for d in dealbreakers))
    if category_summaries:
        sections.append("**Category Analysis:**\n" + "\n".join(f"- {s}" for s in category_summaries))
    if trade_offs:
        sections.append("**Trade-offs:**\n" + "\n".join(f"- ⚖️ {t}" for t in trade_offs))
    if missing_evidence:
        sections.append("**Missing / Not Scored Evidence:**\n" + "\n".join(f"- ℹ️ {m}" for m in missing_evidence))

    full_text = "\n\n".join(sections)

    return ComparisonVerdict(
        status=status,
        headline=headline,
        score_a=score_a,
        score_b=score_b,
        margin=margin,
        coverage=coverage,
        scored_categories=scored_cats,
        unscored_categories=unscored_cats,
        category_summaries=category_summaries,
        dealbreakers=dealbreakers,
        trade_offs=trade_offs,
        missing_evidence=missing_evidence,
        generated_by="deterministic",
        full_text=full_text,
    )


def _extract_numbers(text: str) -> set[float]:
    """Extract numeric values from text for validation."""
    found = set()
    for token in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        try:
            found.add(float(token))
        except ValueError:
            pass
    return found


def validate_ai_rewrite(
    candidate: VerdictRewriteSchema,
    expected_status: VerdictStatus,
    allowed_numbers: set[float],
) -> bool:
    """Validate AI rewrite against strict architecture rules.

    Rejects rewrite if:
    1. Leaning/status differs from deterministic verdict
    2. Candidate contains numbers not present in the allowed set
    """
    if candidate.status != expected_status:
        return False

    candidate_numbers = _extract_numbers(candidate.rewritten_text)
    # Check if all candidate numbers exist in allowed_numbers
    for num in candidate_numbers:
        # Allow if exact match or close integer/float representation
        if not any(abs(num - allowed) < 0.05 for allowed in allowed_numbers):
            return False

    return True


def rewrite_verdict_with_ai(
    verdict: ComparisonVerdict,
    has_consent: bool = False,
) -> ComparisonVerdict:
    """Optionally rewrite deterministic verdict for readability with strict validation.

    Never changes status/leaning or injects new numbers.
    Falls back to template verdict on any failure.
    """
    if not has_consent or is_offline():
        return verdict

    try:
        from src.ca.llm.client import generate_validated

        prompt = (
            "You are a contract intelligence assistant rewriting a comparison verdict for a freelancer.\n"
            "Rewrite the following deterministic verdict for plain-language clarity.\n"
            "RULES:\n"
            "1. You MUST keep the exact status value: " + verdict.status + "\n"
            "2. DO NOT change which contract is favoured or invent any new facts.\n"
            "3. DO NOT introduce ANY numbers that are not explicitly present in the data below.\n"
            "4. Return valid JSON matching the schema.\n\n"
            "<contract_data>\n"
            f"Status: {verdict.status}\n"
            f"Headline: {verdict.headline}\n"
            f"Scores: A={verdict.score_a}, B={verdict.score_b}, margin={verdict.margin}, coverage={int(verdict.coverage * 100)}%\n"
            f"Dealbreakers: {'; '.join(verdict.dealbreakers)}\n"
            f"Category Breakdown: {'; '.join(verdict.category_summaries)}\n"
            f"Trade-offs: {'; '.join(verdict.trade_offs)}\n"
            "</contract_data>\n"
        )

        rewritten = generate_validated(
            prompt=prompt,
            schema=VerdictRewriteSchema,
            prompt_version="verdict_v1",
        )

        # Allowed numbers
        allowed = {
            verdict.score_a,
            verdict.score_b,
            abs(verdict.margin),
            verdict.margin,
            float(int(verdict.coverage * 100)),
            verdict.coverage,
            8.0,  # threshold
            50.0, # coverage threshold
        }
        for s in verdict.category_summaries:
            allowed.update(_extract_numbers(s))

        if validate_ai_rewrite(rewritten, verdict.status, allowed):
            verdict_dict = verdict.model_dump()
            verdict_dict["headline"] = rewritten.headline
            verdict_dict["full_text"] = rewritten.rewritten_text
            verdict_dict["generated_by"] = "llm_rewritten"
            return ComparisonVerdict(**verdict_dict)

    except Exception:
        pass

    return verdict
