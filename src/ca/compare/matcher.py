"""Deterministic slot-based matching between Contract A and Contract B.

Implements Architecture Section G.1:
- Structured slot matching across categories
- Explicit gap representation (missing_a, missing_b, missing_both)
- Comparability checks (fixed fee vs hourly, differing currencies/units)
- Direction rules evaluation (never decided by AI)
- BM25/token similarity fallback for unslotted leftovers (marked uncertain, excluded from scoring)
"""

from __future__ import annotations

import re
from typing import Any
from config.settings import BM25_SIMILARITY_THRESHOLD
from src.ca.schemas import (
    Clause,
    ContractAnalysis,
    ComparisonRow,
    ComparisonStatus,
    Fact,
    FavoursContract,
)

# Canonical slot definitions with human labels, categories, and direction rules
SLOT_CONFIGS: list[dict[str, Any]] = [
    {
        "slot": "payment.due_days",
        "category": "payment",
        "label": "Payment Due Window",
        "unit": "days",
        "direction": "lower_is_better",
        "aliases": ["payment.due_days"],
    },
    {
        "slot": "payment.advance_pct",
        "category": "payment",
        "label": "Advance Payment Percentage",
        "unit": "%",
        "direction": "higher_is_better",
        "aliases": ["payment.advance_pct"],
    },
    {
        "slot": "payment.late_fee_stated",
        "category": "payment",
        "label": "Late Fee / Interest Specified",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["payment.late_fee_stated"],
    },
    {
        "slot": "payment.amount",
        "category": "payment",
        "label": "Fee Amount & Currency",
        "unit": "currency",
        "direction": "higher_is_better",
        "aliases": ["payment.amount", "payment.amount_stated"],
    },
    {
        "slot": "ip.assignment_scope",
        "category": "ip",
        "label": "IP Ownership Scope",
        "unit": "scope",
        "direction": "retained_over_assigned",
        "aliases": ["ip.assignment_scope"],
    },
    {
        "slot": "ip.preexisting_ip_carved_out",
        "category": "ip",
        "label": "Pre-existing IP Carve-out",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["ip.preexisting_ip_carved_out"],
    },
    {
        "slot": "ip.portfolio_right",
        "category": "ip",
        "label": "Portfolio / Showcase Right",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["ip.portfolio_right"],
    },
    {
        "slot": "ip.moral_rights_waived",
        "category": "ip",
        "label": "Moral Rights Waived",
        "unit": "bool",
        "direction": "false_is_better",
        "aliases": ["ip.moral_rights_waived", "ip.moral_rights_waiver"],
    },
    {
        "slot": "termination.client_notice_days",
        "category": "termination",
        "label": "Client Termination Notice",
        "unit": "days",
        "direction": "higher_is_better",
        "aliases": ["termination.client_notice_days", "termination.notice_days_client"],
    },
    {
        "slot": "termination.contractor_notice_days",
        "category": "termination",
        "label": "Contractor Termination Notice",
        "unit": "days",
        "direction": "higher_is_better",
        "aliases": ["termination.contractor_notice_days"],
    },
    {
        "slot": "termination.cure_period_days",
        "category": "termination",
        "label": "Breach Cure Period",
        "unit": "days",
        "direction": "higher_is_better",
        "aliases": ["termination.cure_period_days"],
    },
    {
        "slot": "termination.pay_for_work_done",
        "category": "termination",
        "label": "Payment for Work Completed on Termination",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["termination.pay_for_work_done"],
    },
    {
        "slot": "scope.deliverables_defined",
        "category": "scope",
        "label": "Deliverables & Specs Defined",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["scope.deliverables_defined"],
    },
    {
        "slot": "scope.change_control_required",
        "category": "scope",
        "label": "Written Change Control Process",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["scope.change_control_required", "scope.change_control"],
    },
    {
        "slot": "liability.consequential_damages_excluded",
        "category": "liability",
        "label": "Consequential Damages Excluded",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["liability.consequential_damages_excluded"],
    },
    {
        "slot": "liability.unlimited_contractor",
        "category": "liability",
        "label": "Unlimited Contractor Liability",
        "unit": "bool",
        "direction": "false_is_better",
        "aliases": ["liability.unlimited_contractor"],
    },
    {
        "slot": "confidentiality.mutual",
        "category": "confidentiality",
        "label": "Mutual Confidentiality",
        "unit": "bool",
        "direction": "true_is_better",
        "aliases": ["confidentiality.mutual"],
    },
]

# IP assignment order ranking (higher rank is better for freelancer)
IP_SCOPE_RANK: dict[str, int] = {
    "retained": 5,
    "license_only": 4,
    "deliverables_only": 3,
    "all_work": 1,
    "work_for_hire": 0,
}


def _extract_slot_facts_map(analysis: ContractAnalysis) -> dict[str, tuple[Fact, Clause]]:
    """Index facts by slot name (including aliases) from all clauses in an analysis."""
    slot_map: dict[str, tuple[Fact, Clause]] = {}
    for clause in analysis.clauses:
        for fact in clause.facts:
            slot_map[fact.slot] = (fact, clause)
    return slot_map


def _extract_payment_metadata(analysis: ContractAnalysis) -> dict[str, Any]:
    """Inspect payment clause text for fee basis, currency, and numerical amounts."""
    meta: dict[str, Any] = {
        "fee_basis": None,  # "fixed" or "hourly"
        "currency": None,   # "INR", "USD", etc.
        "amount": None,     # numeric amount
        "raw_text": "",
        "clause": None,
    }
    for clause in analysis.clauses:
        if clause.category == "payment":
            text = clause.text
            meta["raw_text"] = text
            meta["clause"] = clause

            # Basis detection
            if re.search(r"\b(?:hourly|per hour|hour)\b", text, re.IGNORECASE):
                meta["fee_basis"] = "hourly"
            elif re.search(r"\b(?:fixed fee|lump sum|total fee|fixed price)\b", text, re.IGNORECASE):
                meta["fee_basis"] = "fixed"

            # Currency detection
            if re.search(r"\b(?:INR|Rs\.?|Rupees)\b", text, re.IGNORECASE):
                meta["currency"] = "INR"
            elif re.search(r"\b(?:USD|\$|Dollars)\b", text, re.IGNORECASE):
                meta["currency"] = "USD"
            elif re.search(r"\b(?:EUR|€|Euros)\b", text, re.IGNORECASE):
                meta["currency"] = "EUR"

            # Numeric amount extraction
            amt_m = re.search(r"(?:INR|Rs\.?|USD|\$|EUR|€)\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
            if amt_m:
                try:
                    num_str = amt_m.group(1).replace(",", "")
                    meta["amount"] = float(num_str)
                except ValueError:
                    pass
            break
    return meta


def _format_display_value(val: Any, unit: str) -> str:
    """Format a slot value for human readability."""
    if val is None:
        return "Not specified"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if unit == "days":
        return f"{val} days"
    if unit == "%":
        return f"{val}%"
    if unit == "scope":
        return str(val).replace("_", " ").title()
    return str(val)


def _evaluate_direction(
    val_a: Any,
    val_b: Any,
    direction: str,
) -> FavoursContract:
    """Evaluate which contract is favoured by deterministic direction rules."""
    if val_a is None or val_b is None:
        return "none"

    if direction == "lower_is_better":
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if val_a < val_b:
                return "A"
            if val_b < val_a:
                return "B"
            return "tie"

    elif direction == "higher_is_better":
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if val_a > val_b:
                return "A"
            if val_b > val_a:
                return "B"
            return "tie"

    elif direction == "true_is_better":
        bool_a = bool(val_a)
        bool_b = bool(val_b)
        if bool_a and not bool_b:
            return "A"
        if bool_b and not bool_a:
            return "B"
        return "tie"

    elif direction == "false_is_better":
        bool_a = bool(val_a)
        bool_b = bool(val_b)
        if not bool_a and bool_b:
            return "A"
        if not bool_b and bool_a:
            return "B"
        return "tie"

    elif direction == "retained_over_assigned":
        str_a = str(val_a).lower()
        str_b = str(val_b).lower()
        rank_a = IP_SCOPE_RANK.get(str_a, 2)
        rank_b = IP_SCOPE_RANK.get(str_b, 2)
        if rank_a > rank_b:
            return "A"
        if rank_b > rank_a:
            return "B"
        return "tie"

    return "none"


def _compute_token_similarity(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity for unslotted clause comparison."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def match_contracts(
    analysis_a: ContractAnalysis,
    analysis_b: ContractAnalysis,
) -> list[ComparisonRow]:
    """Match Contract A and Contract B across structured slots and leftovers."""
    rows: list[ComparisonRow] = []

    facts_map_a = _extract_slot_facts_map(analysis_a)
    facts_map_b = _extract_slot_facts_map(analysis_b)

    pay_meta_a = _extract_payment_metadata(analysis_a)
    pay_meta_b = _extract_payment_metadata(analysis_b)

    matched_clause_ids_a: set[str] = set()
    matched_clause_ids_b: set[str] = set()

    for config in SLOT_CONFIGS:
        slot = config["slot"]
        cat = config["category"]
        label = config["label"]
        unit = config["unit"]
        direction = config["direction"]
        aliases = config["aliases"]

        # Resolve fact in A
        fact_clause_a = None
        for alias in aliases:
            if alias in facts_map_a:
                fact_clause_a = facts_map_a[alias]
                break

        # Resolve fact in B
        fact_clause_b = None
        for alias in aliases:
            if alias in facts_map_b:
                fact_clause_b = facts_map_b[alias]
                break

        # Special handling for payment.amount
        if slot == "payment.amount":
            amt_a = pay_meta_a.get("amount")
            amt_b = pay_meta_b.get("amount")
            curr_a = pay_meta_a.get("currency")
            curr_b = pay_meta_b.get("currency")
            basis_a = pay_meta_a.get("fee_basis")
            basis_b = pay_meta_b.get("fee_basis")
            cl_a = pay_meta_a.get("clause")
            cl_b = pay_meta_b.get("clause")

            has_a = amt_a is not None or basis_a is not None or fact_clause_a is not None
            has_b = amt_b is not None or basis_b is not None or fact_clause_b is not None

            if not has_a and not has_b:
                status = "missing_both"
                display_a = "Not found in Contract A"
                display_b = "Not found in Contract B"
                favours = "none"
                note = "Neither contract specifies fee amounts or basis."
                comparable = True
            elif has_a and not has_b:
                status = "missing_b"
                display_a = f"{curr_a or ''} {amt_a or ''} ({basis_a or 'stated'})".strip()
                display_b = "Not found in Contract B"
                favours = "none"
                note = "Contract B does not specify a fee amount."
                comparable = True
            elif not has_a and has_b:
                status = "missing_a"
                display_a = "Not found in Contract A"
                display_b = f"{curr_b or ''} {amt_b or ''} ({basis_b or 'stated'})".strip()
                favours = "none"
                note = "Contract A does not specify a fee amount."
                comparable = True
            else:
                # Both have payment data. Comparability check:
                is_diff_curr = curr_a and curr_b and curr_a != curr_b
                is_diff_basis = basis_a and basis_b and basis_a != basis_b

                display_a = f"{curr_a or ''} {amt_a or ''} ({basis_a or 'stated'})".strip()
                display_b = f"{curr_b or ''} {amt_b or ''} ({basis_b or 'stated'})".strip()

                if is_diff_curr or is_diff_basis:
                    status = "not_comparable"
                    comparable = False
                    favours = "none"
                    reasons = []
                    if is_diff_curr:
                        reasons.append(f"different currencies ({curr_a} vs {curr_b})")
                    if is_diff_basis:
                        reasons.append(f"incompatible basis ({basis_a} vs {basis_b})")
                    note = f"Cannot compare: {', '.join(reasons)}."
                else:
                    status = "matched"
                    comparable = True
                    note = ""
                    if amt_a is not None and amt_b is not None:
                        favours = "A" if amt_a > amt_b else ("B" if amt_b > amt_a else "tie")
                    else:
                        favours = "tie"

            quote_a = cl_a.text if cl_a else ""
            quote_b = cl_b.text if cl_b else ""
            cid_a = cl_a.clause_id if cl_a else None
            cid_b = cl_b.clause_id if cl_b else None
            p_a = cl_a.page_start if cl_a else None
            p_b = cl_b.page_start if cl_b else None

            if cl_a:
                matched_clause_ids_a.add(cl_a.clause_id)
            if cl_b:
                matched_clause_ids_b.add(cl_b.clause_id)

            rows.append(
                ComparisonRow(
                    category=cat,
                    slot=slot,
                    label=label,
                    val_a=amt_a or basis_a,
                    val_b=amt_b or basis_b,
                    display_a=display_a,
                    display_b=display_b,
                    status=status,
                    favours=favours,
                    comparable=comparable,
                    note=note,
                    quote_a=quote_a,
                    quote_b=quote_b,
                    clause_id_a=cid_a,
                    clause_id_b=cid_b,
                    page_a=p_a,
                    page_b=p_b,
                    uncertain_match=False,
                )
            )
            continue

        # Normal slot processing
        val_a = fact_clause_a[0].value if fact_clause_a else None
        val_b = fact_clause_b[0].value if fact_clause_b else None

        quote_a = fact_clause_a[0].evidence.quote if fact_clause_a else ""
        quote_b = fact_clause_b[0].evidence.quote if fact_clause_b else ""

        cid_a = fact_clause_a[1].clause_id if fact_clause_a else None
        cid_b = fact_clause_b[1].clause_id if fact_clause_b else None

        p_a = fact_clause_a[1].page_start if fact_clause_a else None
        p_b = fact_clause_b[1].page_start if fact_clause_b else None

        if fact_clause_a:
            matched_clause_ids_a.add(fact_clause_a[1].clause_id)
        if fact_clause_b:
            matched_clause_ids_b.add(fact_clause_b[1].clause_id)

        # Gap determination
        if val_a is None and val_b is None:
            status: ComparisonStatus = "missing_both"
            display_a = "Not found in Contract A"
            display_b = "Not found in Contract B"
            favours: FavoursContract = "none"
            note = f"Neither contract specifies {label.lower()}."
            comparable = True
        elif val_a is not None and val_b is None:
            status = "missing_b"
            display_a = _format_display_value(val_a, unit)
            display_b = "Not found in Contract B"
            favours = "none"
            note = f"Not found in Contract B."
            comparable = True
        elif val_a is None and val_b is not None:
            status = "missing_a"
            display_a = "Not found in Contract A"
            display_b = _format_display_value(val_b, unit)
            favours = "none"
            note = f"Not found in Contract A."
            comparable = True
        else:
            status = "matched"
            display_a = _format_display_value(val_a, unit)
            display_b = _format_display_value(val_b, unit)
            favours = _evaluate_direction(val_a, val_b, direction)
            note = ""
            comparable = True

        rows.append(
            ComparisonRow(
                category=cat,
                slot=slot,
                label=label,
                val_a=val_a,
                val_b=val_b,
                display_a=display_a,
                display_b=display_b,
                status=status,
                favours=favours,
                comparable=comparable,
                note=note,
                quote_a=quote_a,
                quote_b=quote_b,
                clause_id_a=cid_a,
                clause_id_b=cid_b,
                page_a=p_a,
                page_b=p_b,
                uncertain_match=False,
            )
        )

    # Check unslotted leftover clauses between A and B
    unmatched_a = [c for c in analysis_a.clauses if c.clause_id not in matched_clause_ids_a]
    unmatched_b = [c for c in analysis_b.clauses if c.clause_id not in matched_clause_ids_b]

    for ca in unmatched_a:
        for cb in unmatched_b:
            if ca.category == cb.category and ca.category != "other":
                sim = _compute_token_similarity(ca.text, cb.text)
                if sim >= BM25_SIMILARITY_THRESHOLD:
                    rows.append(
                        ComparisonRow(
                            category=ca.category,
                            slot=f"{ca.category}.unslotted_similarity",
                            label=f"Similar Clause: {ca.heading or cb.heading or ca.category.title()}",
                            val_a=ca.text[:60] + "...",
                            val_b=cb.text[:60] + "...",
                            display_a=ca.text[:60] + "...",
                            display_b=cb.text[:60] + "...",
                            status="uncertain",
                            favours="none",
                            comparable=False,
                            note=f"Uncertain similarity match ({sim:.2f}). Excluded from score.",
                            quote_a=ca.text,
                            quote_b=cb.text,
                            clause_id_a=ca.clause_id,
                            clause_id_b=cb.clause_id,
                            page_a=ca.page_start,
                            page_b=cb.page_start,
                            uncertain_match=True,
                        )
                    )

    return rows
