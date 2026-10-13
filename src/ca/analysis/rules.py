"""
Deterministic YAML-Driven Rule Engine for Contract Autopsy Agent.
Evaluates category rules, ambiguity phrases, and absence rules strictly according
to Section F of the architecture document.
"""

from __future__ import annotations

from pathlib import Path
import yaml
import re

from src.ca.schemas import (
    Clause,
    Fact,
    Rating,
    EvidenceSpan,
    EvidenceStatus,
    TriggeredRule,
    RiskFinding,
    Category,
)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "risk_rules.yaml"

_CACHED_RUBRIC: dict | None = None


def load_risk_rules(path: Path | str | None = None) -> dict:
    """Load rubric YAML configuration."""
    global _CACHED_RUBRIC
    rules_path = Path(path) if path else DEFAULT_RULES_PATH
    if not rules_path.exists():
        return {}
    with open(rules_path, "r", encoding="utf-8") as f:
        _CACHED_RUBRIC = yaml.safe_load(f)
    return _CACHED_RUBRIC


def _get_fact_value(facts: list[Fact], slot: str):
    """Retrieve value of first matching fact slot."""
    for f in facts:
        if f.slot == slot:
            return f.value
    return None


def _get_fact_evidence(facts: list[Fact], slot: str) -> EvidenceSpan | None:
    """Retrieve EvidenceSpan of first matching fact slot."""
    for f in facts:
        if f.slot == slot and f.evidence:
            return f.evidence
    return None


def _eval_condition(cond: dict, facts: list[Fact]) -> tuple[bool, list[str]]:
    """
    Evaluate a condition block against extracted facts.
    Returns (is_satisfied, matched_slots).
    """
    if "all" in cond:
        matched_all: list[str] = []
        for sub_cond in cond["all"]:
            sat, slots = _eval_condition(sub_cond, facts)
            if not sat:
                return False, []
            matched_all.extend(slots)
        return True, matched_all

    if "any" in cond:
        for sub_cond in cond["any"]:
            sat, slots = _eval_condition(sub_cond, facts)
            if sat:
                return True, slots
        return False, []

    slot = cond.get("slot")
    op = cond.get("op")
    target_val = cond.get("value")

    fact_val = _get_fact_value(facts, slot)

    if op == "exists":
        return fact_val is not None, [slot] if fact_val is not None else []
    if op == "missing":
        return fact_val is None, [slot]
    if op == "is_number":
        is_num = isinstance(fact_val, (int, float))
        return is_num, [slot] if is_num else []

    if fact_val is None:
        return False, []

    if op == "==":
        res = fact_val == target_val
        return res, [slot] if res else []
    if op == ">":
        res = float(fact_val) > float(target_val)
        return res, [slot] if res else []
    if op == ">=":
        res = float(fact_val) >= float(target_val)
        return res, [slot] if res else []
    if op == "<":
        res = float(fact_val) < float(target_val)
        return res, [slot] if res else []
    if op == "<=":
        res = float(fact_val) <= float(target_val)
        return res, [slot] if res else []
    if op == "between":
        if isinstance(target_val, list) and len(target_val) == 2:
            res = target_val[0] <= float(fact_val) <= target_val[1]
            return res, [slot] if res else []
    if op == "in":
        if isinstance(target_val, list):
            res = str(fact_val) in [str(v) for v in target_val]
            return res, [slot] if res else []

    return False, []


def evaluate_clause(
    clause: Clause,
    category: Category,
    rubric: dict | None = None,
) -> RiskFinding:
    """
    Evaluate a clause against YAML risk rubric for a given category.
    Returns an explainable RiskFinding.
    """
    cfg = rubric if rubric is not None else (_CACHED_RUBRIC or load_risk_rules())
    categories_cfg = cfg.get("categories", {})
    cat_rule_data = categories_cfg.get(category, {})

    rules_list = cat_rule_data.get("rules", [])
    required_slots = cat_rule_data.get("required_slots", [])
    base_score = cfg.get("base_score_present", 30)
    rubric_version = cfg.get("rubric_version", "0.1.0")

    triggered: list[TriggeredRule] = []
    collected_evidence: list[EvidenceSpan] = []
    plain_reasons: list[str] = []

    # 1. Evaluate explicit YAML category rules
    for r in rules_list:
        rule_id = r.get("id")
        severity = r.get("severity")
        points = r.get("points", 0)
        why_template = r.get("why", "")
        when_cond = r.get("when", {})

        is_satisfied, matched_slots = _eval_condition(when_cond, clause.facts)
        if is_satisfied:
            # Format reason template
            fmt_val = ""
            if matched_slots:
                first_val = _get_fact_value(clause.facts, matched_slots[0])
                fmt_val = str(first_val) if first_val is not None else ""
            why_text = why_template.replace("{value}", fmt_val).replace("{target}", "30")

            triggered.append(
                TriggeredRule(
                    rule_id=rule_id,
                    severity=severity,
                    points=points,
                    why=why_text,
                    fact_slots=matched_slots,
                )
            )
            plain_reasons.append(why_text)

            # Collect evidence spans
            for s in matched_slots:
                ev = _get_fact_evidence(clause.facts, s)
                if ev and ev not in collected_evidence:
                    collected_evidence.append(ev)

    # 2. Ambiguity Check
    ambiguity_phrases = cfg.get("ambiguity_phrases", [])
    text_lower = clause.text.lower()
    has_ambiguity = False
    for phrase in ambiguity_phrases:
        if phrase.lower() in text_lower:
            has_ambiguity = True
            # Find match offset in clause text
            match_pos = text_lower.find(phrase.lower())
            end_pos = match_pos + len(phrase)
            ev_amb = EvidenceSpan(
                doc_id=clause.doc_id,
                clause_id=clause.clause_id,
                page=clause.page_start,
                start=match_pos,
                end=end_pos,
                quote=clause.text[match_pos:end_pos],
            )
            if ev_amb not in collected_evidence:
                collected_evidence.append(ev_amb)
            plain_reasons.append(f"Ambiguous wording found: '{ev_amb.quote}'.")

    # 3. Determine Rating
    # Rating Precedence: RED > YELLOW > GREEN > UNRATED
    has_red = any(t.severity == "RED" for t in triggered)
    has_yellow = any(t.severity == "YELLOW" for t in triggered) or has_ambiguity
    has_green = any(t.severity == "GREEN" for t in triggered)

    # Required slots check
    all_required_present = all(_get_fact_value(clause.facts, s) is not None for s in required_slots)

    rating: Rating
    evidence_status: EvidenceStatus

    if has_red:
        rating = "RED"
        evidence_status = "present"
    elif has_yellow:
        rating = "YELLOW"
        evidence_status = "ambiguous" if has_ambiguity else "present"
    elif has_green:
        if all_required_present and not has_ambiguity:
            rating = "GREEN"
            evidence_status = "present"
        else:
            # Missing required slots prevents GREEN certification
            rating = "UNRATED"
            evidence_status = "insufficient"
            plain_reasons.append("Required facts are missing to confirm favorable terms.")
    else:
        rating = "UNRATED"
        evidence_status = "present" if clause.facts else "insufficient"
        plain_reasons.append("No specific risk or favourable rule fired.")

    # 4. Compute clamped Score
    points_delta = sum(t.points for t in triggered)
    if has_ambiguity and not any(t.severity == "YELLOW" for t in triggered):
        points_delta += 15  # Ambiguity adds 15 points
    raw_score = base_score + points_delta
    final_score = max(0, min(100, raw_score))

    # If no evidence was captured from facts, anchor on whole clause
    if not collected_evidence and clause.text:
        collected_evidence.append(
            EvidenceSpan(
                doc_id=clause.doc_id,
                clause_id=clause.clause_id,
                page=clause.page_start,
                start=0,
                end=len(clause.text),
                quote=clause.text,
            )
        )

    finding_id = f"F-{clause.doc_id}-{clause.clause_id}"

    return RiskFinding(
        finding_id=finding_id,
        doc_id=clause.doc_id,
        category=category,
        clause_id=clause.clause_id,
        rating=rating,
        evidence_status=evidence_status,
        score=final_score,
        triggered_rules=triggered,
        evidence=collected_evidence,
        plain_reason=" ".join(plain_reasons),
        kb_refs=[],
        rubric_version=rubric_version,
        conflicts_with=[],
    )


def evaluate_absence(
    category: Category,
    doc_id: str,
    rubric: dict | None = None,
) -> RiskFinding:
    """
    Generate an absent finding when a mandatory category has no clauses in the contract.
    Absent categories are NEVER GREEN (rated YELLOW with 50-60 risk points).
    """
    cfg = rubric if rubric is not None else (_CACHED_RUBRIC or load_risk_rules())
    absence_cfg = cfg.get("absence", {}).get(category, {})
    rubric_version = cfg.get("rubric_version", "0.1.0")

    rating = absence_cfg.get("rating", "YELLOW")
    points = absence_cfg.get("points", 55)
    why = absence_cfg.get("why", f"No {category} terms found in the contract.")

    rule = TriggeredRule(
        rule_id=f"absence.{category}",
        severity=rating,
        points=points,
        why=why,
        fact_slots=[],
    )

    return RiskFinding(
        finding_id=f"F-{doc_id}-ABSENT-{category}",
        doc_id=doc_id,
        category=category,
        clause_id=None,
        rating=rating,
        evidence_status="absent",
        score=points,
        triggered_rules=[rule],
        evidence=[],
        plain_reason=why,
        kb_refs=[],
        rubric_version=rubric_version,
        conflicts_with=[],
    )
