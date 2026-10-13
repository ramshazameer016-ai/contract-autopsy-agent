"""
Stage 2 Comprehensive Acceptance Tests for Contract Autopsy Agent.
Validates fact extraction, deterministic rule engine, YAML rubric,
golden tests G1-G5 & G8, absence handling, and consistency checking.
"""

import json
from pathlib import Path
import pytest

from src.ca.schemas import (
    Clause,
    ExtractedDocument,
    ContractAnalysis,
    RiskFinding,
)
from src.ca.extraction.loader import load_document
from src.ca.analysis.facts import extract_facts_from_clause, extract_facts_for_clauses
from src.ca.analysis.rules import (
    evaluate_clause,
    evaluate_absence,
    load_risk_rules,
)
from src.ca.analysis.consistency import check_consistency
from src.ca.analysis.pipeline import analyse_contract

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_contracts"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "clauses.json"


def test_1_fact_extraction_synthetic():
    """Requirement 1: Extract structured facts with exact EvidenceSpan anchors."""
    clause_text = "The Client shall pay each invoice within 15 days of receipt. Advance payment of 30% is required."
    clause = Clause(
        clause_id="A-001",
        doc_id="A",
        heading="Payment",
        text=clause_text,
        char_start=0,
        char_end=len(clause_text),
        category="payment",
    )
    facts = extract_facts_from_clause(clause)
    slots = {f.slot: f.value for f in facts}

    assert slots.get("payment.due_days") == 15
    assert slots.get("payment.advance_pct") == 30.0

    # Verify evidence span integrity
    for f in facts:
        assert f.evidence is not None
        assert clause.text[f.evidence.start:f.evidence.end] == f.evidence.quote


def test_2_missing_and_unclear_facts():
    """Requirement 2: Distinguish explicit facts from missing or unclear information."""
    vague_text = "The Client shall pay a reasonable fee within a reasonable time."
    clause = Clause(
        clause_id="A-002",
        doc_id="A",
        heading="Payment",
        text=vague_text,
        char_start=0,
        char_end=len(vague_text),
        category="payment",
    )
    facts = extract_facts_from_clause(clause)
    due_day_facts = [f for f in facts if f.slot == "payment.due_days"]
    # Due days cannot be guessed or fabricated
    assert len(due_day_facts) == 0

    finding = evaluate_clause(clause, "payment")
    assert finding.rating == "YELLOW"
    assert finding.evidence_status == "ambiguous"
    assert any("reasonable time" in ev.quote for ev in finding.evidence)


def test_3_deterministic_rule_evaluation():
    """Requirement 3: Deterministic rule evaluation yielding repeatable ratings."""
    text_red = "Payment is due within 90 days of invoice. Payment is only after the client has been paid by customer."
    clause = Clause(
        clause_id="A-003",
        doc_id="A",
        heading="Payment",
        text=text_red,
        char_start=0,
        char_end=len(text_red),
        category="payment",
    )
    extract_facts_from_clause(clause)
    finding = evaluate_clause(clause, "payment")

    assert finding.rating == "RED"
    assert finding.score >= 70
    rule_ids = [r.rule_id for r in finding.triggered_rules]
    assert "PAY-R1" in rule_ids
    assert "PAY-R2" in rule_ids


def test_4_stable_rule_ids_and_configured_severity():
    """Requirement 4: Stable rule IDs and configured severity levels."""
    rubric = load_risk_rules()
    cat_rules = rubric["categories"]["payment"]["rules"]
    rule_map = {r["id"]: r["severity"] for r in cat_rules}

    assert rule_map.get("PAY-R1") == "RED"
    assert rule_map.get("PAY-R2") == "RED"
    assert rule_map.get("PAY-Y1") == "YELLOW"
    assert rule_map.get("PAY-Y2") == "YELLOW"
    assert rule_map.get("PAY-Y3") == "YELLOW"
    assert rule_map.get("PAY-G1") == "GREEN"
    assert rule_map.get("PAY-G2") == "GREEN"


def test_5_evidence_quotes_match_source_clause():
    """Requirement 5: Evidence quotes must strictly equal clause text substring."""
    doc = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    analysis = analyse_contract(doc)

    clause_map = {c.clause_id: c for c in analysis.clauses}
    for finding in analysis.findings:
        for ev in finding.evidence:
            if ev.clause_id in clause_map:
                source_clause = clause_map[ev.clause_id]
                assert ev.quote in source_clause.text
                assert source_clause.text[ev.start:ev.end] == ev.quote


def test_6_missing_category_handling_not_green():
    """Requirement 6: Missing category handling as 'Not found', never GREEN."""
    finding = evaluate_absence("scope", "A")
    assert finding.rating == "YELLOW"
    assert finding.evidence_status == "absent"
    assert finding.clause_id is None
    assert finding.score >= 50
    assert "No scope or deliverables clause found" in finding.plain_reason
    assert finding.rating != "GREEN"


def test_7_consistency_conflicting_value_detection():
    """Requirement 7: Golden G8 conflicting terms across clauses."""
    c1 = Clause(
        clause_id="A-004",
        doc_id="A",
        heading="Payment Schedule",
        text="The Client shall pay each invoice within 15 days of receipt.",
        char_start=0,
        char_end=58,
        category="payment",
    )
    c2 = Clause(
        clause_id="A-011",
        doc_id="A",
        heading="Late Billing",
        text="All payments are due within 60 days of invoice.",
        char_start=60,
        char_end=107,
        category="payment",
    )
    extract_facts_from_clause(c1)
    extract_facts_from_clause(c2)

    f1 = evaluate_clause(c1, "payment")
    f2 = evaluate_clause(c2, "payment")

    check_consistency([f1, f2], [c1, c2])

    assert f1.evidence_status == "conflicting"
    assert f2.evidence_status == "conflicting"
    assert "A-011" in f1.conflicts_with
    assert "A-004" in f2.conflicts_with
    assert f1.rating in ("YELLOW", "RED")
    assert f2.rating in ("YELLOW", "RED")


def test_8_yaml_configuration_and_invalid_fallback():
    """Requirement 8: YAML configuration loading and safe fallback."""
    rubric = load_risk_rules()
    assert "categories" in rubric
    assert "absence" in rubric
    assert "ambiguity_phrases" in rubric

    # Safe handling of non-existent path
    empty_cfg = load_risk_rules(Path("non_existent_rubric.yaml"))
    assert empty_cfg == {}


def test_9_risk_heatmap_data_generation():
    """Requirement 9: Complete ContractAnalysis data for UI heatmap."""
    doc = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    analysis = analyse_contract(doc)

    assert isinstance(analysis, ContractAnalysis)
    assert 0 <= analysis.overall_risk_score <= 100
    for cat in ["payment", "ip", "termination", "scope", "liability", "confidentiality"]:
        assert cat in analysis.category_ratings
        assert cat in analysis.category_scores


def test_10_contract_a_and_b_independent_processing():
    """Requirement 10: Contract A and Contract B processed independently."""
    doc_a = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    doc_b = load_document(SAMPLE_DIR / "contract_b.pdf", doc_id="B")

    analysis_a = analyse_contract(doc_a)
    analysis_b = analyse_contract(doc_b)

    assert analysis_a.doc_id == "A"
    assert analysis_b.doc_id == "B"
    # Contract B contains aggressive client terms, should have higher risk score
    assert analysis_b.overall_risk_score > analysis_a.overall_risk_score
    assert analysis_b.category_ratings["liability"] == "RED"


def test_11_golden_test_set():
    """Requirement 11: Validate golden test set G1-G5."""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    for item in goldens:
        gid = item["id"]
        if gid == "G5":
            # Missing category
            finding = evaluate_absence(item["category"], "A")
            assert finding.rating == item["expected_rating"], f"Golden {gid} rating mismatch"
            assert finding.evidence_status == item["evidence_status"]
        elif gid in ("G1", "G2", "G3", "G4"):
            clause = Clause(
                clause_id=gid,
                doc_id="A",
                heading=item["case"],
                text=item["text"],
                char_start=0,
                char_end=len(item["text"]),
                category=item["category"],
            )
            extract_facts_from_clause(clause)
            finding = evaluate_clause(clause, item["category"])

            assert finding.rating == item["expected_rating"], f"Golden {gid} rating mismatch (got {finding.rating}, expected {item['expected_rating']})"
            assert finding.evidence_status == item["evidence_status"], f"Golden {gid} evidence status mismatch"

            rule_ids = [r.rule_id for r in finding.triggered_rules]
            for exp_r in item["expected_rules"]:
                assert exp_r in rule_ids, f"Golden {gid} expected rule {exp_r} not found in {rule_ids}"
