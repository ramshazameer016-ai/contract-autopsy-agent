
"""Tests for deterministic Stage 4 counter-clause drafting."""

import pytest

from src.ca.generation.counter_clause import generate_counter_clause
from src.ca.schemas import EvidenceSpan, RiskFinding


def make_payment_finding() -> RiskFinding:
    return RiskFinding(
        finding_id="F-A-A-001",
        doc_id="A",
        category="payment",
        clause_id="A-001",
        rating="RED",
        evidence_status="present",
        score=80,
        triggered_rules=[],
        evidence=[
            EvidenceSpan(
                doc_id="A",
                clause_id="A-001",
                page=1,
                start=0,
                end=len("Payment may be delayed indefinitely."),
                quote="Payment may be delayed indefinitely.",
            )
        ],
        plain_reason="Payment timing is unclear.",
    )


def test_counter_clause_preserves_source_finding_and_evidence():
    finding = make_payment_finding()
    original_rating = finding.rating
    original_evidence = [item.model_dump() for item in finding.evidence]

    draft = generate_counter_clause(finding)

    assert draft.finding_id == finding.finding_id
    assert draft.original_clause_id == "A-001"
    assert draft.original_rating == "RED"
    assert draft.original_evidence_quotes == [
        "Payment may be delayed indefinitely."
    ]
    assert finding.rating == original_rating
    assert [item.model_dump() for item in finding.evidence] == original_evidence


def test_counter_clause_contains_proposal_and_negotiation_points():
    draft = generate_counter_clause(make_payment_finding())

    assert draft.generated_by == "template"
    assert "30 calendar days" in draft.proposed_clause
    assert draft.negotiation_points
    assert "Not legal advice" in draft.disclaimer


def test_counter_clause_rejects_unsupported_category():
    finding = make_payment_finding()
    finding.category = "other"

    with pytest.raises(ValueError, match="No counter-clause template"):
        generate_counter_clause(finding)
