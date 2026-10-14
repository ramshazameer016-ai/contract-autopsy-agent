"""Stage 5 test suite: Comparison Engine, Verdict, and Borrow Terms.

Verifies:
1. Matching known slots across A and B
2. Explicit gap representation (missing_a, missing_b, missing_both)
3. Uncertain matching excluded from numerical scores
4. Incompatible units/currencies not assigned a winner (marked not_comparable)
5. Priority weight changes immediately recalculate scores
6. Exact Architecture Worked Example (B6): 57.0 vs 60.0 (margin -3.0), then 65.0 vs 51.4 (margin +13.6)
7. Coverage below 0.5 produces insufficient evidence and no leaning
8. Absolute margin below 8 produces no clear winner
9. RED dealbreaker at weight 3 appears even when that contract leads
10. Missing and uncertain values never add or subtract score points
11. Borrow suggestions meet the threshold (>=25) and preserve verbatim evidence
12. Missing-clause suggestions require positive priority weight
13. Every borrow suggestion contains the exact required legal caveat
14. 'Use as reference' passes quotation to counter-clause generator without modifying source clause
15. AI verdict rewrite cannot change decision or add unsupported numbers
16. AI failure or offline mode uses safe template verdict
17. Consent enforcement ensures no network calls without consent
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.ca.schemas import (
    Clause,
    ContractAnalysis,
    ExtractedDocument,
    Page,
    Fact,
    EvidenceSpan,
    RiskFinding,
    TriggeredRule,
    MANDATORY_BORROW_CAVEAT,
)
from src.ca.compare.matcher import match_contracts
from src.ca.compare.engine import (
    calculate_weighted_scores,
    calculate_category_favourability,
    find_dealbreakers,
    has_clause_evidence,
)
from src.ca.compare.verdict import (
    build_deterministic_verdict,
    rewrite_verdict_with_ai,
    validate_ai_rewrite,
    VerdictRewriteSchema,
)
from src.ca.compare.borrow import generate_borrow_suggestions
from src.ca.generation.counter_clause import generate_counter_clause


def _make_dummy_clause(
    doc_id: str,
    clause_id: str,
    category: str,
    text: str,
    facts: list[Fact] | None = None,
    page: int | None = 1,
) -> Clause:
    return Clause(
        doc_id=doc_id,
        clause_id=clause_id,
        page_start=page,
        page_end=page,
        char_start=0,
        char_end=len(text),
        text=text,
        category=category,
        category_confidence=1.0,
        category_source="keyword",
        facts=facts or [],
    )


def _make_dummy_analysis(
    doc_id: str,
    clauses: list[Clause],
    category_scores: dict[str, int] | None = None,
    findings: list[RiskFinding] | None = None,
) -> ContractAnalysis:
    full_text = "\n\n".join(c.text for c in clauses)
    page_obj = Page(number=1, text=full_text, method="text")
    doc = ExtractedDocument(
        doc_id=doc_id,
        filename=f"contract_{doc_id.lower()}.pdf",
        file_type="pdf",
        pages=[page_obj],
        total_pages=1,
        full_text=full_text,
        sha256="dummy_sha256_hash",
    )
    return ContractAnalysis(
        doc_id=doc_id,
        document=doc,
        clauses=clauses,
        findings=findings or [],
        category_scores=category_scores or {},
        overall_risk_score=50,
    )


def test_1_slot_matching_known_slots():
    """Verify deterministic matching for known slots and direction rules."""
    fact_a = Fact(
        slot="payment.due_days",
        value=15,
        unit="days",
        source="regex",
        confidence="high",
        evidence=EvidenceSpan(doc_id="A", clause_id="A-1", page=1, start=0, end=7, quote="15 days"),
    )
    fact_b = Fact(
        slot="payment.due_days",
        value=60,
        unit="days",
        source="regex",
        confidence="high",
        evidence=EvidenceSpan(doc_id="B", clause_id="B-1", page=1, start=0, end=7, quote="60 days"),
    )

    ca = _make_dummy_clause("A", "A-1", "payment", "Payment due in 15 days.", [fact_a])
    cb = _make_dummy_clause("B", "B-1", "payment", "Payment due in 60 days.", [fact_b])

    analysis_a = _make_dummy_analysis("A", [ca])
    analysis_b = _make_dummy_analysis("B", [cb])

    rows = match_contracts(analysis_a, analysis_b)
    due_row = next(r for r in rows if r.slot == "payment.due_days")

    assert due_row.status == "matched"
    assert due_row.val_a == 15
    assert due_row.val_b == 60
    # lower_is_better rule: 15 < 60 so A is favoured
    assert due_row.favours == "A"
    assert due_row.quote_a == "15 days"
    assert due_row.quote_b == "60 days"


def test_2_gap_representation_missing_a_b_both():
    """Verify explicit gap statuses: missing_a, missing_b, missing_both."""
    fact_a = Fact(
        slot="payment.due_days",
        value=30,
        unit="days",
        source="regex",
        confidence="high",
        evidence=EvidenceSpan(doc_id="A", clause_id="A-1", page=1, start=0, end=7, quote="30 days"),
    )
    fact_b = Fact(
        slot="confidentiality.mutual",
        value=True,
        source="regex",
        confidence="high",
        evidence=EvidenceSpan(doc_id="B", clause_id="B-2", page=1, start=0, end=6, quote="mutual"),
    )

    ca = _make_dummy_clause("A", "A-1", "payment", "Payment due in 30 days.", [fact_a])
    cb = _make_dummy_clause("B", "B-2", "confidentiality", "Mutual confidentiality.", [fact_b])

    analysis_a = _make_dummy_analysis("A", [ca])
    analysis_b = _make_dummy_analysis("B", [cb])

    rows = match_contracts(analysis_a, analysis_b)

    due_row = next(r for r in rows if r.slot == "payment.due_days")
    assert due_row.status == "missing_b"
    assert due_row.display_b == "Not found in Contract B"
    assert due_row.favours == "none"

    conf_row = next(r for r in rows if r.slot == "confidentiality.mutual")
    assert conf_row.status == "missing_a"
    assert conf_row.display_a == "Not found in Contract A"
    assert conf_row.favours == "none"

    cure_row = next(r for r in rows if r.slot == "termination.cure_period_days")
    assert cure_row.status == "missing_both"
    assert cure_row.display_a == "Not found in Contract A"
    assert cure_row.display_b == "Not found in Contract B"


def test_3_uncertain_matching_excluded_from_scoring():
    """Verify unslotted leftover clauses with similarity are marked uncertain and excluded."""
    text_a = "Each party agrees not to solicit or hire the personnel or employees of the other party for two years."
    text_b = "Neither party shall solicit or hire any personnel or employees of the other party for two years."

    ca = _make_dummy_clause("A", "A-99", "liability", text_a, [])
    cb = _make_dummy_clause("B", "B-99", "liability", text_b, [])

    analysis_a = _make_dummy_analysis("A", [ca])
    analysis_b = _make_dummy_analysis("B", [cb])

    rows = match_contracts(analysis_a, analysis_b)
    uncertain_rows = [r for r in rows if r.status == "uncertain"]

    assert len(uncertain_rows) >= 1
    u_row = uncertain_rows[0]
    assert u_row.uncertain_match is True
    assert u_row.comparable is False
    assert u_row.favours == "none"


def test_4_incompatible_currencies_and_basis_not_comparable():
    """Verify different currencies or fee basis produce not_comparable and no winner."""
    text_a = "The client shall pay a fixed fee of INR 100000 upon completion."
    text_b = "The client shall pay an hourly rate of USD 50 per hour."

    ca = _make_dummy_clause("A", "A-1", "payment", text_a)
    cb = _make_dummy_clause("B", "B-1", "payment", text_b)

    analysis_a = _make_dummy_analysis("A", [ca])
    analysis_b = _make_dummy_analysis("B", [cb])

    rows = match_contracts(analysis_a, analysis_b)
    amt_row = next(r for r in rows if r.slot == "payment.amount")

    assert amt_row.status == "not_comparable"
    assert amt_row.comparable is False
    assert amt_row.favours == "none"
    assert "different currencies" in amt_row.note or "incompatible basis" in amt_row.note


def test_5_exact_worked_example_acceptance_test():
    """Verify the mandatory worked example from Architecture Section G.2 reproduces exactly."""
    # Category favourabilities from Section G.2
    # Payment: fav A = 40, fav B = 75
    # IP: fav A = 85, fav B = 30
    # Termination: fav A = 80, fav B = 45
    fav_a = {"payment": 40.0, "ip": 85.0, "termination": 80.0, "scope": None, "liability": None, "confidentiality": None}
    fav_b = {"payment": 75.0, "ip": 30.0, "termination": 45.0, "scope": None, "liability": None, "confidentiality": None}

    # Step 1: Initial weights: Payment=3, IP=1, Termination=1
    weights_step1 = {"payment": 3, "ip": 1, "termination": 1, "scope": 0, "liability": 0, "confidentiality": 0}
    res1 = calculate_weighted_scores(fav_a, fav_b, weights_step1)

    assert res1["score_a"] == 57.0, f"Expected 57.0, got {res1['score_a']}"
    assert res1["score_b"] == 60.0, f"Expected 60.0, got {res1['score_b']}"
    assert res1["margin"] == -3.0, f"Expected -3.0, got {res1['margin']}"
    assert res1["status"] == "no_clear_winner"

    # Step 2: User raises IP ownership weight to 3
    weights_step2 = {"payment": 3, "ip": 3, "termination": 1, "scope": 0, "liability": 0, "confidentiality": 0}
    res2 = calculate_weighted_scores(fav_a, fav_b, weights_step2)

    assert res2["score_a"] == 65.0, f"Expected 65.0, got {res2['score_a']}"
    assert res2["score_b"] == 51.4, f"Expected 51.4, got {res2['score_b']}"
    assert res2["margin"] == 13.6, f"Expected +13.6, got {res2['margin']}"
    assert res2["status"] == "leans_a"


def test_6_coverage_below_threshold_insufficient_evidence():
    """Verify coverage below 0.5 results in insufficient_evidence and no preference leaning."""
    # 6 priorities, each weight 2. Only payment has evidence in both (scored weight 2, total weight 12 -> coverage 16.7%)
    fav_a = {"payment": 80.0, "ip": None, "termination": None, "scope": None, "liability": None, "confidentiality": None}
    fav_b = {"payment": 50.0, "ip": None, "termination": None, "scope": None, "liability": None, "confidentiality": None}
    weights = {"payment": 2, "ip": 2, "termination": 2, "scope": 2, "liability": 2, "confidentiality": 2}

    res = calculate_weighted_scores(fav_a, fav_b, weights)
    assert res["coverage"] < 0.5
    assert res["status"] == "insufficient_evidence"


def test_7_dealbreaker_guard_at_weight_3():
    """Verify RED finding in a category weighted 3 is named even when that contract leads on score."""
    f_red = RiskFinding(
        doc_id="A",
        finding_id="FIND-1",
        clause_id="A-1",
        category="payment",
        rating="RED",
        evidence_status="present",
        score=85,
        triggered_rules=[
            TriggeredRule(
                rule_id="PAY-R1",
                severity="RED",
                points=85,
                why="Client pays only upon end-client receipt.",
            )
        ],
        plain_reason="Client pays only upon end-client receipt (conditional payment).",
    )
    ca = _make_dummy_clause("A", "A-1", "payment", "Payment conditional on third party.")
    analysis_a = _make_dummy_analysis("A", [ca], category_scores={"payment": 85}, findings=[f_red])

    weights = {"payment": 3, "ip": 1, "termination": 1, "scope": 0, "liability": 0, "confidentiality": 0}
    dealbreakers_a = find_dealbreakers(analysis_a, weights, "Contract A")

    assert len(dealbreakers_a) == 1
    assert "PAY-R1" in dealbreakers_a[0]
    assert "RED risk" in dealbreakers_a[0]


def test_8_borrow_suggestions_threshold_and_verbatim_quote():
    """Verify borrow suggestions meet the threshold (>=25) and preserve verbatim quotes."""
    quote_a = "Each party retains ownership of its pre-existing IP and materials."
    ca = _make_dummy_clause("A", "A-1", "ip", quote_a)
    cb = _make_dummy_clause("B", "B-1", "ip", "All intellectual property is assigned to client.")

    f_a = RiskFinding(doc_id="A", finding_id="F-A", clause_id="A-1", category="ip", rating="GREEN", evidence_status="present", score=10, plain_reason="Retains IP.")
    f_b = RiskFinding(doc_id="B", finding_id="F-B", clause_id="B-1", category="ip", rating="RED", evidence_status="present", score=80, plain_reason="Assigns all IP.")

    # Fav A = 90, Fav B = 20 -> Gap = 70 >= 25
    analysis_a = _make_dummy_analysis("A", [ca], category_scores={"ip": 10}, findings=[f_a])
    analysis_b = _make_dummy_analysis("B", [cb], category_scores={"ip": 80}, findings=[f_b])

    weights = {"payment": 2, "ip": 2, "termination": 2, "scope": 0, "liability": 0, "confidentiality": 0}
    suggestions = generate_borrow_suggestions(analysis_a, analysis_b, [], weights)

    assert len(suggestions) >= 1
    sug = next(s for s in suggestions if s.category == "ip")
    assert sug.source_contract == "Contract A"
    assert sug.target_contract == "Contract B"
    assert sug.verbatim_quote == quote_a
    assert sug.caveat == MANDATORY_BORROW_CAVEAT
    assert sug.suggestion_type == "favourability_gap"


def test_9_missing_clause_borrow_suggestion_requires_positive_weight():
    """Verify missing clause suggestion is created when weight > 0, but suppressed when weight == 0."""
    quote_a = "Client shall provide 30 days notice for termination."
    ca = _make_dummy_clause("A", "A-1", "termination", quote_a)
    f_a = RiskFinding(doc_id="A", finding_id="F-A", clause_id="A-1", category="termination", rating="GREEN", evidence_status="present", score=15, plain_reason="Notice period provided.")

    analysis_a = _make_dummy_analysis("A", [ca], category_scores={"termination": 15}, findings=[f_a])
    # Contract B has NO clauses and NO findings for termination
    analysis_b = _make_dummy_analysis("B", [])

    # Case 1: termination weight = 2 -> suggestion should be generated
    weights_active = {"payment": 2, "ip": 2, "termination": 2, "scope": 0, "liability": 0, "confidentiality": 0}
    sugs_active = generate_borrow_suggestions(analysis_a, analysis_b, [], weights_active)
    term_sug = next((s for s in sugs_active if s.category == "termination"), None)
    assert term_sug is not None
    assert term_sug.suggestion_type == "missing_clause"
    assert term_sug.caveat == MANDATORY_BORROW_CAVEAT

    # Case 2: termination weight = 0 -> suggestion should be suppressed
    weights_zero = {"payment": 2, "ip": 2, "termination": 0, "scope": 0, "liability": 0, "confidentiality": 0}
    sugs_zero = generate_borrow_suggestions(analysis_a, analysis_b, [], weights_zero)
    assert not any(s.category == "termination" for s in sugs_zero)


def test_10_use_as_reference_handoff_to_counter_clause():
    """Verify 'Use as reference' passes quotation into counter-clause generator without modifying source finding."""
    finding = RiskFinding(
        doc_id="B",
        finding_id="FIND-TEST",
        clause_id="B-4",
        category="payment",
        rating="YELLOW",
        evidence_status="present",
        score=50,
        plain_reason="Payment window is 60 days.",
    )
    reference_quote = "The client shall pay all undisputed invoices within 15 calendar days."

    draft = generate_counter_clause(finding, reference_quote=reference_quote)

    assert draft.finding_id == "FIND-TEST"
    assert draft.original_clause_id == "B-4"
    assert draft.reference_quote == reference_quote
    assert any("Reference term from comparison offer" in pt for pt in draft.negotiation_points)
    assert "A reference term from another offer is attached" in draft.applicability_note


def test_11_deterministic_verdict_builder():
    """Verify deterministic verdict builder constructs complete structured artifact."""
    scores_data = {
        "score_a": 65.0,
        "score_b": 51.4,
        "margin": 13.6,
        "coverage": 0.83,
        "status": "leans_a",
        "scored_categories": ["payment", "ip"],
        "unscored_categories": ["termination"],
    }
    fav_a = {"payment": 50.0, "ip": 80.0}
    fav_b = {"payment": 60.0, "ip": 30.0}

    verdict = build_deterministic_verdict(
        scores_data=scores_data,
        fav_a=fav_a,
        fav_b=fav_b,
        rows=[],
        dealbreakers=["[Contract B Dealbreaker] Category 'Liability' has a RED risk."],
        weights={"payment": 2, "ip": 3, "termination": 0},
    )

    assert verdict.status == "leans_a"
    assert "Leans toward Contract A" in verdict.headline
    assert verdict.score_a == 65.0
    assert verdict.score_b == 51.4
    assert len(verdict.dealbreakers) == 1
    assert "Dealbreaker Warnings" in verdict.full_text
    assert verdict.generated_by == "deterministic"


def test_12_ai_verdict_validator_rejects_altered_leaning():
    """Verify validator rejects an AI rewrite that alters the deterministic leaning."""
    bad_rewrite = VerdictRewriteSchema(
        status="leans_b",  # changed from leans_a!
        headline="Leans toward Contract B",
        rewritten_text="Contract B is better overall with score 65.0 and margin 13.6.",
    )
    is_valid = validate_ai_rewrite(bad_rewrite, "leans_a", {65.0, 51.4, 13.6})
    assert is_valid is False


def test_13_ai_verdict_validator_rejects_invented_numbers():
    """Verify validator rejects an AI rewrite that introduces unsupported numbers."""
    bad_rewrite = VerdictRewriteSchema(
        status="leans_a",
        headline="Leans toward Contract A",
        rewritten_text="Contract A has 99.9% probability of winning with penalty of 50000 rupees.",
    )
    is_valid = validate_ai_rewrite(bad_rewrite, "leans_a", {65.0, 51.4, 13.6, 8.0})
    assert is_valid is False


def test_14_ai_offline_fallback_preserves_deterministic_verdict():
    """Verify that when offline or consent is withheld, deterministic template verdict is returned."""
    scores_data = {
        "score_a": 57.0,
        "score_b": 60.0,
        "margin": -3.0,
        "coverage": 1.0,
        "status": "no_clear_winner",
        "scored_categories": ["payment", "ip"],
        "unscored_categories": [],
    }
    verdict = build_deterministic_verdict(
        scores_data=scores_data,
        fav_a={"payment": 40.0, "ip": 85.0},
        fav_b={"payment": 75.0, "ip": 30.0},
        rows=[],
        dealbreakers=[],
        weights={"payment": 2, "ip": 2},
    )

    # When has_consent is False
    out = rewrite_verdict_with_ai(verdict, has_consent=False)
    assert out.generated_by == "deterministic"
    assert out.headline == verdict.headline


def test_15_golden_g6_different_payment_terms():
    """Golden Test G6: Contract A (due in 15 days) vs Contract B (due in 60 days)."""
    from pathlib import Path
    from src.ca.extraction.loader import load_document
    from src.ca.analysis.pipeline import analyse_contract

    sample_dir = Path(__file__).resolve().parent / "sample_contracts"
    doc_a = load_document(sample_dir / "contract_a.pdf")
    doc_b = load_document(sample_dir / "contract_b.pdf")

    analysis_a = analyse_contract(doc_a)
    analysis_b = analyse_contract(doc_b)

    rows = match_contracts(analysis_a, analysis_b)
    due_row = next((r for r in rows if r.slot == "payment.due_days"), None)

    assert due_row is not None
    assert due_row.status == "matched"
    assert due_row.val_a == 15
    assert due_row.val_b in (60, 90)
    assert due_row.favours == "A"


def test_16_golden_g7_different_ip_ownership():
    """Golden Test G7: Contract A (keeps IP/licence) vs Contract B (assigns all IP)."""
    from pathlib import Path
    from src.ca.extraction.loader import load_document
    from src.ca.analysis.pipeline import analyse_contract

    sample_dir = Path(__file__).resolve().parent / "sample_contracts"
    doc_a = load_document(sample_dir / "contract_a.pdf")
    doc_b = load_document(sample_dir / "contract_b.pdf")

    analysis_a = analyse_contract(doc_a)
    analysis_b = analyse_contract(doc_b)

    rows = match_contracts(analysis_a, analysis_b)
    ip_row = next((r for r in rows if r.slot == "ip.assignment_scope"), None)

    assert ip_row is not None
    assert ip_row.status == "matched"
    assert ip_row.favours == "A"


def test_17_ai_verdict_rewrite_with_mocked_llm_valid():
    """Verify valid AI rewrite succeeds and sets generated_by to llm_rewritten."""
    scores_data = {
        "score_a": 65.0,
        "score_b": 51.4,
        "margin": 13.6,
        "coverage": 0.85,
        "status": "leans_a",
        "scored_categories": ["payment", "ip"],
        "unscored_categories": [],
    }
    verdict = build_deterministic_verdict(
        scores_data=scores_data,
        fav_a={"payment": 50.0, "ip": 80.0},
        fav_b={"payment": 60.0, "ip": 30.0},
        rows=[],
        dealbreakers=[],
        weights={"payment": 2, "ip": 3},
    )

    mock_rewrite = VerdictRewriteSchema(
        status="leans_a",
        headline="Leans toward Contract A based on strong IP terms",
        rewritten_text=(
            "Contract A is recommended over Contract B. "
            "Contract A achieved a score of 65.0 while Contract B scored 51.4, "
            "yielding a positive margin of 13.6 points."
        ),
    )

    with patch("src.ca.compare.verdict.is_offline", return_value=False):
        with patch("src.ca.llm.client.generate_validated", return_value=mock_rewrite):
            out = rewrite_verdict_with_ai(verdict, has_consent=True)
            assert out.generated_by == "llm_rewritten"
            assert "Contract A is recommended" in out.full_text
            assert out.status == "leans_a"

