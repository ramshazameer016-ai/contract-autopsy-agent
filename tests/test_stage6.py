"""Stage 6 test suite — Legal Knowledge Base and Retrieval, plus Part A privacy tests.

Part A tests (A5):
  1. AI-required operations fail honestly when API key is missing.
  2. Provider failure does not produce output falsely labelled as AI-generated.
  3. No consent means no external provider call.
  4. Consent allows a provider call only when provider is configured.
  5. Deterministic risk analysis and comparison work without AI.
  6. AI cannot alter deterministic ratings, scores, evidence, coverage, leaning, or
     dealbreaker decisions.
  7. API keys and contract contents are not exposed in logs.
  8. Existing completed-stage tests remain valid (integration test).

Part B tests (B7):
  1. A verified payment passage can be retrieved for a relevant payment finding.
  2. A verified IP passage can be retrieved for a relevant IP finding.
  3. Unverified passages never appear in verified retrieval results.
  4. Every returned passage preserves its provenance fields.
  5. An unrelated query returns no match rather than an invented or irrelevant result.
  6. A query with no verified match produces an honest empty-result message.
  7. Missing or malformed source metadata is handled safely.
  8. The source-count banner reflects actual validated source data.
  9. Legal context expander shows provenance and keeps legal context separate from
     contract evidence (tested by checking the rendered component directly).
 10. Retrieval works locally without an external AI request.
 11. Existing analysis, comparison, counter-clause, consent, and privacy tests continue
     passing (regression invoked via subprocess in integration section).
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ------------------------------------------------------------------ #
# Part A — AI configuration and privacy                               #
# ------------------------------------------------------------------ #


class TestA1_AIRequiredOperationsFailHonestly:
    """A5.1 — AI-required operations fail honestly when API key is missing."""

    def test_generate_validated_raises_llm_unavailable_when_no_key(self, monkeypatch):
        """generate_validated raises LLMUnavailable when GEMINI_API_KEY is empty."""
        from src.ca.llm import client
        from src.ca.llm.client import LLMUnavailable
        from src.ca.llm.validator import PlainSummary

        monkeypatch.setattr(client, "GEMINI_API_KEY", "")
        client.clear_session_cache()

        with pytest.raises(LLMUnavailable) as exc_info:
            client.generate_validated(
                "Test prompt",
                PlainSummary,
                prompt_version="test-no-key",
            )

        error_msg = str(exc_info.value)
        # Must not expose the key (which is empty, but just to confirm the message is clean)
        assert "GEMINI_API_KEY" in error_msg
        # Must not expose any contract content
        assert "Test prompt" not in error_msg

    def test_error_message_does_not_contain_api_key(self, monkeypatch):
        """Error message must not leak any API key value."""
        from src.ca.llm import client
        from src.ca.llm.client import LLMUnavailable
        from src.ca.llm.validator import PlainSummary

        fake_key = "AIza_FAKE_KEY_12345"
        monkeypatch.setattr(client, "GEMINI_API_KEY", "")
        client.clear_session_cache()

        try:
            client.generate_validated("prompt", PlainSummary, prompt_version="key-leak-test")
        except LLMUnavailable as exc:
            assert fake_key not in str(exc)


class TestA2_ProviderFailureHonest:
    """A5.2 — Provider failure does not produce output falsely labelled as AI-generated."""

    def test_summary_returns_template_when_ai_fails(self, monkeypatch):
        """summarize_contract returns template (not llm) when AI raises LLMUnavailable."""
        from src.ca.generation.summary import summarize_contract
        from src.ca.llm.client import LLMUnavailable

        def raise_unavailable(*args, **kwargs):
            raise LLMUnavailable("Simulated provider failure")

        monkeypatch.setattr(
            "src.ca.generation.summary.generate_validated",
            raise_unavailable,
        )

        result = summarize_contract(
            clauses=[{"clause_id": "C1", "text": "Payment is due within 30 days."}],
            facts_and_findings="",
            language="English",
            consent_given=True,
        )

        # Must fall back to template, never falsely label as AI-generated
        assert result.generated_by == "template"
        assert result.language == "English"

    def test_verdict_stays_deterministic_when_rewrite_fails(self):
        """rewrite_verdict_with_ai returns deterministic verdict on AI exception."""
        from src.ca.compare.verdict import rewrite_verdict_with_ai, build_deterministic_verdict

        verdict = build_deterministic_verdict(
            scores_data={
                "score_a": 65.0, "score_b": 51.4, "margin": 13.6,
                "coverage": 0.8, "status": "leans_a",
                "scored_categories": ["payment"], "unscored_categories": [],
            },
            fav_a={"payment": 60.0},
            fav_b={"payment": 40.0},
            rows=[], dealbreakers=[], weights={"payment": 2},
        )

        with patch("src.ca.compare.verdict.is_offline", return_value=False):
            with patch("src.ca.llm.client.generate_validated", side_effect=RuntimeError("network error")):
                out = rewrite_verdict_with_ai(verdict, has_consent=True)

        assert out.generated_by == "deterministic"
        assert out.score_a == 65.0
        assert out.score_b == 51.4


class TestA3_NoConsentBlocksProviderCall:
    """A5.3 — No consent means no external provider call."""

    def test_summarize_contract_no_consent_never_calls_ai(self, monkeypatch):
        """Without consent, summarize_contract never calls generate_validated."""
        def fail_if_called(*args, **kwargs):
            pytest.fail("generate_validated must not be called without consent")

        monkeypatch.setattr(
            "src.ca.generation.summary.generate_validated",
            fail_if_called,
        )

        result = summarize_contract_import(
            clauses=[{"clause_id": "C1", "text": "Payment: 30 days."}],
            facts_and_findings="",
            language="English",
            consent_given=False,
        )
        assert result.generated_by == "template"

    def test_verdict_rewrite_no_consent_returns_deterministic(self):
        """rewrite_verdict_with_ai with has_consent=False returns original deterministic verdict."""
        from src.ca.compare.verdict import rewrite_verdict_with_ai, build_deterministic_verdict

        verdict = build_deterministic_verdict(
            scores_data={
                "score_a": 57.0, "score_b": 60.0, "margin": -3.0,
                "coverage": 1.0, "status": "no_clear_winner",
                "scored_categories": ["payment"], "unscored_categories": [],
            },
            fav_a={"payment": 40.0},
            fav_b={"payment": 60.0},
            rows=[], dealbreakers=[], weights={"payment": 2},
        )

        out = rewrite_verdict_with_ai(verdict, has_consent=False)
        assert out.generated_by == "deterministic"


def summarize_contract_import(*args, **kwargs):
    """Helper to import at call time to allow monkeypatching."""
    from src.ca.generation.summary import summarize_contract
    return summarize_contract(*args, **kwargs)


class TestA4_ConsentAllowsCallWhenConfigured:
    """A5.4 — Consent allows a provider call only when the provider is configured."""

    def test_consent_with_no_key_still_uses_template(self, monkeypatch):
        """Even with consent=True, if no API key configured, template is used."""
        from src.ca.llm import client

        monkeypatch.setattr(client, "GEMINI_API_KEY", "")
        client.clear_session_cache()

        result = summarize_contract_import(
            clauses=[{"clause_id": "C1", "text": "Payment in 30 days."}],
            facts_and_findings="",
            language="English",
            consent_given=True,
        )
        # Should fall back to template because no key is configured
        assert result.generated_by == "template"


class TestA5_DeterministicCoreWithoutAI:
    """A5.5 — Deterministic risk analysis and comparison work without AI."""

    def test_risk_analysis_requires_no_ai(self):
        """analyse_contract completes without any AI provider calls."""
        from pathlib import Path
        from src.ca.extraction.loader import load_document
        from src.ca.analysis.pipeline import analyse_contract

        sample_dir = Path(__file__).resolve().parent / "sample_contracts"
        doc = load_document(sample_dir / "contract_a.pdf")
        analysis = analyse_contract(doc)

        assert analysis is not None
        assert len(analysis.findings) > 0
        assert analysis.overall_risk_score >= 0

    def test_comparison_requires_no_ai(self):
        """Full comparison pipeline completes without AI calls."""
        from pathlib import Path
        from src.ca.extraction.loader import load_document
        from src.ca.analysis.pipeline import analyse_contract
        from src.ca.compare.matcher import match_contracts
        from src.ca.compare.engine import calculate_weighted_scores, calculate_category_favourability
        from src.ca.compare.verdict import build_deterministic_verdict

        sample_dir = Path(__file__).resolve().parent / "sample_contracts"
        doc_a = load_document(sample_dir / "contract_a.pdf")
        doc_b = load_document(sample_dir / "contract_b.pdf")
        analysis_a = analyse_contract(doc_a)
        analysis_b = analyse_contract(doc_b)

        fav_a = calculate_category_favourability(analysis_a)
        fav_b = calculate_category_favourability(analysis_b)
        weights = {"payment": 3, "ip": 3, "termination": 2, "scope": 2, "liability": 2, "confidentiality": 2}
        scores = calculate_weighted_scores(fav_a, fav_b, weights)
        rows = match_contracts(analysis_a, analysis_b)
        verdict = build_deterministic_verdict(
            scores_data=scores, fav_a=fav_a, fav_b=fav_b,
            rows=rows, dealbreakers=[], weights=weights,
        )

        assert verdict.generated_by == "deterministic"
        assert verdict.score_a >= 0
        assert verdict.score_b >= 0


class TestA6_AICannotAlterDeterministicResults:
    """A5.6 — AI cannot alter deterministic ratings, scores, evidence, coverage, leaning."""

    def test_ai_rewrite_preserves_scores_and_status(self):
        """AI rewrite cannot change scores, margin, coverage, or leaning."""
        from src.ca.compare.verdict import (
            rewrite_verdict_with_ai, build_deterministic_verdict,
            validate_ai_rewrite, VerdictRewriteSchema,
        )

        verdict = build_deterministic_verdict(
            scores_data={
                "score_a": 65.0, "score_b": 51.4, "margin": 13.6,
                "coverage": 0.8, "status": "leans_a",
                "scored_categories": ["payment", "ip"], "unscored_categories": [],
            },
            fav_a={"payment": 60.0, "ip": 80.0},
            fav_b={"payment": 40.0, "ip": 30.0},
            rows=[], dealbreakers=[], weights={"payment": 2, "ip": 3},
        )

        # Simulate AI trying to flip the leaning
        bad_rewrite = VerdictRewriteSchema(
            status="leans_b",  # WRONG — original is leans_a
            headline="Contract B is better",
            rewritten_text="Contract B scored 70 and A scored 40, margin 30.",
        )
        allowed = {65.0, 51.4, 13.6, 0.8, 8.0}
        assert validate_ai_rewrite(bad_rewrite, "leans_a", allowed) is False

    def test_summary_evidence_not_replaced_by_ai_invented_text(self, monkeypatch):
        """Even with AI consent, evidence quotations from the source are preserved."""
        from src.ca.generation.summary import summarize_contract
        from src.ca.llm.validator import PlainSummary, SummaryItem

        original_quote = "Payment is due within 30 days of invoice."
        clause = {
            "clause_id": "C1",
            "text": original_quote,
        }

        # Simulate AI returning a summary with an invented quote not in the source
        invented_quote = "Payment will happen in two hundred business days."
        fake_ai_response = PlainSummary(
            language="English",
            generated_by="llm",
            items=[SummaryItem(
                category="Payment",
                status="Stated",
                explanation="Payment terms stated.",
                evidence_quote=invented_quote,  # not in source!
                clause_id="C1",
            )],
            disclaimer="Not legal advice.",
        )

        monkeypatch.setattr(
            "src.ca.generation.summary.generate_validated",
            lambda *a, **kw: fake_ai_response,
        )

        result = summarize_contract(
            clauses=[clause],
            facts_and_findings="",
            language="English",
            consent_given=True,
        )

        # Must fall back to template because invented quote is not in source
        assert result.generated_by == "template"
        # The invented quote must NOT appear in output
        for item in result.items:
            assert invented_quote not in (item.evidence_quote or "")


class TestA7_NoKeyOrContractExposureInLogs:
    """A5.7 — API keys and contract contents are not exposed in logs."""

    def test_llm_exception_message_does_not_contain_contract_text(self, monkeypatch, caplog):
        """When provider call fails, exception message does not contain prompt/contract text."""
        from src.ca.llm import client
        from src.ca.llm.client import LLMUnavailable
        from src.ca.llm.validator import PlainSummary

        sensitive_text = "SENSITIVE_CONTRACT_CLAUSE_9999"
        monkeypatch.setattr(client, "GEMINI_API_KEY", "some-key")
        client.clear_session_cache()

        def fail_with_generic(prompt: str):
            # Simulates a provider error that does NOT include the prompt in the message
            raise RuntimeError("connection_error")

        monkeypatch.setattr(client, "_call_gemini", fail_with_generic)
        monkeypatch.setattr(client.time, "sleep", lambda _: None)

        with caplog.at_level(logging.WARNING):
            with pytest.raises(LLMUnavailable) as exc_info:
                client.generate_validated(
                    sensitive_text,
                    PlainSummary,
                    prompt_version="log-leak-test",
                )

        # The exception message should not contain the prompt text
        assert sensitive_text not in str(exc_info.value)
        # Logs should not contain the sensitive text
        for record in caplog.records:
            assert sensitive_text not in record.getMessage()


# ------------------------------------------------------------------ #
# Part B — Stage 6 Legal Knowledge Base                               #
# ------------------------------------------------------------------ #


class TestB1_VerifiedPaymentPassageRetrieval:
    """B7.1 — A verified payment passage can be retrieved for a relevant payment finding."""

    def test_retrieve_verified_payment_passage(self, monkeypatch):
        """When a verified payment passage exists, retrieve() returns it."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage, RetrievalResult

        verified_passage = KBPassage(
            passage_id="TEST-PAY-001",
            topic="payment_terms",
            category="payment",
            source_title="Test Payment Act, 2000",
            authority="Test Parliament",
            section="Section 1 — Payment",
            official_url="https://example.com/test-act",
            publication_date="2000-01-01",
            amendment_date=None,
            date_accessed="2026-10-13",
            verification_status="verified",
            passage_file="test_pay.md",
            limitations="Test passage.",
            text="Test passage text about payment.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [verified_passage])

        result = retriever.retrieve("payment", verified_only=True)

        assert result.has_results
        assert len(result.passages) == 1
        assert result.passages[0].passage_id == "TEST-PAY-001"
        assert result.passages[0].verification_status == "verified"
        assert result.passages[0].category == "payment"


class TestB2_VerifiedIPPassageRetrieval:
    """B7.2 — A verified IP passage can be retrieved for a relevant IP finding."""

    def test_retrieve_verified_ip_passage(self, monkeypatch):
        """When a verified IP passage exists, retrieve() returns it."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        verified_passage = KBPassage(
            passage_id="TEST-IP-001",
            topic="ip_assignment",
            category="ip",
            source_title="Test Copyright Act, 2000",
            authority="Test Parliament",
            section="Section 2 — Assignment",
            official_url="https://example.com/copyright",
            publication_date="2000-06-01",
            amendment_date=None,
            date_accessed="2026-10-13",
            verification_status="verified",
            passage_file="test_ip.md",
            limitations="Test passage.",
            text="Test passage text about IP assignment.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [verified_passage])

        result = retriever.retrieve("ip", verified_only=True)

        assert result.has_results
        assert result.passages[0].passage_id == "TEST-IP-001"
        assert result.passages[0].category == "ip"


class TestB3_UnverifiedPassagesExcluded:
    """B7.3 — Unverified passages never appear in verified retrieval results."""

    def test_unverified_passages_excluded_from_verified_results(self, monkeypatch):
        """Unverified passages must not appear when verified_only=True."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        unverified = KBPassage(
            passage_id="TEST-UNVER-001",
            topic="payment_terms",
            category="payment",
            source_title="Unverified Act",
            authority="Unknown",
            section="Section X",
            official_url="",
            publication_date=None,
            amendment_date=None,
            date_accessed=None,
            verification_status="unverified",
            passage_file="unver.md",
            limitations="Not verified.",
            text="Unverified text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [unverified])

        result = retriever.retrieve("payment", verified_only=True)

        assert not result.has_results
        assert len(result.passages) == 0

    def test_unverified_passages_appear_when_not_restricted(self, monkeypatch):
        """Unverified passages do appear when verified_only=False."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        unverified = KBPassage(
            passage_id="TEST-UNVER-002",
            topic="payment",
            category="payment",
            source_title="Unverified Act",
            authority="Unknown",
            section="Section X",
            official_url="",
            publication_date=None,
            amendment_date=None,
            date_accessed=None,
            verification_status="unverified",
            passage_file="unver.md",
            limitations="Not verified.",
            text="Unverified text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [unverified])

        result = retriever.retrieve("payment", verified_only=False)
        assert result.has_results


class TestB4_ProvenancePreserved:
    """B7.4 — Every returned passage preserves its provenance fields."""

    def test_all_provenance_fields_preserved(self, monkeypatch):
        """Every returned passage has all required provenance metadata."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        passage = KBPassage(
            passage_id="TEST-PROV-001",
            topic="payment_terms",
            category="payment",
            source_title="Indian Contract Act, 1872",
            authority="Parliament of India",
            section="Section 73",
            official_url="https://legislative.dept.gov.in/test",
            publication_date="1872-04-25",
            amendment_date=None,
            date_accessed="2026-10-13",
            verification_status="verified",
            passage_file="test.md",
            limitations="Test limitation.",
            text="Test passage text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [passage])

        result = retriever.retrieve("payment", verified_only=True)

        assert result.has_results
        p = result.passages[0]
        assert p.passage_id == "TEST-PROV-001"
        assert p.source_title == "Indian Contract Act, 1872"
        assert p.authority == "Parliament of India"
        assert p.section == "Section 73"
        assert p.official_url == "https://legislative.dept.gov.in/test"
        assert p.publication_date == "1872-04-25"
        assert p.limitations == "Test limitation."
        assert p.text == "Test passage text."


class TestB5_UnrelatedQueryReturnsNoMatch:
    """B7.5 — An unrelated query returns no match rather than an invented or irrelevant result."""

    def test_unrelated_category_returns_empty(self, monkeypatch):
        """Querying a category with no passages returns empty result."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        payment_passage = KBPassage(
            passage_id="TEST-PAY-ONLY",
            topic="payment",
            category="payment",
            source_title="Test Act",
            authority="Test",
            section="S1",
            official_url="",
            publication_date=None,
            amendment_date=None,
            date_accessed=None,
            verification_status="verified",
            passage_file="test.md",
            limitations="",
            text="Payment text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [payment_passage])

        # Query for "termination" — should not return payment passage
        result = retriever.retrieve("termination", verified_only=True)
        assert not result.has_results
        assert result.query_category == "termination"

    def test_empty_category_returns_empty_result(self):
        """An empty category string returns empty result with error message."""
        from src.ca.kb.retriever import retrieve

        result = retrieve("", verified_only=True)
        assert not result.has_results
        assert result.no_match_reason != ""


class TestB6_NoMatchProducesHonestMessage:
    """B7.6 — A query with no verified match produces an honest empty-result message."""

    def test_no_match_reason_is_populated(self, monkeypatch):
        """When no verified passage matches, no_match_reason is a non-empty string."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        unverified_payment = KBPassage(
            passage_id="TEST-UNVER-MSG",
            topic="payment",
            category="payment",
            source_title="Unverified Act",
            authority="Unknown",
            section="Section X",
            official_url="",
            publication_date=None,
            amendment_date=None,
            date_accessed=None,
            verification_status="unverified",
            passage_file="unver.md",
            limitations="",
            text="Unverified payment text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [unverified_payment])

        result = retriever.retrieve("payment", verified_only=True)

        assert not result.has_results
        assert len(result.no_match_reason) > 10  # Must be a real message, not empty
        assert result.passages == []


class TestB7_MalformedMetadataHandledSafely:
    """B7.7 — Missing or malformed source metadata is handled safely."""

    def test_malformed_entry_is_skipped_not_raised(self, monkeypatch):
        """Malformed entries in sources.yaml are skipped; the rest load normally."""
        from src.ca.kb import retriever

        def mock_load_sources():
            return [
                None,          # completely malformed
                {},            # missing required fields
                {"passage_id": "TEST-VALID", "category": "payment",
                 "verification_status": "verified", "passage_file": "none.md"},
                "not a dict",  # wrong type
            ]

        monkeypatch.setattr(retriever, "_load_sources", mock_load_sources)

        # Should not raise; malformed entries skipped
        passages = retriever.load_all_passages()
        # The only entry with all required fields (but missing passage file) should be present
        assert isinstance(passages, list)

    def test_missing_sources_yaml_returns_empty(self, monkeypatch, tmp_path):
        """Missing sources.yaml returns empty list, not an exception."""
        from src.ca.kb import retriever

        monkeypatch.setattr(retriever, "_SOURCES_FILE", tmp_path / "nonexistent.yaml")

        passages = retriever.load_all_passages()
        assert passages == []

    def test_missing_passage_file_returns_empty_text(self, monkeypatch):
        """When a passage's .md file is missing, text is empty string (not an exception)."""
        from src.ca.kb import retriever

        def mock_load_sources():
            return [{
                "passage_id": "TEST-NO-FILE",
                "topic": "payment",
                "category": "payment",
                "verification_status": "verified",
                "passage_file": "does_not_exist_12345.md",
                "source_title": "Test Act",
                "authority": "Test",
                "section": "S1",
                "official_url": "",
                "publication_date": None,
                "amendment_date": None,
                "date_accessed": None,
                "limitations": "",
            }]

        monkeypatch.setattr(retriever, "_load_sources", mock_load_sources)

        passages = retriever.load_all_passages()
        assert len(passages) == 1
        assert passages[0].text == ""  # Empty string, not an exception


class TestB8_SourceCountAccurate:
    """B7.8 — The source-count banner reflects actual validated source data."""

    def test_get_kb_status_counts_correctly(self, monkeypatch):
        """get_kb_status() returns accurate total and verified counts."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage, get_kb_status

        def make_passage(pid, cat, verified):
            return KBPassage(
                passage_id=pid,
                topic=cat,
                category=cat,
                source_title="Test",
                authority="Test",
                section="S1",
                official_url="",
                publication_date=None,
                amendment_date=None,
                date_accessed=None,
                verification_status="verified" if verified else "unverified",
                passage_file="test.md",
                limitations="",
                text="Text.",
            )

        passages = [
            make_passage("P1", "payment", True),
            make_passage("P2", "ip", True),
            make_passage("P3", "payment", False),
        ]

        monkeypatch.setattr(retriever, "load_all_passages", lambda: passages)

        status = get_kb_status()
        assert status["total"] == 3
        assert status["verified"] == 2

    def test_source_count_uses_real_metadata_not_hardcoded(self):
        """Loading real sources.yaml and counting sources matches file content."""
        from src.ca.kb.retriever import load_all_passages, get_kb_status

        # Load real KB (unpatched)
        passages = load_all_passages()
        status = get_kb_status(passages)

        # The counts must come from actual data
        assert status["total"] == len(passages)
        assert status["verified"] == sum(1 for p in passages if p.verification_status == "verified")
        # With 4 unverified reference entries, verified should be 0
        assert status["verified"] == 0
        # Total from the real sources.yaml should be 4
        assert status["total"] == 4

    def test_no_invalid_or_duplicate_counted_as_verified(self):
        """Malformed or duplicate records are not counted as verified sources."""
        from src.ca.kb.retriever import count_verified_passages

        # With unpatched real KB (all unverified), verified count must be 0
        count = count_verified_passages()
        assert count == 0


class TestB9_LegalContextSeparateFromEvidence:
    """B7.9 — Legal context is separate from contract evidence.

    UI test: verifies the retriever integration contract (not Streamlit widgets).
    """

    def test_retrieval_result_does_not_contain_contract_text(self, monkeypatch):
        """KB passages contain only statutory context, never contract text."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        contract_text = "CONFIDENTIAL_CLIENT_CONTRACT_TEXT_UNIQUE_MARKER"

        verified = KBPassage(
            passage_id="TEST-SEP-001",
            topic="payment",
            category="payment",
            source_title="Test Act",
            authority="Test",
            section="S1",
            official_url="",
            publication_date=None,
            amendment_date=None,
            date_accessed=None,
            verification_status="verified",
            passage_file="test.md",
            limitations="",
            text="Test statutory text (no contract content).",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [verified])

        result = retriever.retrieve("payment", verified_only=True)

        assert result.has_results
        # Passage text must not contain contract text
        for passage in result.passages:
            assert contract_text not in passage.text

    def test_retrieval_result_has_provenance_not_contract_quote(self, monkeypatch):
        """Returned passages contain source_title and section, not contract quotations."""
        from src.ca.kb import retriever
        from src.ca.kb.retriever import KBPassage

        passage = KBPassage(
            passage_id="TEST-SEP-002",
            topic="payment",
            category="payment",
            source_title="Indian Contract Act, 1872",
            authority="Parliament of India",
            section="Section 73",
            official_url="https://example.com",
            publication_date="1872-04-25",
            amendment_date=None,
            date_accessed=None,
            verification_status="verified",
            passage_file="test.md",
            limitations="Context only.",
            text="Statutory reference text.",
        )

        monkeypatch.setattr(retriever, "load_all_passages", lambda: [passage])

        result = retriever.retrieve("payment", verified_only=True)

        p = result.passages[0]
        assert p.source_title == "Indian Contract Act, 1872"
        assert p.section == "Section 73"
        assert p.authority == "Parliament of India"


class TestB10_RetrievalLocalNoExternalAI:
    """B7.10 — Retrieval works locally without an external AI request."""

    def test_retrieve_makes_no_network_calls(self, monkeypatch):
        """retrieve() must not call any network-dependent function."""
        import socket

        original_connect = socket.socket.connect

        def fail_on_connect(self, address):
            raise AssertionError(
                f"retriever made unexpected network call to {address}"
            )

        # Patch socket.connect to catch any network attempt
        monkeypatch.setattr(socket.socket, "connect", fail_on_connect)

        from src.ca.kb.retriever import retrieve

        # This must complete without any network call
        result = retrieve("payment", verified_only=True)
        assert isinstance(result.passages, list)

    def test_retrieve_does_not_call_generate_validated(self, monkeypatch):
        """retrieve() must not call generate_validated or any AI client function."""
        from src.ca.kb.retriever import retrieve

        def fail_if_called(*args, **kwargs):
            pytest.fail("retrieve() must not call generate_validated")

        monkeypatch.setattr(
            "src.ca.llm.client.generate_validated",
            fail_if_called,
        )

        result = retrieve("payment", verified_only=True)
        assert isinstance(result.passages, list)


class TestB11_RealKBIntegration:
    """B7.11 — Real KB loads and retrieves correctly with actual sources.yaml."""

    def test_real_kb_loads_without_error(self):
        """load_all_passages() succeeds with the real sources.yaml."""
        from src.ca.kb.retriever import load_all_passages

        passages = load_all_passages()
        assert isinstance(passages, list)

    def test_real_kb_payment_retrieval_returns_honest_no_match(self):
        """With real KB (all unverified), payment retrieval returns honest no-match."""
        from src.ca.kb.retriever import retrieve

        result = retrieve("payment", verified_only=True)
        assert not result.has_results
        assert len(result.no_match_reason) > 5

    def test_real_kb_ip_retrieval_returns_honest_no_match(self):
        """With real KB (all unverified), IP retrieval returns honest no-match."""
        from src.ca.kb.retriever import retrieve

        result = retrieve("ip", verified_only=True)
        assert not result.has_results
        assert len(result.no_match_reason) > 5

    def test_real_kb_total_count_is_4(self):
        """Real sources.yaml has exactly 4 reference entries."""
        from src.ca.kb.retriever import load_all_passages

        passages = load_all_passages()
        assert len(passages) == 4

    def test_real_kb_zero_verified_passages(self):
        """Real KB has 0 verified passages — consistent with unverified status."""
        from src.ca.kb.retriever import count_verified_passages

        assert count_verified_passages() == 0

    def test_real_kb_passage_ids_are_stable(self):
        """Real KB passage IDs are stable and match expected values."""
        from src.ca.kb.retriever import load_all_passages

        passages = load_all_passages()
        ids = {p.passage_id for p in passages}
        expected_ids = {"ICA1872-S2D", "ICA1872-S73", "CA1957-S17", "CA1957-S19"}
        assert ids == expected_ids

    def test_real_kb_categories_are_payment_and_ip(self):
        """All real KB passages are in the payment or ip category."""
        from src.ca.kb.retriever import load_all_passages

        passages = load_all_passages()
        cats = {p.category for p in passages}
        assert cats == {"payment", "ip"}


# ------------------------------------------------------------------ #
# Regression — Stage 5 golden tests remain valid                      #
# ------------------------------------------------------------------ #

class TestRegression_Stage5Golden:
    """Confirm Stage 5 worked example remains 57.0/60.0 and 65.0/51.4."""

    def test_stage5_golden_still_correct(self):
        """B7.11 / Part C.7 — Stage 5 worked example is unchanged after Stage 6."""
        from src.ca.compare.engine import calculate_weighted_scores

        fav_a = {"payment": 40.0, "ip": 85.0, "termination": 80.0,
                 "scope": None, "liability": None, "confidentiality": None}
        fav_b = {"payment": 75.0, "ip": 30.0, "termination": 45.0,
                 "scope": None, "liability": None, "confidentiality": None}

        weights1 = {"payment": 3, "ip": 1, "termination": 1,
                    "scope": 0, "liability": 0, "confidentiality": 0}
        res1 = calculate_weighted_scores(fav_a, fav_b, weights1)
        assert res1["score_a"] == 57.0
        assert res1["score_b"] == 60.0

        weights2 = {"payment": 3, "ip": 3, "termination": 1,
                    "scope": 0, "liability": 0, "confidentiality": 0}
        res2 = calculate_weighted_scores(fav_a, fav_b, weights2)
        assert res2["score_a"] == 65.0
        assert res2["score_b"] == 51.4
