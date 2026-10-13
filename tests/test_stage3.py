
import pytest
from pydantic import ValidationError

from src.ca.llm.validator import PlainSummary, SummaryItem
from src.ca.generation.summary import summarize_contract
from src.ca.llm import client
import pytest

from src.ca.generation.summary import _template_summary
from src.ca.llm.client import LLMUnavailable


def test_summary_schema_rejects_invalid_status():
    with pytest.raises(ValidationError):
        SummaryItem(
            category="Payment",
            status="Risky",
            explanation="Payment terms need review.",
        )


def test_summary_schema_accepts_valid_status():
    item = SummaryItem(
        category="Payment",
        status="Stated",
        explanation="Payment is due within 30 days.",
        evidence_quote="Payment is due within 30 days.",
    )
    assert item.status == "Stated"


def test_no_consent_uses_template_without_calling_ai(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("AI gateway must not be called without consent")

    monkeypatch.setattr(
        "src.ca.generation.summary.generate_validated",
        fail_if_called,
    )

    clauses = [
        {
            "clause_id": "C1",
            "page": 1,
            "text": "Payment is due within 30 days.",
        }
    ]

    result = summarize_contract(
        clauses,
        facts_and_findings="Payment: stated",
        language="English",
        consent_given=False,
    )

    assert result.generated_by == "template"
    assert result.language == "English"


def test_offline_gateway_makes_no_provider_call(monkeypatch):
    monkeypatch.setattr(client, "is_offline", lambda: True)
    monkeypatch.setattr(client, "GEMINI_API_KEY", "")

    def fail_if_called(*args, **kwargs):
        pytest.fail("Provider must not be called in offline mode")

    monkeypatch.setattr(client, "_call_gemini", fail_if_called)

    with pytest.raises(client.LLMUnavailable):
        client.generate_validated(
            "Test prompt",
            PlainSummary,
            prompt_version="test",
        )


def test_invalid_json_is_retried_three_times(monkeypatch):
    monkeypatch.setattr(client, "is_offline", lambda: False)
    monkeypatch.setattr(client, "GEMINI_API_KEY", "test-key")

    calls = {"count": 0}

    def invalid_json(prompt):
        calls["count"] += 1
        return "not valid JSON"

    monkeypatch.setattr(client, "_call_gemini", invalid_json)
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    client.clear_session_cache()

    with pytest.raises(client.LLMUnavailable):
        client.generate_validated(
            "Test prompt",
            PlainSummary,
            prompt_version="invalid-json-test",
        )

    assert calls["count"] == 3


def test_offline_summary_uses_selected_language():
    clauses = [
        {
            "clause_id": "A-001",
            "text": "Payment is due within 30 days.",
        }
    ]

    expected_phrases = {
        "English": "Source-text preview only.",
        "Hindi": "Source-text preview only.",
        "Telugu": "Source-text preview only.",
        "Tamil": "Source-text preview only.",
    }

    for language, expected in expected_phrases.items():
        result = _template_summary(clauses, language)

        assert result.language == language
        assert result.generated_by == "template"
        assert result.items[0].explanation.startswith(expected)


def test_offline_summary_does_not_call_ai(monkeypatch):
    import src.ca.generation.summary as summary_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AI must not be called for offline fallback")

    monkeypatch.setattr(summary_module, "generate_validated", fail_if_called)

    for language in ("English", "Hindi", "Telugu", "Tamil"):
        result = summary_module.summarize_contract(
            clauses=[{"clause_id": "A-001", "text": "Payment is due within 30 days."}],
            facts_and_findings="",
            language=language,
            consent_given=False,
        )

        assert result.generated_by == "template"
        assert result.language == language


def test_template_summary_preserves_exact_evidence_quote():
    clause_text = "Payment is due within 30 days."
    clauses = [{"clause_id": "A-001", "text": clause_text}]

    findings = (
        "FINDING | Category: Payment | Status: Stated | "
        "Reason: A payment deadline is stated.\n"
        "EVIDENCE | Category: Payment | Clause: A-001 | "
        f"Quote: {clause_text}"
    )

    result = _template_summary(clauses, "English", findings)

    payment_items = [
        item for item in result.items
        if item.category == "Payment"
    ]

    assert payment_items
    assert payment_items[0].evidence_quote == clause_text
    assert payment_items[0].clause_id == "A-001"
