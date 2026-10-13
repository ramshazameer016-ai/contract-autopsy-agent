"""Tests for Stage 3 multilingual plain-language contract summary generator."""

import pytest
from pydantic import ValidationError

from src.ca.llm.validator import PlainSummary, SummaryItem
from src.ca.generation.summary import summarize_contract, _template_summary
from src.ca.llm import client
from src.ca.llm.client import LLMUnavailable
from src.ca.translation.translate import (
    supported_languages,
    offline_summary,
    is_devanagari,
    is_telugu,
    is_tamil,
    script_percentage,
)


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
    """Verify that selecting each language produces genuine target-script text."""
    clauses = [
        {
            "clause_id": "A-001",
            "text": "Payment is due within 30 days.",
        }
    ]

    expected_phrases = {
        "English": "Source-text preview only.",
        "Hindi": "केवल स्रोत-पाठ पूर्वावलोकन",
        "Telugu": "మూల పాఠ్య ప్రివ్యూ మాత్రమే",
        "Tamil": "மூல உரை முன்னோட்டம் மட்டுமே",
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


# ---------------------------------------------------------------------
# Stage 3 Multilingual Fix: Genuine target script & status preservation
# ---------------------------------------------------------------------

def test_hindi_offline_summary_devanagari_script_and_facts():
    """Verify Hindi offline summary contains genuine Devanagari script and preserves facts."""
    clause_text = "Payment is due 90 days after invoice. You would finance the client for that long."
    clauses = [{"clause_id": "A-001", "text": clause_text}]

    findings = (
        "FINDING | Category: payment | Status: Stated | "
        "Reason: Payment is due 90 days after invoice. You would finance the client for that long.\n"
        "EVIDENCE | Category: payment | Clause: A-001 | "
        f"Quote: {clause_text}"
    )

    result = _template_summary(clauses, "Hindi", findings)

    assert result.language == "Hindi"
    assert result.generated_by == "template"
    assert len(result.items) >= 1

    item = result.items[0]
    assert is_devanagari(item.explanation)
    assert script_percentage(item.explanation, "Hindi") >= 0.50
    # Important contract value '90' is preserved
    assert "90" in item.explanation
    # Evidence quote is verbatim original text, never translated
    assert item.evidence_quote == clause_text
    assert item.clause_id == "A-001"


def test_telugu_offline_summary_script_and_facts():
    """Verify Telugu offline summary contains genuine Telugu script and preserves facts."""
    clause_text = "Payment is due 90 days after invoice. You would finance the client for that long."
    clauses = [{"clause_id": "A-001", "text": clause_text}]

    findings = (
        "FINDING | Category: payment | Status: Stated | "
        "Reason: Payment is due 90 days after invoice. You would finance the client for that long.\n"
        "EVIDENCE | Category: payment | Clause: A-001 | "
        f"Quote: {clause_text}"
    )

    result = _template_summary(clauses, "Telugu", findings)

    assert result.language == "Telugu"
    assert result.generated_by == "template"
    assert len(result.items) >= 1

    item = result.items[0]
    assert is_telugu(item.explanation)
    assert script_percentage(item.explanation, "Telugu") >= 0.50
    assert "90" in item.explanation
    assert item.evidence_quote == clause_text


def test_tamil_offline_summary_script_and_facts():
    """Verify Tamil offline summary contains genuine Tamil script and preserves facts."""
    clause_text = "Payment is due 90 days after invoice. You would finance the client for that long."
    clauses = [{"clause_id": "A-001", "text": clause_text}]

    findings = (
        "FINDING | Category: payment | Status: Stated | "
        "Reason: Payment is due 90 days after invoice. You would finance the client for that long.\n"
        "EVIDENCE | Category: payment | Clause: A-001 | "
        f"Quote: {clause_text}"
    )

    result = _template_summary(clauses, "Tamil", findings)

    assert result.language == "Tamil"
    assert result.generated_by == "template"
    assert len(result.items) >= 1

    item = result.items[0]
    assert is_tamil(item.explanation)
    assert script_percentage(item.explanation, "Tamil") >= 0.50
    assert "90" in item.explanation
    assert item.evidence_quote == clause_text


def test_each_language_preserves_stated_missing_unclear_status():
    """Verify Stated, Missing, and Unclear statuses are preserved across all four languages."""
    clauses = [
        {"clause_id": "A-001", "text": "All work product vests perpetually in the Client."},
        {"clause_id": "A-002", "text": "The client may pay a reasonable fee at a reasonable time."},
    ]

    findings = (
        "FINDING | Category: payment | Status: Unclear | "
        "Reason: No clear payment deadline (for example 'reasonable time').\n"
        "FINDING | Category: ip | Status: Stated | "
        "Reason: All work is assigned with no carve-out for your pre-existing tools or material.\n"
        "FINDING | Category: termination | Status: Missing | "
        "Reason: No termination clause found. Exit rights and notice are undefined.\n"
    )

    for language in ("English", "Hindi", "Telugu", "Tamil"):
        res = _template_summary(clauses, language, findings)
        status_set = {it.status for it in res.items}

        assert "Stated" in status_set
        assert "Missing" in status_set
        assert "Unclear" in status_set

        # Verify missing term is NEVER reported as Stated
        missing_items = [it for it in res.items if it.status == "Missing"]
        assert len(missing_items) == 1
        assert missing_items[0].evidence_quote == ""

        # Verify that non-English languages have target script in explanation
        for it in res.items:
            if language == "Hindi":
                assert is_devanagari(it.explanation)
            elif language == "Telugu":
                assert is_telugu(it.explanation)
            elif language == "Tamil":
                assert is_tamil(it.explanation)


def test_ai_fallback_when_llm_returns_english_for_hindi(monkeypatch):
    """When LLM returns English despite requested Hindi, it must fallback to offline Hindi template."""
    import src.ca.generation.summary as summary_module

    # Mock LLM returning pure English when Hindi was requested
    def fake_generate_validated(prompt, schema, **kwargs):
        return PlainSummary(
            language="Hindi",
            generated_by="llm",
            items=[
                SummaryItem(
                    category="Payment",
                    status="Stated",
                    explanation="Payment is due within 30 days.",
                    evidence_quote="Payment is due within 30 days.",
                    clause_id="A-001",
                    page=1,
                )
            ],
        )

    monkeypatch.setattr(summary_module, "generate_validated", fake_generate_validated)

    clauses = [{"clause_id": "A-001", "text": "Payment is due within 30 days."}]
    findings = (
        "FINDING | Category: payment | Status: Stated | "
        "Reason: A payment deadline is stated.\n"
        "EVIDENCE | Category: payment | Clause: A-001 | "
        "Quote: Payment is due within 30 days."
    )

    result = summary_module.summarize_contract(
        clauses=clauses,
        facts_and_findings=findings,
        language="Hindi",
        consent_given=True,
    )

    # Must fall back to template because AI output failed Hindi script check
    assert result.generated_by == "template"
    assert is_devanagari(result.items[0].explanation)


def test_ai_fallback_when_llm_raises_exception(monkeypatch):
    """When LLM raises an error, summary generator must safely return local template."""
    import src.ca.generation.summary as summary_module

    def error_llm(*args, **kwargs):
        raise LLMUnavailable("Provider quota exceeded")

    monkeypatch.setattr(summary_module, "generate_validated", error_llm)

    clauses = [{"clause_id": "A-001", "text": "Payment is due within 30 days."}]
    findings = (
        "FINDING | Category: payment | Status: Stated | "
        "Reason: A payment deadline is stated.\n"
        "EVIDENCE | Category: payment | Clause: A-001 | "
        "Quote: Payment is due within 30 days."
    )

    result = summary_module.summarize_contract(
        clauses=clauses,
        facts_and_findings=findings,
        language="Telugu",
        consent_given=True,
    )

    assert result.generated_by == "template"
    assert is_telugu(result.items[0].explanation)


def test_unsupported_language_handling():
    """Verify unsupported languages are rejected in offline_summary and handled safely in summarize."""
    with pytest.raises(ValueError, match="Unsupported language"):
        offline_summary([{"clause_id": "1", "text": "test"}], language="French")

    # summarize_contract defaults safely to English for unknown languages
    res = summarize_contract([{"clause_id": "1", "text": "test"}], "", language="German", consent_given=False)
    assert res.language == "English"


def test_translate_helpers():
    """Verify supported_languages and script detection helper functions."""
    langs = supported_languages()
    assert "English" in langs
    assert "Hindi" in langs
    assert "Telugu" in langs
    assert "Tamil" in langs

    assert is_devanagari("नमस्ते दुनिया")
    assert not is_devanagari("Hello world")

    assert is_telugu("నమస్కారం ప్రపంచం")
    assert not is_telugu("Hello world")

    assert is_tamil("வணக்கம் உலகம்")
    assert not is_tamil("Hello world")

    assert script_percentage("नमस्ते 123", "Hindi") > 0.8
