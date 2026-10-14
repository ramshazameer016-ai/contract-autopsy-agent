
"""Plain-language contract summary with evidence verification and safe fallback."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Iterator

from src.ca.llm.client import LLMUnavailable, generate_validated
from src.ca.llm.validator import PlainSummary, SummaryItem
from src.ca.translation.translate import (
    SUPPORTED_LANGUAGES,
    translate_category,
    translate_explanation,
    translate_disclaimer,
    is_devanagari,
    is_telugu,
    is_tamil,
)

logger = logging.getLogger(__name__)

PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "llm"
    / "prompts"
    / "summary_v1.txt"
)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a dictionary or a model object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_page(clause: Any) -> int | None:
    """Support UI dictionaries and Clause schema objects."""
    page = _get_value(clause, "page", None)
    if page is None:
        page = _get_value(clause, "page_start", None)
    return page


def _normalise_text(value: Any) -> str:
    """Normalise whitespace for comparisons."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _source_text(clauses: list[Any]) -> str:
    """Build source text for the LLM prompt."""
    blocks: list[str] = []
    for clause in clauses:
        text = str(_get_value(clause, "text", "") or "").strip()
        if not text:
            continue
        clause_id = _get_value(clause, "clause_id", "UNKNOWN")
        page = _get_page(clause)
        blocks.append(f"[CLAUSE_ID: {clause_id}; PAGE: {page}]\n{text}")
    return "\n\n".join(blocks)


def _parse_fields(line: str) -> dict[str, str]:
    """Parse pipe-separated FACT/FINDING/EVIDENCE fields."""
    fields: dict[str, str] = {}
    parts = line.split(" | ")
    for part in parts[1:]:
        key, separator, value = part.partition(": ")
        if separator:
            fields[key.strip().casefold()] = value.strip()
    return fields


def _iter_records(text: str) -> Iterator[str]:
    """Yield individual FACT, FINDING and EVIDENCE records."""
    if not text:
        return

    marker_pattern = re.compile(
        r"(?<!\S)(?:FACT|FINDING|EVIDENCE)\s*\|",
        flags=re.IGNORECASE,
    )
    matches = list(marker_pattern.finditer(text))

    if not matches:
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield line
        return

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        record = text[start:end].strip()
        if record:
            yield record


def _find_source_clause(
    clauses: list[Any],
    clause_id: str | None,
) -> Any | None:
    """Find a source clause by its exact ID."""
    if not clause_id:
        return None

    requested_id = str(clause_id).strip()
    for clause in clauses:
        current_id = str(_get_value(clause, "clause_id", "") or "").strip()
        if current_id == requested_id:
            return clause
    return None


def _is_heading_only_quote(quote: str, clause: Any) -> bool:
    """Reject evidence quotes that contain only a clause heading."""
    quote_normalised = _normalise_text(quote).casefold()
    if not quote_normalised:
        return True

    heading = _normalise_text(_get_value(clause, "heading", "")).casefold()
    number = _normalise_text(_get_value(clause, "number", "")).casefold()

    heading_variants: set[str] = set()
    if heading:
        heading_variants.add(heading)
    if heading and number:
        heading_variants.update({
            f"{number} {heading}",
            f"{number}. {heading}",
            f"{number}: {heading}",
        })

    if quote_normalised in heading_variants:
        return True

    return bool(re.fullmatch(
        r"(?:section\s+)?\d+(?:\.\d+)*\.?\s+[A-Za-z][A-Za-z &()/,\-]*",
        quote_normalised,
        flags=re.IGNORECASE,
    ))


def _normalise_category(value: Any) -> str:
    """Normalise category labels for matching."""
    return str(value or "").strip().casefold()


def _template_summary(
    clauses: list[Any],
    language: str,
    facts_and_findings: str = "",
) -> PlainSummary:
    """Create a local summary without calling an AI provider."""
    language_name = SUPPORTED_LANGUAGES.get(
        str(language).strip().casefold(),
        "English",
    )

    source_clauses = [
        clause
        for clause in clauses
        if str(_get_value(clause, "text", "") or "").strip()
    ]

    evidence_by_category: dict[str, list[tuple[str, str]]] = {}
    finding_rows: list[dict[str, str]] = []

    for raw_record in _iter_records(facts_and_findings or ""):
        line = raw_record.strip()
        if not line:
            continue

        record_type = line.split("|", 1)[0].strip().casefold()

        if record_type == "evidence":
            fields = _parse_fields(line)
            category = fields.get("category", "").strip()
            clause_id = (
                fields.get("clause_id")
                or fields.get("clause id")
                or fields.get("clause")
                or ""
            ).strip()
            quote = fields.get("quote", "").strip()

            if category and clause_id and quote:
                evidence_by_category.setdefault(
                    _normalise_category(category), []
                ).append((clause_id, quote))

        elif record_type == "finding":
            fields = _parse_fields(line)
            if not fields.get("status"):
                fields["status"] = fields.get("evidence status", "")
            if fields.get("category"):
                finding_rows.append(fields)

    grouped: dict[str, list[dict[str, str]]] = {}
    for finding in finding_rows:
        category = finding.get("category", "").strip()
        if category:
            grouped.setdefault(category, []).append(finding)

    items: list[SummaryItem] = []

    for category, findings in grouped.items():
        statuses = [
            finding.get("status", "").strip().casefold()
            for finding in findings
        ]

        if any(status in {"absent", "missing", "not found"} for status in statuses):
            status_label = "Missing"
        elif any(
            status in {"ambiguous", "conflicting", "insufficient", "unclear"}
            for status in statuses
        ):
            status_label = "Unclear"
        elif any(
            status in {"present", "explicit", "clear", "stated"}
            for status in statuses
        ):
            status_label = "Stated"
        else:
            status_label = "Unclear"

        reasons = [
            finding.get("reason", "").strip()
            for finding in findings
            if finding.get("reason", "").strip()
        ]

        reason = (
            " ".join(dict.fromkeys(reasons))
            if reasons
            else f"The available analysis marks {category} as {status_label.lower()}."
        )

        verified_candidates: list[tuple[Any, str]] = []
        for evidence_clause_id, candidate_quote in evidence_by_category.get(
            _normalise_category(category), []
        ):
            source_clause = _find_source_clause(
                source_clauses, evidence_clause_id
            )
            if source_clause is None:
                continue

            source_text = str(_get_value(source_clause, "text", "") or "")
            if candidate_quote not in source_text:
                continue
            if _is_heading_only_quote(candidate_quote, source_clause):
                continue

            verified_candidates.append((source_clause, candidate_quote))

        quote = ""
        selected_clause_id = None
        selected_page = None

        if verified_candidates:
            selected_clause, quote = max(
                verified_candidates,
                key=lambda candidate: len(candidate[1]),
            )
            selected_clause_id = _get_value(
                selected_clause, "clause_id", None
            )
            selected_page = _get_page(selected_clause)

        translated_cat = (
            category if language_name == "English"
            else translate_category(category, language_name)
        )
        translated_explanation = translate_explanation(
            reason, language_name, category=category, status=status_label
        )

        items.append(
            SummaryItem(
                category=translated_cat,
                status=status_label,
                explanation=translated_explanation,
                evidence_quote=quote,
                clause_id=selected_clause_id,
                page=selected_page,
            )
        )

    if not items:
        if source_clauses:
            first_clause = source_clauses[0]
            first_text = str(
                _get_value(first_clause, "text", "") or ""
            ).strip()

            preview_cat = (
                "Extracted clause" if language_name == "English"
                else translate_category("extracted clause", language_name)
            )
            preview_explanation = translate_explanation(
                "Source-text preview only. Verify the original contract before relying on this text.",
                language_name,
            )

            items.append(
                SummaryItem(
                    category=preview_cat,
                    status="Stated",
                    explanation=preview_explanation,
                    evidence_quote=first_text[:1200],
                    clause_id=_get_value(first_clause, "clause_id", None),
                    page=_get_page(first_clause),
                )
            )
        else:
            missing_cat = (
                "Contract text" if language_name == "English"
                else translate_category("contract text", language_name)
            )
            missing_explanation = translate_explanation(
                "No readable contract text was available to generate a summary.",
                language_name,
            )
            items.append(
                SummaryItem(
                    category=missing_cat,
                    status="Missing",
                    explanation=missing_explanation,
                    evidence_quote="",
                    clause_id=None,
                    page=None,
                )
            )

    return PlainSummary(
        language=language_name,
        generated_by="template",
        items=items,
        prompt_version="summary_v1",
    )


def _fallback(
    clauses: list[Any],
    language: str,
    facts_and_findings: str,
    reason: str,
) -> PlainSummary:
    """Log a safe diagnostic reason and return the local template."""
    # Log only the reason, never prompts, source text, or API credentials.
    logger.warning("AI summary fallback: %s", reason)
    return _template_summary(
        clauses=clauses,
        language=language,
        facts_and_findings=facts_and_findings,
    )


def summarize_contract(
    clauses: list[Any],
    facts_and_findings: str,
    language: str = "English",
    consent_given: bool = False,
) -> PlainSummary:
    """
    Use validated AI output when consent is given.
    Fall back safely when AI is unavailable or validation fails.
    """
    language_name = SUPPORTED_LANGUAGES.get(
        str(language).strip().casefold(),
        "English",
    )

    if not consent_given:
        return _fallback(
            clauses, language_name, facts_and_findings,
            "Consent not given; no external AI request was made.",
        )

    try:
        prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

        lang_instructions = {
            "English": (
                "Write all explanations in simple, clear English suitable "
                "for a freelancer without legal training."
            ),
            "Hindi": (
                "Write explanations in natural Hindi using Devanagari script. "
                "Keep evidence_quote exactly as written in the source contract."
            ),
            "Telugu": (
                "Write explanations in natural Telugu using Telugu script. "
                "Keep evidence_quote exactly as written in the source contract."
            ),
            "Tamil": (
                "Write explanations in natural Tamil using Tamil script. "
                "Keep evidence_quote exactly as written in the source contract."
            ),
        }
        lang_instruction = (
            "\n\nCRITICAL LANGUAGE INSTRUCTION:\n"
            + lang_instructions[language_name]
            + "\nThe status must be Stated, Missing, Unclear, or Inferred."
        )

        prompt = (
            prompt_template
            + "\n\nRequested language: "
            + language_name
            + lang_instruction
            + "\n\nUse only the supplied facts, findings, and contract source. "
            "Do not invent terms, facts, citations, or evidence. If information "
            "is missing or unclear, state that. Evidence quotes must be exact "
            "substrings of their identified source clauses. Do not use headings "
            "alone as evidence of substantive contract terms.\n\n"
            "FACTS_AND_FINDINGS:\n"
            + (facts_and_findings or "(none supplied)")
            + "\n\nCONTRACT_SOURCE:\n<contract_data>\n"
            + _source_text(clauses)
            + "\n</contract_data>\n"
        )

        result = generate_validated(
            prompt,
            PlainSummary,
            prompt_version="summary_v1",
        )

    except (LLMUnavailable, OSError, ValueError, TypeError) as exc:
        return _fallback(
            clauses, language_name, facts_and_findings,
            f"AI request or response validation failed ({type(exc).__name__}: {exc}).",
        )
    except Exception as exc:
        # Do not log exception text; provider exceptions can contain sensitive data.
        return _fallback(
            clauses, language_name, facts_and_findings,
            f"Unexpected AI error ({type(exc).__name__}).",
        )

    if str(_get_value(result, "language", "")).casefold() != language_name.casefold():
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI response language did not match the requested language.",
        )

    if _get_value(result, "generated_by", "") != "llm":
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI response was not marked as LLM-generated.",
        )

    items = _get_value(result, "items", []) or []
    if not items:
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI response contained no summary items.",
        )

    if language_name == "Hindi" and not any(
        is_devanagari(_get_value(item, "explanation", "")) for item in items
    ):
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI explanations did not contain Hindi script.",
        )

    if language_name == "Telugu" and not any(
        is_telugu(_get_value(item, "explanation", "")) for item in items
    ):
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI explanations did not contain Telugu script.",
        )

    if language_name == "Tamil" and not any(
        is_tamil(_get_value(item, "explanation", "")) for item in items
    ):
        return _fallback(
            clauses, language_name, facts_and_findings,
            "AI explanations did not contain Tamil script.",
        )

    
    for item in items:
        evidence_quote = str(
            _get_value(item, "evidence_quote", "") or ""
        )
        if not evidence_quote:
            continue

        item_clause_id = _get_value(item, "clause_id", None)
        source_clause = _find_source_clause(
            clauses, str(item_clause_id or "")
        )
        if source_clause is None:
            return _fallback(
                clauses,
                language_name,
                facts_and_findings,
                "AI evidence referenced an unknown clause ID.",
            )

        source_text = str(
            _get_value(source_clause, "text", "") or ""
        )

        if evidence_quote not in source_text:
            logger.warning(
                "Summary evidence mismatch: clause_id=%r, "
                "quote_length=%d, source_length=%d",
                item_clause_id,
                len(evidence_quote),
                len(source_text),
            )
            return _fallback(
                clauses,
                language_name,
                facts_and_findings,
                "AI evidence quote did not exactly match its source clause.",
            )

        if _is_heading_only_quote(evidence_quote, source_clause):
            return _fallback(
                clauses,
                language_name,
                facts_and_findings,
                "AI evidence quote contained only a clause heading.",
            )

        actual_page = _get_page(source_clause)
        if isinstance(item, dict):
            item["page"] = actual_page
        else:
            try:
                item.page = actual_page
            except (AttributeError, TypeError):
                pass

    return result
