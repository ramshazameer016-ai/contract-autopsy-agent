
"""Translation helpers for Contract Autopsy Agent."""

from __future__ import annotations

from typing import Any

from src.ca.generation.summary import (
    SUPPORTED_LANGUAGES,
    _template_summary,
)
from src.ca.llm.validator import PlainSummary


def supported_languages() -> tuple[str, ...]:
    """Return the languages supported by the current summary pipeline."""
    return tuple(SUPPORTED_LANGUAGES.values())


def offline_summary(
    clauses: list[Any],
    language: str = "English",
) -> PlainSummary:
    """Create a local template summary without calling an AI provider.

    Accepts language names regardless of capitalization, for example
    'Hindi' or 'hindi'. Evidence quotations remain in their original
    source language and must not be translated or altered.
    """
    language_key = str(language).strip().casefold()

    if language_key not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported language. Choose one of: "
            + ", ".join(SUPPORTED_LANGUAGES.values())
        )

    language_name = SUPPORTED_LANGUAGES[language_key]

    return _template_summary(
        clauses,
        language_name,
    )
