"""Legal Knowledge Base retriever — Stage 6.

Implements a transparent, deterministic, keyword/category-based retriever
that returns provenance-tracked legal passages for contract findings.

Design principles (Architecture Stage 6):
- No external AI service is called for retrieval.
- Only passages with verification_status == "verified" are returned as
  verified legal context.
- Unverified passages are loaded but excluded from verified results.
- Provenance metadata is preserved on every returned passage.
- An empty result is returned honestly rather than fabricating a match.
- Malformed or missing source metadata is handled safely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths — resolved relative to this file so the module works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_KB_ROOT = _REPO_ROOT / "knowledge_base"
_SOURCES_FILE = _KB_ROOT / "sources.yaml"
_PASSAGES_DIR = _KB_ROOT / "passages"


@dataclass(frozen=True)
class KBPassage:
    """A single knowledge-base passage with full provenance metadata."""

    passage_id: str
    topic: str
    category: str
    source_title: str
    authority: str
    section: str
    official_url: str
    publication_date: str | None
    amendment_date: str | None
    date_accessed: str | None
    verification_status: str        # "verified" | "unverified"
    passage_file: str
    limitations: str
    text: str                       # content of the passage markdown file


@dataclass
class RetrievalResult:
    """Result of a KB retrieval query."""

    query_category: str
    query_topic: str | None
    passages: list[KBPassage] = field(default_factory=list)
    no_match_reason: str = ""

    @property
    def has_results(self) -> bool:
        return bool(self.passages)


def _load_sources() -> list[dict[str, Any]]:
    """Load and parse sources.yaml.  Returns an empty list on any error."""
    if not _SOURCES_FILE.exists():
        logger.warning("knowledge_base/sources.yaml not found at %s", _SOURCES_FILE)
        return []
    try:
        import yaml  # PyYAML is already installed via requirements
        with _SOURCES_FILE.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            logger.warning("sources.yaml: top-level value is not a mapping")
            return []
        passages = data.get("passages", [])
        if not isinstance(passages, list):
            logger.warning("sources.yaml: 'passages' is not a list")
            return []
        return passages
    except Exception as exc:
        logger.warning("sources.yaml could not be loaded: %s", exc)
        return []


def _read_passage_text(passage_file: str) -> str:
    """Read the markdown text for a passage file.  Returns empty string on error."""
    if not passage_file:
        return ""
    path = _PASSAGES_DIR / passage_file
    if not path.exists():
        logger.warning("Passage file not found: %s", path)
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Could not read passage file %s: %s", path, exc)
        return ""


def _parse_passage(raw: Any) -> KBPassage | None:
    """Parse a raw dict from sources.yaml into a KBPassage.

    Returns None for malformed entries (with a warning).
    """
    if not isinstance(raw, dict):
        logger.warning("Malformed passage entry (not a dict): %r", raw)
        return None

    required = ("passage_id", "category", "verification_status", "passage_file")
    for key in required:
        if not raw.get(key):
            logger.warning("Passage entry missing required field '%s': %r", key, raw)
            return None

    passage_file = str(raw["passage_file"])
    text = _read_passage_text(passage_file)

    return KBPassage(
        passage_id=str(raw["passage_id"]),
        topic=str(raw.get("topic") or ""),
        category=str(raw.get("category") or ""),
        source_title=str(raw.get("source_title") or ""),
        authority=str(raw.get("authority") or ""),
        section=str(raw.get("section") or ""),
        official_url=str(raw.get("official_url") or ""),
        publication_date=str(raw["publication_date"]) if raw.get("publication_date") else None,
        amendment_date=str(raw["amendment_date"]) if raw.get("amendment_date") else None,
        date_accessed=str(raw["date_accessed"]) if raw.get("date_accessed") else None,
        verification_status=str(raw["verification_status"]),
        passage_file=passage_file,
        limitations=str(raw.get("limitations") or ""),
        text=text,
    )


def load_all_passages() -> list[KBPassage]:
    """Load all passages (verified and unverified) from sources.yaml.

    Skips malformed entries.  Always returns a list (possibly empty).
    """
    raw_list = _load_sources()
    passages: list[KBPassage] = []
    for raw in raw_list:
        parsed = _parse_passage(raw)
        if parsed is not None:
            passages.append(parsed)
    return passages


def count_verified_passages(passages: list[KBPassage] | None = None) -> int:
    """Return the number of passages with verification_status == 'verified'."""
    if passages is None:
        passages = load_all_passages()
    return sum(1 for p in passages if p.verification_status == "verified")


def count_total_passages(passages: list[KBPassage] | None = None) -> int:
    """Return the total number of successfully loaded passages."""
    if passages is None:
        passages = load_all_passages()
    return len(passages)


def retrieve(
    category: str,
    topic: str | None = None,
    *,
    verified_only: bool = True,
) -> RetrievalResult:
    """Retrieve relevant passages for a given finding category and optional topic.

    Algorithm (simple keyword/category matching):
    1. Filter passages by category (exact match, case-insensitive).
    2. If topic is given, further prefer passages whose topic matches.
    3. If verified_only=True (default), exclude unverified passages.
    4. If no match, return an honest empty result.

    No external service is called.  Retrieval is deterministic.

    Args:
        category:      Risk finding category (e.g. "payment", "ip").
        topic:         Optional sub-topic keyword (e.g. "ip_assignment").
        verified_only: If True, exclude unverified passages from results.

    Returns:
        RetrievalResult with matching passages (possibly empty).
    """
    if not category or not isinstance(category, str):
        return RetrievalResult(
            query_category="",
            query_topic=topic,
            passages=[],
            no_match_reason="Empty or invalid category supplied.",
        )

    norm_category = category.strip().casefold()
    norm_topic = topic.strip().casefold() if topic else None

    all_passages = load_all_passages()

    # Step 1: filter by category
    category_matches = [
        p for p in all_passages
        if p.category.strip().casefold() == norm_category
    ]

    # Step 2: filter by verification status
    if verified_only:
        category_matches = [
            p for p in category_matches
            if p.verification_status == "verified"
        ]

    if not category_matches:
        if verified_only:
            return RetrievalResult(
                query_category=category,
                query_topic=topic,
                passages=[],
                no_match_reason=(
                    f"No verified legal context found for category '{category}'. "
                    "The knowledge base has reference entries for this category "
                    "but they have not yet been verified against official sources."
                ),
            )
        return RetrievalResult(
            query_category=category,
            query_topic=topic,
            passages=[],
            no_match_reason=(
                f"No passages found for category '{category}'."
            ),
        )

    # Step 3: if topic given, prefer topic-matching passages
    if norm_topic:
        topic_matches = [
            p for p in category_matches
            if norm_topic in p.topic.strip().casefold()
        ]
        # Only narrow if we got matches; otherwise return all category matches.
        if topic_matches:
            category_matches = topic_matches

    return RetrievalResult(
        query_category=category,
        query_topic=topic,
        passages=category_matches,
        no_match_reason="",
    )


def get_kb_status(passages: list[KBPassage] | None = None) -> dict[str, int]:
    """Return a dict with total and verified passage counts from actual metadata.

    Used by the UI to display an honest source count.

    Args:
        passages: Pre-loaded passage list.  If None, sources.yaml is loaded.
    """
    if passages is None:
        passages = load_all_passages()
    return {
        "total": count_total_passages(passages),
        "verified": count_verified_passages(passages),
    }
