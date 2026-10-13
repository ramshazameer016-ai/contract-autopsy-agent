"""
Deterministic keyword-based category tagger for contract clauses.
Reads keywords from config/categories.yaml without using any LLM.
"""

from __future__ import annotations

from pathlib import Path
import yaml
import re

from src.ca.schemas import Clause, Category

# Default fallback keywords if YAML cannot be loaded
DEFAULT_CATEGORIES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "categories.yaml"

_CACHED_CONFIG: dict | None = None


def load_categories_config(config_path: Path | str | None = None) -> dict:
    """Load categories and keyword mappings from YAML file."""
    global _CACHED_CONFIG
    path = Path(config_path) if config_path else DEFAULT_CATEGORIES_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _CACHED_CONFIG = data.get("categories", {})
    return _CACHED_CONFIG


def tag_clause(clause: Clause, categories_config: dict | None = None) -> Clause:
    """
    Assign a Category, category_confidence, and category_source to a Clause
    based on deterministic keyword scoring across clause heading and text.
    """
    config = categories_config if categories_config is not None else (_CACHED_CONFIG or load_categories_config())

    text_lower = clause.text.lower()
    heading_lower = (clause.heading or "").lower()

    scores: dict[str, float] = {}

    for cat_name, cat_data in config.items():
        if cat_name == "other":
            continue

        keywords = cat_data.get("keywords", [])
        score = 0.0

        for kw in keywords:
            kw_clean = kw.lower().strip()
            # Word boundary regex for clean matching
            pattern = re.compile(rf"\b{re.escape(kw_clean)}\b")

            # Heading match is strongly weighted (weight: 3.0)
            if pattern.search(heading_lower):
                score += 3.0

            # Body text match (weight: 1.0 per distinct keyword match)
            if pattern.search(text_lower):
                score += 1.0

        if score > 0:
            scores[cat_name] = score

    if not scores:
        clause.category = "other"
        clause.category_confidence = 0.0
        clause.category_source = "keyword"
        return clause

    # Sort categories by match score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = sorted_scores[0]

    # Check for ambiguous tie (two top categories have identical scores)
    if len(sorted_scores) > 1 and sorted_scores[1][1] == best_score:
        # Insufficient evidence to disambiguate
        clause.category = "other"
        clause.category_confidence = 0.25
        clause.category_source = "keyword"
        return clause

    # Compute confidence based on score strength
    # A heading match alone gives score >= 3.0 -> confidence >= 0.80
    if best_score >= 4.0:
        confidence = 0.95
    elif best_score >= 3.0:
        confidence = 0.85
    elif best_score >= 2.0:
        confidence = 0.70
    else:
        # Only 1 weak keyword occurrence
        confidence = 0.45

    clause.category = best_cat  # type: ignore
    clause.category_confidence = round(confidence, 2)
    clause.category_source = "keyword"
    return clause


def tag_clauses(clauses: list[Clause], categories_config: dict | None = None) -> list[Clause]:
    """Tag a list of clauses in place and return them."""
    cfg = categories_config if categories_config is not None else (_CACHED_CONFIG or load_categories_config())
    for clause in clauses:
        tag_clause(clause, cfg)
    return clauses
