"""
Analysis pipeline for Contract Autopsy Agent.
Orchestrates segmentation, tagging, fact extraction, rubric rule evaluation,
absence handling, and consistency checking into a ContractAnalysis artifact.
Identical code path used for Contract A and Contract B.
"""

from __future__ import annotations

from src.ca.schemas import (
    ExtractedDocument,
    Clause,
    RiskFinding,
    ContractAnalysis,
    Rating,
    Category,
)
from src.ca.segmentation.segmenter import segment_document
from src.ca.segmentation.tagger import tag_clauses
from src.ca.analysis.facts import extract_facts_for_clauses
from src.ca.analysis.rules import evaluate_clause, evaluate_absence, load_risk_rules
from src.ca.analysis.consistency import check_consistency

MANDATORY_CATEGORIES: list[Category] = [
    "payment",
    "ip",
    "termination",
    "scope",
    "liability",
    "confidentiality",
]


def analyse_contract(
    doc: ExtractedDocument,
    clauses: list[Clause] | None = None,
    rubric: dict | None = None,
) -> ContractAnalysis:
    """
    Run complete deterministic analysis pipeline on an ExtractedDocument.
    Returns ContractAnalysis model with all clauses, facts, and risk findings.
    """
    # 1. Segment document if not provided
    segmented_clauses = clauses if clauses is not None else segment_document(doc)

    # 2. Category tagging
    tag_clauses(segmented_clauses)

    # 3. Fact extraction
    extract_facts_for_clauses(segmented_clauses)

    # 4. Evaluate risk findings per clause
    rubric_cfg = rubric if rubric is not None else load_risk_rules()
    findings: list[RiskFinding] = []
    found_categories: set[str] = set()

    for c in segmented_clauses:
        # Evaluate for its tagged category
        if c.category != "other":
            finding = evaluate_clause(c, c.category, rubric_cfg)
            findings.append(finding)
            found_categories.add(c.category)

    # 5. Check for absent mandatory categories
    for mandatory_cat in MANDATORY_CATEGORIES:
        if mandatory_cat not in found_categories:
            absent_finding = evaluate_absence(mandatory_cat, doc.doc_id, rubric_cfg)
            findings.append(absent_finding)

    # 6. Apply cross-clause consistency checks
    check_consistency(findings, segmented_clauses)

    # 7. Aggregate category ratings and scores
    category_ratings: dict[str, Rating] = {}
    category_scores: dict[str, int] = {}

    for cat in MANDATORY_CATEGORIES:
        cat_findings = [f for f in findings if f.category == cat]
        if not cat_findings:
            category_ratings[cat] = "UNRATED"
            category_scores[cat] = 0
            continue

        # Highest severity wins
        if any(f.rating == "RED" for f in cat_findings):
            category_ratings[cat] = "RED"
        elif any(f.rating == "YELLOW" for f in cat_findings):
            category_ratings[cat] = "YELLOW"
        elif any(f.rating == "GREEN" for f in cat_findings):
            category_ratings[cat] = "GREEN"
        else:
            category_ratings[cat] = "UNRATED"

        # Worst (maximum) risk score in category
        scores = [f.score for f in cat_findings if f.score is not None]
        category_scores[cat] = max(scores) if scores else 0

    # Overall contract risk score (average of category scores)
    valid_scores = list(category_scores.values())
    overall_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

    return ContractAnalysis(
        doc_id=doc.doc_id,
        document=doc,
        clauses=segmented_clauses,
        findings=findings,
        category_ratings=category_ratings,
        category_scores=category_scores,
        overall_risk_score=overall_score,
    )
