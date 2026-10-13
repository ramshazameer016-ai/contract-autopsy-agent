"""
Data schemas for Contract Autopsy Agent.
Defined strictly according to Architecture Document (Section E).
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

# E.1 Shared types and evidence
Category = Literal["payment", "ip", "termination", "scope", "liability", "confidentiality", "other"]
Rating = Literal["RED", "YELLOW", "GREEN", "UNRATED"]
EvidenceStatus = Literal["present", "absent", "ambiguous", "conflicting", "insufficient"]


class EvidenceSpan(BaseModel):
    """Character offset span and quote within a clause supporting a finding or fact."""
    doc_id: str                          # "A" or "B"
    clause_id: str                       # e.g. "A-001"
    page: int | None = None              # None for DOCX and pasted text
    start: int                           # character offset inside clause.text
    end: int                             # character offset inside clause.text
    quote: str                           # must equal clause.text[start:end]


# E.2 Extracted document
class Page(BaseModel):
    """Single page or logical block of extracted document."""
    number: int                          # 1-indexed page or paragraph index
    text: str                            # extracted text
    method: Literal["text", "ocr", "pasted"]


class ExtractedDocument(BaseModel):
    """Full extracted document representation with verbatim text and metadata."""
    doc_id: str                          # "A" or "B"
    filename: str
    file_type: Literal["pdf", "docx", "txt"]
    pages: list[Page]
    full_text: str
    ocr_used: bool = False
    scanned_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sha256: str                          # content hash for deduplication/caching


# E.3 Structured clause (with extracted facts)
class Fact(BaseModel):
    """Structured fact extracted from a contract clause."""
    slot: str                            # from categories.yaml, e.g. "payment.due_days"
    value: str | float | bool | None = None
    unit: str | None = None
    source: Literal["regex", "llm", "user"] = "regex"
    confidence: Literal["high", "medium", "low"] = "high"
    evidence: EvidenceSpan | None = None


class Clause(BaseModel):
    """Segmented contract clause with verbatim text and location tracking."""
    clause_id: str                       # e.g. "A-001"
    doc_id: str                          # "A" or "B"
    number: str | None = None            # e.g. "1.1", "Article 2"
    heading: str | None = None           # e.g. "Payment Terms"
    text: str                            # verbatim clause text
    page_start: int | None = None        # starting page (PDF) or paragraph index
    page_end: int | None = None          # ending page (PDF)
    char_start: int                      # start offset in document full_text
    char_end: int                        # end offset in document full_text
    category: Category = "other"
    category_confidence: float = 0.0     # 0.0 to 1.0
    category_source: Literal["keyword", "llm", "manual"] = "keyword"
    facts: list[Fact] = Field(default_factory=list)


# E.4 Risk finding
class TriggeredRule(BaseModel):
    """Rule that fired during rubric evaluation."""
    rule_id: str                         # e.g. "PAY-R1"
    severity: Rating                     # RED, YELLOW, GREEN, UNRATED
    points: int                          # risk score points added or subtracted
    why: str                             # explanation template
    fact_slots: list[str] = Field(default_factory=list)


class RiskFinding(BaseModel):
    """Evaluated risk finding for a clause or absent category."""
    finding_id: str                      # unique finding ID, e.g. "F-A-002"
    doc_id: str                          # "A" or "B"
    category: Category
    clause_id: str | None = None         # None when the category is absent
    rating: Rating                       # RED, YELLOW, GREEN, UNRATED
    evidence_status: EvidenceStatus      # present, absent, ambiguous, conflicting, insufficient
    score: int | None = None             # 0-100 risk points clamped
    triggered_rules: list[TriggeredRule] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    plain_reason: str                    # built from rule text, not free-form AI
    kb_refs: list[str] = Field(default_factory=list)
    rubric_version: str = "0.1.0"
    conflicts_with: list[str] = Field(default_factory=list) # conflicting clause_ids


class ContractAnalysis(BaseModel):
    """Complete Stage 2 contract analysis artifact."""
    doc_id: str
    document: ExtractedDocument
    clauses: list[Clause]
    findings: list[RiskFinding]
    category_ratings: dict[str, Rating] = Field(default_factory=dict)
    category_scores: dict[str, int] = Field(default_factory=dict)
    overall_risk_score: int = 0
