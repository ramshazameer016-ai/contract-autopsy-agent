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
