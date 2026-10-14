"""
Data schemas for Contract Autopsy Agent.
Defined strictly according to Architecture Document (Section E).
"""

from __future__ import annotations

from typing import Literal, Any
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


ComparisonStatus = Literal["matched", "missing_a", "missing_b", "missing_both", "not_comparable", "uncertain"]
FavoursContract = Literal["A", "B", "tie", "none"]
VerdictStatus = Literal["leans_a", "leans_b", "no_clear_winner", "insufficient_evidence"]

MANDATORY_BORROW_CAVEAT: str = (
    "This wording comes from another offer. It may not be legally transferable "
    "or acceptable to this party; use it only as a negotiation reference."
)


class ComparisonRow(BaseModel):
    """A matched slot or clause comparison row between Contract A and Contract B."""
    category: str
    slot: str
    label: str
    val_a: Any = None
    val_b: Any = None
    display_a: str = ""
    display_b: str = ""
    status: ComparisonStatus = "matched"
    favours: FavoursContract = "none"
    comparable: bool = True
    note: str = ""
    quote_a: str = ""
    quote_b: str = ""
    clause_id_a: str | None = None
    clause_id_b: str | None = None
    page_a: int | None = None
    page_b: int | None = None
    uncertain_match: bool = False


class BorrowSuggestion(BaseModel):
    """A cross-contract borrowing suggestion (Feature 7)."""
    source_contract: str
    target_contract: str
    category: str
    slot: str | None = None
    clause_id: str | None = None
    page: int | None = None
    verbatim_quote: str = Field(min_length=1)
    reason: str
    caveat: str = MANDATORY_BORROW_CAVEAT
    fav_diff: float | None = None
    suggestion_type: Literal["favourability_gap", "missing_clause"] = "favourability_gap"


class PriorityWeights(BaseModel):
    """User priority weights (0=ignore, 1=low, 2=medium, 3=high)."""
    payment: int = Field(default=2, ge=0, le=3)
    ip: int = Field(default=2, ge=0, le=3)
    termination: int = Field(default=2, ge=0, le=3)
    scope: int = Field(default=2, ge=0, le=3)
    liability: int = Field(default=2, ge=0, le=3)
    confidentiality: int = Field(default=2, ge=0, le=3)

    def to_dict(self) -> dict[str, int]:
        return {
            "payment": self.payment,
            "ip": self.ip,
            "termination": self.termination,
            "scope": self.scope,
            "liability": self.liability,
            "confidentiality": self.confidentiality,
        }


class ComparisonVerdict(BaseModel):
    """Structured decision and verdict artifact."""
    status: VerdictStatus
    headline: str
    score_a: float
    score_b: float
    margin: float
    coverage: float
    scored_categories: list[str] = Field(default_factory=list)
    unscored_categories: list[str] = Field(default_factory=list)
    category_summaries: list[str] = Field(default_factory=list)
    dealbreakers: list[str] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    generated_by: Literal["deterministic", "llm_rewritten"] = "deterministic"
    full_text: str = ""


class ComparisonResult(BaseModel):
    """Full Stage 5 comparison bundle."""
    doc_id_a: str
    doc_id_b: str
    filename_a: str
    filename_b: str
    weights: dict[str, int]
    category_fav_a: dict[str, float]
    category_fav_b: dict[str, float]
    rows: list[ComparisonRow]
    verdict: ComparisonVerdict
    borrow_suggestions: list[BorrowSuggestion]

