"""
Stage 1 Comprehensive Acceptance Tests for Contract Autopsy Agent.
Validates extraction, segmentation, offsets, category tagging,
and scanned PDF fallback handling using synthetic data only.
"""

from pathlib import Path
import pytest

from src.ca.schemas import ExtractedDocument, Clause
from src.ca.extraction.loader import load_document, load_pasted_text
from src.ca.extraction.pdf_reader import read_pdf
from src.ca.extraction.docx_reader import read_docx
from src.ca.extraction.ocr_reader import is_ocr_available, get_ocr_status_message
from src.ca.segmentation.segmenter import segment_document
from src.ca.segmentation.tagger import tag_clause, tag_clauses, load_categories_config

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_contracts"


def test_1_pdf_text_extraction():
    """Requirement 1: PDF extraction using PyMuPDF."""
    pdf_path = SAMPLE_DIR / "contract_a.pdf"
    doc = read_pdf(pdf_path, doc_id="A")
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 2
    assert "SYNTHETIC TEST CONTRACT A" in doc.full_text
    assert doc.ocr_used is False
    assert len(doc.sha256) == 64


def test_2_docx_text_extraction():
    """Requirement 2: DOCX extraction using python-docx."""
    docx_path = SAMPLE_DIR / "contract_c.docx"
    doc = read_docx(docx_path, doc_id="C")
    assert doc.file_type == "docx"
    assert len(doc.pages) > 0
    assert "SYNTHETIC TEST CONTRACT C" in doc.full_text
    assert len(doc.sha256) == 64


def test_3_pasted_text_processing():
    """Requirement 3: Pasted plain-text input."""
    raw = (
        "1. Scope of Work\n"
        "Contractor shall perform backend architecture design.\n\n"
        "2. Payment Terms\n"
        "Client shall pay INR 50,000 within 15 days of invoice."
    )
    doc = load_pasted_text(raw, doc_id="A", filename="custom_pasted.txt")
    assert doc.file_type == "txt"
    assert len(doc.pages) == 2
    assert doc.pages[0].method == "pasted"
    assert doc.full_text == raw


def test_4_numbered_clauses_and_headings():
    """Requirement 4: Detect numbered clauses and headings."""
    doc = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    clauses = segment_document(doc)
    numbers = [c.number for c in clauses if c.number is not None]
    headings = [c.heading for c in clauses if c.heading is not None]

    assert "1" in numbers
    assert "2" in numbers
    assert any("Scope of Work" in h for h in headings)
    assert any("Fees and Payment" in h for h in headings)


def test_5_paragraph_splitting_fallback():
    """Requirement 5: Fallback paragraph splitting when no numbered headings exist."""
    unstructured_text = (
        "This is an unnumbered contract between Party Alpha and Party Beta.\n\n"
        "The first party will deliver consulting work over the next two months.\n\n"
        "Payment will be remitted following review and acceptance."
    )
    doc = load_pasted_text(unstructured_text, doc_id="A")
    clauses = segment_document(doc)
    # Must fallback to paragraphs
    assert len(clauses) >= 2
    for c in clauses:
        assert c.text in unstructured_text


def test_6_clause_text_preservation():
    """Requirement 6: Preserve original wording, punctuation and clause text."""
    doc = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    clauses = segment_document(doc)
    clause_payment = next(c for c in clauses if c.number == "2")
    assert "2. Fees and Payment Terms" in clause_payment.text


def test_7_every_extracted_clause_is_substring():
    """Requirement 7: Every extracted clause text is a substring of source document."""
    for sample in ["contract_a.pdf", "contract_b.pdf", "contract_c.docx"]:
        doc = load_document(SAMPLE_DIR / sample, doc_id="A")
        clauses = segment_document(doc)
        assert len(clauses) > 0
        for c in clauses:
            assert c.text in doc.full_text, f"Clause {c.clause_id} is not substring of {sample}"


def test_8_character_offsets_exact_match():
    """Requirement 8: Character offsets point to the exact source text."""
    for sample in ["contract_a.pdf", "contract_b.pdf", "contract_c.docx"]:
        doc = load_document(SAMPLE_DIR / sample, doc_id="A")
        clauses = segment_document(doc)
        for c in clauses:
            slice_text = doc.full_text[c.char_start:c.char_end]
            assert slice_text == c.text, (
                f"Offset mismatch in {c.clause_id} ({sample}): "
                f"offsets ({c.char_start}, {c.char_end}) did not produce exact clause text"
            )


def test_9_category_tagging_and_uncertainty():
    """Requirement 9: Category tagging using categories.yaml and uncertainty handling."""
    # Deterministic category tags
    doc = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    clauses = segment_document(doc)
    tag_clauses(clauses)

    categories_found = {c.category for c in clauses}
    assert "payment" in categories_found
    assert "ip" in categories_found
    assert "termination" in categories_found
    assert "scope" in categories_found

    # Check high confidence on explicit heading matches
    payment_clause = next(c for c in clauses if c.number == "2")
    assert payment_clause.category == "payment"
    assert payment_clause.category_confidence >= 0.85
    assert payment_clause.category_source == "keyword"

    # Check uncertainty handling on vague text with no keywords
    vague_clause = Clause(
        clause_id="A-999",
        doc_id="A",
        heading="Miscellaneous Item",
        text="The quick brown fox jumps over the lazy dog.",
        char_start=0,
        char_end=43,
    )
    tag_clause(vague_clause)
    assert vague_clause.category == "other"
    assert vague_clause.category_confidence == 0.0


def test_10_empty_and_scanned_pdf_warning():
    """Requirement 10: Empty or scanned PDF warning/fallback behaviour."""
    scanned_path = SAMPLE_DIR / "scanned.pdf"
    doc = load_document(scanned_path, doc_id="A")
    assert len(doc.scanned_pages) > 0
    assert any("scanned" in w.lower() or "image-only" in w.lower() for w in doc.warnings)
    assert any("paste text" in w.lower() for w in doc.warnings)

    # OCR status is honestly reported as unavailable
    assert is_ocr_available() is False
    assert "not available" in get_ocr_status_message().lower()


def test_11_contract_a_and_b_identical_processing():
    """Requirement 11: Contract A and Contract B use the same processing path."""
    doc_a = load_document(SAMPLE_DIR / "contract_a.pdf", doc_id="A")
    clauses_a = segment_document(doc_a)
    tag_clauses(clauses_a)

    doc_b = load_document(SAMPLE_DIR / "contract_b.pdf", doc_id="B")
    clauses_b = segment_document(doc_b)
    tag_clauses(clauses_b)

    assert doc_a.doc_id == "A"
    assert doc_b.doc_id == "B"
    assert all(c.doc_id == "A" for c in clauses_a)
    assert all(c.doc_id == "B" for c in clauses_b)
    assert all(c.clause_id.startswith("A-") for c in clauses_a)
    assert all(c.clause_id.startswith("B-") for c in clauses_b)


def test_12_stage_0_functionality_intact():
    """Requirement 12: Stage 0 functionality remains intact."""
    from config.settings import GEMINI_MODEL, AI_MODE, is_offline, FAVOURABILITY_GAP_THRESHOLD
    assert GEMINI_MODEL != ""
    assert FAVOURABILITY_GAP_THRESHOLD == 25
    assert is_offline() is True or AI_MODE in ("live", "offline")
    assert (Path(__file__).resolve().parent.parent / "requirements.txt").exists()
    assert (Path(__file__).resolve().parent.parent / ".gitignore").exists()
    assert (Path(__file__).resolve().parent.parent / ".env.example").exists()
