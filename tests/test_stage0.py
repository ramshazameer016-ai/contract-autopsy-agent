"""
Stage 0 Automated Verification Tests for Contract Autopsy Agent.
Validates dependencies, synthetic contract availability, PDF parsing,
settings loading, and offline mode.
"""

from pathlib import Path
import pytest
import pymupdf
from docx import Document

from config.settings import (
    AI_MODE,
    GEMINI_MODEL,
    FAVOURABILITY_GAP_THRESHOLD,
    BM25_SIMILARITY_THRESHOLD,
    is_offline,
    mask_key,
)


def test_project_root_and_configs():
    """Verify core Stage 0 files exist."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "requirements.txt").exists(), "requirements.txt missing"
    assert (root / ".gitignore").exists(), ".gitignore missing"
    assert (root / ".env.example").exists(), ".env.example missing"
    assert (root / "README.md").exists(), "README.md missing"
    assert (root / "app.py").exists(), "app.py missing"
    assert (root / "config" / "settings.py").exists(), "config/settings.py missing"


def test_synthetic_contracts_exist():
    """Verify the 3 synthetic sample contracts exist."""
    contracts_dir = Path(__file__).resolve().parent / "sample_contracts"
    contract_a = contracts_dir / "contract_a.pdf"
    contract_b = contracts_dir / "contract_b.pdf"
    contract_c = contracts_dir / "contract_c.docx"

    assert contract_a.exists(), "contract_a.pdf missing"
    assert contract_b.exists(), "contract_b.pdf missing"
    assert contract_c.exists(), "contract_c.docx missing"


def test_pymupdf_pdf_extraction():
    """Acceptance Test: PyMuPDF opens a synthetic PDF and extracts text."""
    contracts_dir = Path(__file__).resolve().parent / "sample_contracts"
    doc = pymupdf.open(contracts_dir / "contract_a.pdf")
    assert len(doc) >= 1, "PDF has no pages"
    text = doc[0].get_text()
    assert "SYNTHETIC TEST CONTRACT" in text
    assert len(text) > 100
    doc.close()


def test_python_docx_extraction():
    """Verify python-docx can open and read synthetic contract C."""
    contracts_dir = Path(__file__).resolve().parent / "sample_contracts"
    doc = Document(contracts_dir / "contract_c.docx")
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "SYNTHETIC TEST CONTRACT" in full_text
    assert len(full_text) > 100


def test_settings_and_offline_mode():
    """Verify settings defaults and offline fallback."""
    assert GEMINI_MODEL != ""
    assert FAVOURABILITY_GAP_THRESHOLD == 25
    assert BM25_SIMILARITY_THRESHOLD == 0.35
    assert mask_key("") == "[NOT SET]"
    assert mask_key("1234567890abcdef") == "1234...cdef"
    # When no key is set, system should default to offline mode
    if not is_offline():
        assert AI_MODE == "live"
    else:
        assert AI_MODE == "offline"
