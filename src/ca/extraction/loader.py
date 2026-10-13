"""
Universal document loader dispatching PDF, DOCX, and pasted text input.
Ensures identical processing pipeline for Contract A and Contract B.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

from src.ca.schemas import Page, ExtractedDocument
from src.ca.extraction.pdf_reader import read_pdf
from src.ca.extraction.docx_reader import read_docx


def load_pasted_text(
    text: str,
    doc_id: str = "A",
    filename: str = "pasted_text.txt",
) -> ExtractedDocument:
    """
    Wrap pasted plain text into an ExtractedDocument schema.
    Validates that text is not empty or pure whitespace.
    """
    clean_text = text.strip()
    warnings: list[str] = []

    if not clean_text:
        warnings.append("Pasted text is empty. No content was provided for analysis.")

    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Split into paragraphs to give paragraph-level location tracking
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pages: list[Page] = []
    if paragraphs:
        for idx, p in enumerate(paragraphs):
            pages.append(
                Page(
                    number=idx + 1,
                    text=p,
                    method="pasted",
                )
            )
    else:
        pages.append(
            Page(
                number=1,
                text=text,
                method="pasted",
            )
        )

    return ExtractedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type="txt",
        pages=pages,
        full_text=text,
        ocr_used=False,
        scanned_pages=[],
        warnings=warnings,
        sha256=sha256,
    )


def load_document(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
    doc_id: str = "A",
    file_type: str | None = None,
) -> ExtractedDocument:
    """
    Load document by dispatching based on file extension or provided file_type.
    Never silently accepts corrupt or unextractable files as valid contracts.
    """
    # Determine filename and extension
    resolved_filename = filename
    if not resolved_filename:
        if isinstance(source, (str, Path)):
            resolved_filename = Path(source).name
        elif hasattr(source, "name"):
            resolved_filename = getattr(source, "name")
        else:
            resolved_filename = f"document_{doc_id}.pdf"

    ext = Path(resolved_filename).suffix.lower()
    if file_type:
        detected_type = file_type.lower().replace(".", "")
    elif ext in (".pdf",):
        detected_type = "pdf"
    elif ext in (".docx",):
        detected_type = "docx"
    elif ext in (".txt",):
        detected_type = "txt"
    else:
        raise ValueError(
            f"Unsupported file format '{ext}' for file '{resolved_filename}'. "
            "Contract Autopsy Agent supports PDF, DOCX, and plain text."
        )

    if detected_type == "pdf":
        doc = read_pdf(source, doc_id=doc_id, filename=resolved_filename)
    elif detected_type == "docx":
        doc = read_docx(source, doc_id=doc_id, filename=resolved_filename)
    elif detected_type == "txt":
        if isinstance(source, (str, Path)) and Path(source).exists():
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        elif isinstance(source, bytes):
            raw_text = source.decode("utf-8", errors="replace")
        elif hasattr(source, "read"):
            data = source.read()
            raw_text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
        elif isinstance(source, str):
            raw_text = source
        else:
            raise ValueError(f"Cannot read plain text from source type: {type(source)}")
        doc = load_pasted_text(raw_text, doc_id=doc_id, filename=resolved_filename)
    else:
        raise ValueError(f"Unknown document type: {detected_type}")

    # Check for empty extracted text
    if not doc.full_text.strip():
        if "Document contains no readable text" not in " ".join(doc.warnings):
            doc.warnings.append(
                f"File '{resolved_filename}' yielded 0 characters of readable text. "
                "The file may be blank, password-protected, or scanned."
            )

    return doc
