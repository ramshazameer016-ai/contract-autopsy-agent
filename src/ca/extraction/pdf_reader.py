"""
PDF extraction module using PyMuPDF.
Extracts verbatim text per page, detects pages with no extractable text,
and preserves character content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO
import pymupdf

from src.ca.schemas import Page, ExtractedDocument


def read_pdf(
    source: str | Path | bytes | BinaryIO,
    doc_id: str = "A",
    filename: str = "document.pdf",
) -> ExtractedDocument:
    """
    Read a PDF document using PyMuPDF and return an ExtractedDocument.
    Detects scanned/image-only pages when no extractable text is found.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = path.name
        with open(path, "rb") as f:
            data = f.read()
    elif isinstance(source, bytes):
        data = source
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise ValueError(f"Unsupported source type for PDF extraction: {type(source)}")

    sha256 = hashlib.sha256(data).hexdigest()
    doc = pymupdf.open(stream=data, filetype="pdf")

    pages: list[Page] = []
    scanned_pages: list[int] = []
    page_texts: list[str] = []

    for idx, page in enumerate(doc):
        page_num = idx + 1
        text = page.get_text()
        stripped = text.strip()

        # If page has fewer than 15 alphanumeric characters, flag as scanned/image-only
        alpha_count = sum(c.isalnum() for c in stripped)
        if alpha_count < 15:
            scanned_pages.append(page_num)

        pages.append(
            Page(
                number=page_num,
                text=text,
                method="text",
            )
        )
        page_texts.append(text)

    doc.close()

    full_text = "\n".join(page_texts)
    warnings: list[str] = []

    if len(pages) == 0:
        warnings.append("Document has 0 pages.")
    elif len(scanned_pages) == len(pages):
        warnings.append(
            f"All {len(pages)} page(s) appear to be scanned or image-only without an embedded text layer. "
            "OCR is currently unavailable on this system. Please use the Paste Text option."
        )
    elif scanned_pages:
        warnings.append(
            f"Page(s) {scanned_pages} have no text layer. Text from these pages may be missing. "
            "OCR is unavailable."
        )

    return ExtractedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type="pdf",
        pages=pages,
        full_text=full_text,
        ocr_used=False,
        scanned_pages=scanned_pages,
        warnings=warnings,
        sha256=sha256,
    )
