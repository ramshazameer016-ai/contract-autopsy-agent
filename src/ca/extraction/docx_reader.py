"""
DOCX extraction module using python-docx.
Extracts paragraphs and table cell contents with block index tracking.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import BinaryIO
from docx import Document

from src.ca.schemas import Page, ExtractedDocument


def read_docx(
    source: str | Path | bytes | BinaryIO,
    doc_id: str = "A",
    filename: str = "document.docx",
) -> ExtractedDocument:
    """
    Read a DOCX document using python-docx and return an ExtractedDocument.
    Extracts all paragraphs and table contents in linear order.
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
        raise ValueError(f"Unsupported source type for DOCX extraction: {type(source)}")

    sha256 = hashlib.sha256(data).hexdigest()
    doc = Document(io.BytesIO(data))

    pages: list[Page] = []
    block_texts: list[str] = []
    block_num = 1

    # Extract standard paragraphs
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            pages.append(
                Page(
                    number=block_num,
                    text=p.text,
                    method="text",
                )
            )
            block_texts.append(p.text)
            block_num += 1

    # Extract tables if present
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                row_str = " | ".join(row_cells)
                pages.append(
                    Page(
                        number=block_num,
                        text=row_str,
                        method="text",
                    )
                )
                block_texts.append(row_str)
                block_num += 1

    full_text = "\n\n".join(block_texts)
    warnings: list[str] = []
    if not full_text.strip():
        warnings.append("DOCX document contains no readable text paragraphs or tables.")

    return ExtractedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type="docx",
        pages=pages,
        full_text=full_text,
        ocr_used=False,
        scanned_pages=[],
        warnings=warnings,
        sha256=sha256,
    )
