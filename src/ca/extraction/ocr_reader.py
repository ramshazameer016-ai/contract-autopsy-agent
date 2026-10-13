"""
OCR reader module for scanned documents.
Honestly reports OCR availability status without faking execution.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import BinaryIO

from src.ca.schemas import ExtractedDocument


def is_ocr_available() -> bool:
    """Return True only if an OCR engine is genuinely available in the current environment."""
    has_rapidocr = importlib.util.find_spec("rapidocr_onnxruntime") is not None
    has_pytesseract = importlib.util.find_spec("pytesseract") is not None
    has_tesseract_cli = shutil.which("tesseract") is not None
    return bool(has_rapidocr or (has_pytesseract and has_tesseract_cli))


def get_ocr_status_message() -> str:
    """Return descriptive status message explaining OCR availability."""
    if is_ocr_available():
        return "OCR engine is available."
    return (
        "OCR is not available on this system (rapidocr-onnxruntime and tesseract are not installed). "
        "For scanned documents or image-based contracts, please use the Paste Text option."
    )


def extract_with_ocr(
    source: str | Path | bytes | BinaryIO,
    doc_id: str = "A",
    filename: str = "scanned.pdf",
) -> ExtractedDocument:
    """
    Attempt OCR extraction. Raises RuntimeError if OCR dependencies are not installed.
    Never fakes or simulates successful OCR.
    """
    if not is_ocr_available():
        raise RuntimeError(
            "OCR dependencies are not installed on this machine. "
            "Please copy and paste the contract text directly into the 'Paste Text' tab."
        )

    # Note: If OCR is installed in a future stage, implementation will plug in here.
    raise NotImplementedError("OCR backend is not configured.")
