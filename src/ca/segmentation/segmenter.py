"""
Clause segmentation module.
Splits contract text into verbatim clauses with exact character offsets,
heading detection, and paragraph-splitting fallback.
"""

from __future__ import annotations

import re
from src.ca.schemas import ExtractedDocument, Clause

# Regular expression pattern to detect numbered clauses and section headings
# Matches:
# - "1.", "1.1", "2.1.3", "1)"
# - "Article 1", "ARTICLE II", "Clause 3", "Section 4.1"
# - "Schedule 1", "Exhibit A"
HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:"
    r"(?:Article|ARTICLE|Clause|CLAUSE|Section|SECTION|Schedule|Exhibit)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[:.\s-]*[^\n]*|"
    r"[0-9]+(?:\.[0-9]+)+[:.\s-]*[^\n]*|"
    r"[0-9]+[.)][ \t]+[^\n]+"
    r")"
)


def _compute_page_boundaries(doc: ExtractedDocument) -> list[tuple[int, int, int]]:
    """
    Compute (page_num, start_offset, end_offset) for each page in doc.full_text.
    """
    boundaries: list[tuple[int, int, int]] = []
    current_offset = 0

    for p in doc.pages:
        p_len = len(p.text)
        boundaries.append((p.number, current_offset, current_offset + p_len))
        # Account for newline separator used when joining pages
        current_offset += p_len + 1

    return boundaries


def _find_page_for_offset(offset: int, boundaries: list[tuple[int, int, int]]) -> int | None:
    """Find page number containing a given character offset."""
    for page_num, start, end in boundaries:
        if start <= offset <= end:
            return page_num
    if boundaries:
        return boundaries[-1][0]
    return None


def segment_document(doc: ExtractedDocument) -> list[Clause]:
    """
    Segment an ExtractedDocument into a list of verbatim Clause models.
    Preserves exact character offsets such that doc.full_text[char_start:char_end] == clause.text.
    """
    full_text = doc.full_text
    doc_id = doc.doc_id
    page_boundaries = _compute_page_boundaries(doc) if doc.file_type == "pdf" else []

    if not full_text.strip():
        return []

    # Find heading matches
    matches = list(HEADING_PATTERN.finditer(full_text))

    # If fewer than 2 headings are found, use paragraph-splitting fallback
    if len(matches) < 2:
        return _segment_paragraphs(doc, page_boundaries)

    clauses: list[Clause] = []
    clause_index = 1

    # Check for text preceding the first heading (Preamble / Recitals)
    first_start = matches[0].start()
    if first_start > 0 and full_text[:first_start].strip():
        preamble_text = full_text[:first_start]
        p_start = _find_page_for_offset(0, page_boundaries)
        p_end = _find_page_for_offset(first_start - 1, page_boundaries)

        clauses.append(
            Clause(
                clause_id=f"{doc_id}-000",
                doc_id=doc_id,
                number="0",
                heading="Preamble / Parties",
                text=preamble_text,
                page_start=p_start,
                page_end=p_end,
                char_start=0,
                char_end=first_start,
                category="other",
                category_confidence=0.5,
                category_source="keyword",
            )
        )

    # Segment each heading up to the next heading or EOF
    for i, match in enumerate(matches):
        char_start = match.start()
        char_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        clause_text = full_text[char_start:char_end]

        # Extract number and heading title from the first line
        first_line = clause_text.splitlines()[0].strip()
        clause_number, clause_heading = _parse_number_and_heading(first_line)

        p_start = _find_page_for_offset(char_start, page_boundaries)
        p_end = _find_page_for_offset(max(char_start, char_end - 1), page_boundaries)

        clause_id = f"{doc_id}-{clause_index:03d}"
        clause_index += 1

        clauses.append(
            Clause(
                clause_id=clause_id,
                doc_id=doc_id,
                number=clause_number,
                heading=clause_heading,
                text=clause_text,
                page_start=p_start,
                page_end=p_end,
                char_start=char_start,
                char_end=char_end,
                category="other",
                category_confidence=0.0,
                category_source="keyword",
            )
        )

    return clauses


def _segment_paragraphs(
    doc: ExtractedDocument,
    page_boundaries: list[tuple[int, int, int]],
) -> list[Clause]:
    """
    Fallback segmenter when regular numbered headings are not found.
    Splits by blank-line paragraph delimiters while maintaining exact offsets.
    """
    full_text = doc.full_text
    doc_id = doc.doc_id
    clauses: list[Clause] = []

    # Find blocks of consecutive non-whitespace lines
    para_pattern = re.compile(r"(?s)\S.*?(?=\n\s*\n|\Z)")
    matches = list(para_pattern.finditer(full_text))

    if not matches:
        # Fallback to single block if no regex match
        return [
            Clause(
                clause_id=f"{doc_id}-001",
                doc_id=doc_id,
                number="1",
                heading="Document Content",
                text=full_text,
                page_start=1 if doc.pages else None,
                page_end=len(doc.pages) if doc.pages else None,
                char_start=0,
                char_end=len(full_text),
                category="other",
                category_confidence=0.0,
                category_source="keyword",
            )
        ]

    clause_index = 1
    for match in matches:
        char_start = match.start()
        char_end = match.end()
        text = full_text[char_start:char_end]

        first_line = text.splitlines()[0].strip()
        num, heading = _parse_number_and_heading(first_line)

        p_start = _find_page_for_offset(char_start, page_boundaries)
        p_end = _find_page_for_offset(char_end - 1, page_boundaries)

        clauses.append(
            Clause(
                clause_id=f"{doc_id}-{clause_index:03d}",
                doc_id=doc_id,
                number=num,
                heading=heading or f"Paragraph {clause_index}",
                text=text,
                page_start=p_start,
                page_end=p_end,
                char_start=char_start,
                char_end=char_end,
                category="other",
                category_confidence=0.0,
                category_source="keyword",
            )
        )
        clause_index += 1

    return clauses


def _parse_number_and_heading(line: str) -> tuple[str | None, str | None]:
    """Parse clause number and heading title from the first line of a clause."""
    clean_line = line.strip()

    # Pattern: "1. Scope of Work" or "1.1 Deliverables"
    m = re.match(r"^((?:Article|ARTICLE|Clause|CLAUSE|Section|SECTION)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*|[0-9]+(?:\.[0-9]+)*[.)]?)\s*[:.-]?\s*(.*)$", clean_line)
    if m:
        num = m.group(1).rstrip(".:-")
        title = m.group(2).strip()
        return num, title if title else None

    # Fallback: take line as heading
    return None, clean_line[:80]
