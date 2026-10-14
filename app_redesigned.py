"""
Contract Autopsy Agent - Streamlit Application Entry Point.
Modern SaaS Legal-Tech Frontend designed for Indian Freelancers and Evaluators.

Five Core Sections:
1. Upload & Analyze
2. Understand Your Risks
3. Explain in My Language
4. Negotiate Better Terms
5. Compare Two Offers

Front-end-only redesign: home page with five colour-coded workflow cards, compact
top navigation, no sidebar. All backend calls, session-state keys, consent rules
and AI-availability checks are unchanged.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import (
    AI_MODE,
    GEMINI_MODEL,
    GROQ_MODEL,
    GEMINI_API_KEY,
    FAVOURABILITY_GAP_THRESHOLD,
    BM25_SIMILARITY_THRESHOLD,
    COVERAGE_THRESHOLD,
    MARGIN_THRESHOLD,
    DISCLAIMER_TEXT,
    APP_VERSION,
    mask_key,
    is_offline,
    is_ai_available,
    get_ai_status,
)
from src.ca.extraction.loader import load_document, load_pasted_text
from src.ca.extraction.ocr_reader import get_ocr_status_message
from src.ca.segmentation.segmenter import segment_document
from src.ca.segmentation.tagger import tag_clauses
from src.ca.analysis.pipeline import analyse_contract
from src.ca.ui.components import (
    render_heatmap_summary,
    render_findings_details,
    get_category_display_name,
    RATING_COLORS,
    RATING_ICONS,
)
from src.ca.schemas import (
    ExtractedDocument,
    Clause,
    ContractAnalysis,
    ComparisonRow,
    BorrowSuggestion,
    ComparisonVerdict,
    MANDATORY_BORROW_CAVEAT,
)
from src.ca.generation.summary import summarize_contract
from src.ca.generation.counter_clause import generate_counter_clause
from src.ca.translation.translate import translate_category, translate_status
from src.ca.compare import (
    match_contracts,
    ALL_PRIORITY_CATEGORIES,
    calculate_category_favourability,
    calculate_weighted_scores,
    find_dealbreakers,
    build_deterministic_verdict,
    rewrite_verdict_with_ai,
    generate_borrow_suggestions,
)
from src.ca.kb.retriever import get_kb_status
from src.ca.llm.client import clear_session_cache


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Contract Autopsy Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# Navigation definitions
# ---------------------------------------------------------------------

PAGE_HOME = "Home"
PAGE_UPLOAD = "Upload & Analyze"
PAGE_RISKS = "Understand Your Risks"
PAGE_EXPLAIN = "Explain in My Language"
PAGE_NEGOTIATE = "Negotiate Better Terms"
PAGE_COMPARE = "Compare Two Offers"

PAGES = [PAGE_HOME, PAGE_UPLOAD, PAGE_RISKS, PAGE_EXPLAIN, PAGE_NEGOTIATE, PAGE_COMPARE]

# page -> (slug, icon, accent colour, card description, page subtitle)
PAGE_META: dict[str, tuple[str, str, str, str, str]] = {
    PAGE_UPLOAD: (
        "upload", "📁", "#3b82f6",
        "Upload or paste a contract to extract its clauses and identify potential risks.",
        "Upload a PDF or DOCX agreement, paste raw contract text, or test immediately "
        "using safe, synthetic sample contracts.",
    ),
    PAGE_RISKS: (
        "risks", "🔍", "#1b2a4a",
        "Explore risk findings, important terms, missing clauses, and supporting evidence.",
        "Deterministic risk ratings calculated directly from transparent YAML rules. "
        "Scores range from 0 (safest freelance terms) to 100 (highest risk / client-leaning). "
        "Missing protections automatically incur caution points.",
    ),
    PAGE_EXPLAIN: (
        "explain", "🌐", "#4a78b8",
        "Understand your contract in simple language with English, Hindi, Telugu, or Tamil support.",
        "Complex contract terms translated into simple, everyday language for Indian freelancers. "
        "Select your preferred language. All claims are grounded directly in contract quotes.",
    ),
    PAGE_NEGOTIATE: (
        "negotiate", "🤝", "#2b4c85",
        "Review concerning clauses, prepare alternative wording, and get practical negotiation talking points.",
        "Turn high-risk clauses into fair counter-clauses and concrete negotiation talking points. "
        "All proposals are negotiation starting points, not formal legal counsel.",
    ),
    PAGE_COMPARE: (
        "compare", "⚖️", "#2f6fb5",
        "Compare two contracts side by side, review category-level differences, and identify important trade-offs.",
        "Evaluate two contracts side-by-side to compare risk scores, identify dealbreakers, "
        "and borrow better terms before making a decision.",
    ),
}

if st.session_state.get("nav_page") not in PAGES:
    st.session_state["nav_page"] = PAGE_HOME


def _go(page: str) -> None:
    """Button callback: switch workflow (callbacks may safely set the nav widget key)."""
    st.session_state["nav_page"] = page


def _reset_all() -> None:
    """Button callback: same behaviour as the previous Reset All button."""
    clear_session_cache()
    st.session_state.clear()


# ---------------------------------------------------------------------
# Design system (CSS)
# ---------------------------------------------------------------------

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --ca-bg: #f4f6f9;
    --ca-navy: #1b2a4a;
    --ca-slate: #5f6b7e;
    --ca-border: #dde3ec;
    --ca-card: #ffffff;
    --ca-blue: #3b82f6;
    --ca-blue-dark: #2563eb;
    --ca-radius: 16px;
    --ca-shadow: 0 1px 2px rgba(27,42,74,0.05), 0 6px 18px rgba(27,42,74,0.07);
    --ca-serif: "Playfair Display", "Georgia", "Times New Roman", serif;
    --ca-sans: "Source Sans 3", "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--ca-bg);
    color: var(--ca-navy);
    font-family: var(--ca-sans);
}
.stMarkdown, .stText, label, p, li, input, textarea { font-family: var(--ca-sans); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4, h5 { color: var(--ca-navy); font-family: var(--ca-serif); letter-spacing: -0.005em; font-weight: 700; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--ca-slate); }

/* No sidebar, no toggle */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"] { display: none !important; }

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--ca-card);
    border-radius: var(--ca-radius);
    border-color: var(--ca-border) !important;
    box-shadow: var(--ca-shadow);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.ca-card) {
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.ca-card):hover {
    transform: translateY(-4px);
    box-shadow: 0 2px 4px rgba(27,42,74,0.06), 0 16px 32px rgba(27,42,74,0.14);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: none;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--ca-border);
    color: var(--ca-navy);
    background: #fff;
    font-family: var(--ca-sans);
    font-weight: 600;
    padding: 0.55rem 1rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover { border-color: var(--ca-blue); color: var(--ca-blue-dark); }
.stButton > button:focus-visible { outline: 3px solid rgba(59,130,246,0.4); outline-offset: 2px; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--ca-blue);
    border: 1px solid var(--ca-blue);
    color: #fff;
    box-shadow: 0 4px 12px rgba(59,130,246,0.30);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--ca-blue-dark);
    border-color: var(--ca-blue-dark);
    color: #fff;
    box-shadow: 0 6px 16px rgba(37,99,235,0.38);
}

/* Pill navigation (horizontal radios) */
div[role="radiogroup"] { gap: 0.45rem; flex-wrap: wrap; }
div[role="radiogroup"] > label {
    background: #fff;
    border: 1px solid var(--ca-border);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    margin: 0 !important;
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease;
}
div[role="radiogroup"] > label > div:first-child { display: none; }
div[role="radiogroup"] > label:hover { border-color: var(--ca-blue); }
div[role="radiogroup"] > label:has(input:checked) {
    background: var(--ca-navy);
    border-color: var(--ca-navy);
}
div[role="radiogroup"] > label:has(input:checked) p,
div[role="radiogroup"] > label:has(input:checked) div { color: #fff !important; }

/* Brand bar */
.ca-brand { font-family: var(--ca-serif); font-size: 1.3rem; font-weight: 700; color: var(--ca-navy); padding-top: 0.25rem; }
.ca-brand .ca-logo {
    display: inline-block; width: 34px; height: 34px; line-height: 34px; text-align: center;
    border-radius: 8px; margin-right: 0.55rem; font-size: 1rem;
    background: var(--ca-navy);
}

/* Hero */
.ca-hero {
    background: linear-gradient(135deg, #1b2a4a 0%, #223a68 100%);
    border-radius: 22px;
    padding: 3rem 2.4rem 2.4rem 2.4rem;
    margin: 0.8rem 0 1.4rem 0;
    box-shadow: 0 12px 30px rgba(27,42,74,0.25);
}
.ca-hero h1 {
    font-family: var(--ca-serif); font-size: 2.8rem; line-height: 1.18; margin: 0 0 0.8rem 0;
    font-weight: 700; color: #ffffff;
}
.ca-hero .ca-sub { font-family: var(--ca-serif); font-style: italic; font-size: 1.3rem; color: #93c5fd; font-weight: 500; margin-bottom: 0.9rem; }
.ca-hero .ca-desc { font-size: 1.08rem; color: #cbd5e1; max-width: 700px; line-height: 1.65; }
.ca-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.3rem; }
.ca-trust {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
    padding: 0.3rem 0.9rem; font-size: 0.86rem; color: #e2e8f0; font-weight: 500;
}

/* Section titles */
.ca-section-title { font-family: var(--ca-serif); font-size: 1.5rem; font-weight: 700; margin: 1.4rem 0 0.2rem 0; color: var(--ca-navy); }
.ca-section-sub { color: var(--ca-slate); margin-bottom: 0.9rem; }

/* Workflow cards */
.ca-chip {
    width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 1.5rem; flex-shrink: 0;
}
.ca-card-title { font-family: var(--ca-serif); font-size: 1.22rem; font-weight: 700; margin: 0.7rem 0 0.35rem 0; color: var(--ca-navy); }
.ca-card-desc { color: var(--ca-slate); font-size: 0.97rem; line-height: 1.55; min-height: 4.6em; margin-bottom: 0.5rem; }

/* Page header */
.ca-ph {
    display: flex; align-items: center; gap: 1rem; background: #fff;
    border: 1px solid var(--ca-border); border-left: 6px solid var(--accent);
    border-radius: var(--ca-radius); padding: 1rem 1.3rem; margin: 0.4rem 0 1.2rem 0;
    box-shadow: var(--ca-shadow);
}
.ca-ph-title { font-family: var(--ca-serif); font-size: 1.6rem; font-weight: 700; color: var(--ca-navy); }
.ca-ph-desc { color: var(--ca-slate); font-size: 0.97rem; line-height: 1.5; margin-top: 0.15rem; }

/* Panel labels (Contract A / B) */
.ca-panel { display: flex; align-items: center; gap: 0.6rem; font-family: var(--ca-serif); font-weight: 700; font-size: 1.1rem; margin-bottom: 0.2rem; color: var(--ca-navy); }
.ca-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }

/* Badges (status colours stay semantic) */
.ca-badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.8rem;
    font-weight: 600; border: 1px solid transparent; white-space: nowrap;
}
.ca-badge-green { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
.ca-badge-amber { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.ca-badge-red { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.ca-badge-neutral { background: #eaeff6; color: #3b4a63; border-color: var(--ca-border); }
.ca-item-head { display: flex; justify-content: space-between; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.3rem; }
.ca-item-title { font-family: var(--ca-serif); font-size: 1.1rem; font-weight: 700; color: var(--ca-navy); }

/* Verbatim quotes */
.ca-label { font-size: 0.82rem; font-weight: 600; color: var(--ca-slate); margin: 0.5rem 0 0.2rem 0; }
.ca-quote {
    background: #f4f6f9; border-left: 4px solid var(--ca-blue); border-radius: 8px;
    padding: 0.75rem 1rem; white-space: pre-wrap; overflow-wrap: anywhere;
    font-size: 0.95rem; line-height: 1.55; color: var(--ca-navy);
}
.ca-src { color: var(--ca-slate); font-size: 0.82rem; margin-top: 0.25rem; }
.ca-gap {
    background: #fffbeb; border: 1px dashed #f59e0b; border-radius: 10px;
    padding: 0.7rem 0.95rem; color: #92400e; font-size: 0.95rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #fff; border: 1px solid var(--ca-border); border-radius: 14px; padding: 0.8rem 1rem;
}

/* Footer */
.ca-footer { color: var(--ca-slate); font-size: 0.86rem; line-height: 1.55; }

@media (max-width: 640px) {
    .ca-hero { padding: 1.8rem 1.3rem; }
    .ca-hero h1 { font-size: 1.9rem; }
    .ca-hero .ca-sub { font-size: 1.1rem; }
    .ca-card-desc { min-height: 0; }
    .ca-ph { flex-direction: column; align-items: flex-start; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
"""

# Per-card button / top-border colours (each workflow has its own accent)
_CARD_CSS = "<style>"
for _page, (_slug, _icon, _color, _d, _s) in PAGE_META.items():
    _CARD_CSS += (
        f'[data-testid="stVerticalBlockBorderWrapper"]:has(.ca-card-{_slug}) '
        f"{{ border-top: 5px solid {_color} !important; }}"
        f'[data-testid="stVerticalBlockBorderWrapper"]:has(.ca-card-{_slug}) .stButton button '
        f"{{ background: {_color}; border-color: {_color}; color: #fff; box-shadow: none; }}"
        f'[data-testid="stVerticalBlockBorderWrapper"]:has(.ca-card-{_slug}) .stButton button:hover '
        f"{{ filter: brightness(0.92); color: #fff; border-color: {_color}; }}"
    )
_CARD_CSS += "</style>"

st.markdown(_BASE_CSS, unsafe_allow_html=True)
st.markdown(_CARD_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Backend Data Helpers (unchanged deterministic pipeline)
# ---------------------------------------------------------------------

def _normalise_category(value: Any) -> str:
    """Return a normalized category name for conservative comparisons."""
    if value is None:
        return ""
    return str(value).strip().casefold()


def _build_summary_inputs(
    analysis: ContractAnalysis,
) -> tuple[list[dict[str, Any]], str]:
    """
    Build summary input from deterministic analysis.
    Evidence is verified against source clause text.
    """
    clauses = list(analysis.clauses)

    summary_clauses: list[dict[str, Any]] = [
        {
            "clause_id": str(clause.clause_id or ""),
            "page": clause.page_start,
            "text": str(clause.text or ""),
            "category": str(clause.category or ""),
        }
        for clause in clauses
    ]

    clauses_by_id: dict[str, list[Clause]] = {}
    for clause in clauses:
        clause_id = str(clause.clause_id or "").strip()
        clauses_by_id.setdefault(clause_id, []).append(clause)

    context_lines: list[str] = []

    # Preserve extracted facts
    for clause in clauses:
        for fact in clause.facts:
            context_lines.append(
                f"FACT | Clause: {clause.clause_id} | "
                f"Slot: {fact.slot} | Value: {fact.value} | "
                f"Confidence: {fact.confidence}"
            )

    # Preserve deterministic findings and verify evidence
    for finding in analysis.findings:
        finding_category = _normalise_category(finding.category)

        context_lines.append(
            f"FINDING | Category: {finding.category} | "
            f"Rating: {finding.rating} | "
            f"Evidence status: {finding.evidence_status} | "
            f"Reason: {finding.plain_reason}"
        )

        for evidence in finding.evidence:
            evidence_clause_id = str(evidence.clause_id or "").strip()
            evidence_quote = str(evidence.quote or "")

            if not evidence_clause_id or not evidence_quote:
                continue

            candidate_clauses = clauses_by_id.get(evidence_clause_id, [])
            verified_clause: Clause | None = None

            for clause in candidate_clauses:
                source_text = str(clause.text or "")
                clause_category = _normalise_category(clause.category)

                if clause_category != finding_category:
                    continue

                if evidence_quote not in source_text:
                    continue

                verified_clause = clause
                break

            if verified_clause is None:
                continue

            context_lines.append(
                f"EVIDENCE | Category: {finding.category} | "
                f"Clause: {evidence_clause_id} | "
                f"Quote: {evidence_quote}"
            )

    return summary_clauses, "\n".join(context_lines)


def _clear_summary_for_contract(doc_id: str) -> None:
    """Remove cached summaries for a contract after it is reprocessed."""
    st.session_state.pop(f"summary_requested_{doc_id}", None)
    prefixes = (
        f"summary_result_{doc_id}_",
        f"summary_result_{doc_id}",
    )
    for key in list(st.session_state.keys()):
        if any(key == prefix or key.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def _summary_cache_key(doc_id: str, language: str) -> str:
    """Keep summaries for different languages separate."""
    return f"summary_result_{doc_id}_{language.casefold()}"


def _process_contract_data(
    doc_id: str,
    source_type: str,
    file_obj=None,
    text_content: str = "",
    sample_choice: str = "",
) -> tuple[
    ExtractedDocument | None,
    list[Clause],
    ContractAnalysis | None,
]:
    """Execute extraction, segmentation, tagging, and deterministic analysis."""
    try:
        if source_type == "Sample":
            sample_path = Path("tests/sample_contracts") / sample_choice
            doc = load_document(sample_path, doc_id=doc_id)

        elif source_type == "Upload" and file_obj is not None:
            doc = load_document(
                file_obj.getvalue(),
                filename=file_obj.name,
                doc_id=doc_id,
            )

        elif source_type == "Paste" and text_content.strip():
            doc = load_pasted_text(
                text_content,
                doc_id=doc_id,
                filename=f"pasted_contract_{doc_id}.txt",
            )

        else:
            return None, [], None

        clauses = segment_document(doc)
        tag_clauses(clauses)
        analysis = analyse_contract(doc, clauses=clauses)

        return doc, clauses, analysis

    except Exception as exc:
        st.error(f"Error processing Contract {doc_id}: {exc}")
        return None, [], None


def _get_contract_facts_dict(analysis: ContractAnalysis) -> dict[str, Any]:
    """Collect key deterministic facts extracted across all clauses."""
    facts: dict[str, Any] = {}
    for clause in analysis.clauses:
        for fact in clause.facts:
            facts[fact.slot] = fact.value
    return facts


# ---------------------------------------------------------------------
# Small UI helpers (presentation only)
# ---------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _restore_widget(key: str) -> None:
    """Restore a widget value Streamlit dropped while its page was not on screen."""
    kept = st.session_state.get(f"_keep_{key}")
    if key not in st.session_state and kept is not None:
        st.session_state[key] = kept


def _remember_widget(key: str) -> None:
    """Remember a widget value so it survives switching pages."""
    if key in st.session_state:
        st.session_state[f"_keep_{key}"] = st.session_state[key]


def _badge(text: Any, kind: str = "neutral") -> str:
    return f"<span class='ca-badge ca-badge-{kind}'>{_esc(text)}</span>"


def _quote_block(quote: Any, source: str = "") -> None:
    st.markdown(f"<div class='ca-quote'>{_esc(quote)}</div>", unsafe_allow_html=True)
    if source:
        st.markdown(f"<div class='ca-src'>{_esc(source)}</div>", unsafe_allow_html=True)


def _page_header(page: str) -> None:
    """Back button plus colour-coded header for a workflow page."""
    slug, icon, color, _desc, subtitle = PAGE_META[page]
    st.button("← Home", key=f"back_home_{slug}", on_click=_go, args=(PAGE_HOME,))
    st.markdown(
        f"""
        <div class="ca-ph" style="--accent:{color};">
            <div class="ca-chip" style="background:{color}1A;">{icon}</div>
            <div>
                <div class="ca-ph-title">{_esc(page)}</div>
                <div class="ca-ph-desc">{_esc(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _empty_state(page: str) -> None:
    """Friendly prompt when a workflow needs a contract that is not loaded yet."""
    with st.container(border=True):
        st.markdown("**No contract loaded yet**")
        st.write("Please upload or select a contract in **Upload & Analyze** first.")
        st.button(
            "Go to Upload & Analyze",
            key=f"empty_go_upload_{PAGE_META[page][0]}",
            type="primary",
            on_click=_go,
            args=(PAGE_UPLOAD,),
        )


# ---------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------

doc_a_active: ExtractedDocument | None = st.session_state.get("doc_A")
doc_b_active: ExtractedDocument | None = st.session_state.get("doc_B")
analysis_a: ContractAnalysis | None = st.session_state.get("analysis_A")
analysis_b: ContractAnalysis | None = st.session_state.get("analysis_B")


def render_top_bar() -> None:
    brand_col, status_col, reset_col = st.columns([3, 3, 1])
    with brand_col:
        st.markdown(
            "<div class='ca-brand'><span class='ca-logo'>⚖️</span>Contract Autopsy Agent</div>",
            unsafe_allow_html=True,
        )
    with status_col:
        parts = []
        if doc_a_active:
            parts.append(f"<b>A:</b> {_esc(doc_a_active.filename[:18])}")
        if doc_b_active:
            parts.append(f"<b>B:</b> {_esc(doc_b_active.filename[:18])}")
        if parts:
            st.markdown(
                "<div style='padding-top:0.45rem;color:#475569;font-size:0.9rem;'>"
                + " &nbsp;|&nbsp; ".join(parts)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No contract loaded yet")
    with reset_col:
        st.button(
            "Reset All",
            use_container_width=True,
            key="btn_top_reset",
            help="Clear all loaded contracts and reset session memory",
            on_click=_reset_all,
        )

    st.radio(
        "Navigate",
        PAGES,
        key="nav_page",
        horizontal=True,
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------

def page_home() -> None:
    st.markdown(
        """
        <div class="ca-hero">
            <h1>Understand your contract. Protect your work.</h1>
            <div class="ca-sub">Know what you're agreeing to before you sign.</div>
            <div class="ca-desc">
                Contract Autopsy Agent helps Indian freelancers understand contract risks,
                identify important clauses, and compare offers. It is an informational tool
                and does not replace advice from a qualified lawyer.
            </div>
            <div class="ca-chips">
                <span class="ca-trust">Evidence quoted from your contract</span>
                <span class="ca-trust">English, Hindi, Telugu, Tamil</span>
                <span class="ca-trust">Processed in session memory only</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col, _ = st.columns([1, 3])
    with cta_col:
        st.button(
            "Analyze a contract",
            key="hero_cta",
            type="primary",
            use_container_width=True,
            on_click=_go,
            args=(PAGE_UPLOAD,),
        )

    if doc_a_active or doc_b_active:
        loaded = []
        if doc_a_active:
            loaded.append(f"Contract A: `{doc_a_active.filename}`")
        if doc_b_active:
            loaded.append(f"Contract B: `{doc_b_active.filename}`")
        st.success("Loaded in this session: " + "  |  ".join(loaded))

    st.markdown("<div class='ca-section-title'>Choose a workflow</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ca-section-sub'>Five tools, one goal: sign with confidence.</div>",
        unsafe_allow_html=True,
    )

    workflow_pages = [PAGE_UPLOAD, PAGE_RISKS, PAGE_EXPLAIN, PAGE_NEGOTIATE, PAGE_COMPARE]
    for row_start in range(0, len(workflow_pages), 3):
        cols = st.columns(3)
        for col, page in zip(cols, workflow_pages[row_start:row_start + 3]):
            slug, icon, color, desc, _sub = PAGE_META[page]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="ca-card ca-card-{slug}">
                            <div class="ca-chip" style="background:{color}1A;">{icon}</div>
                            <div class="ca-card-title">{_esc(page)}</div>
                            <div class="ca-card-desc">{_esc(desc)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"Open {page}",
                        key=f"card_{slug}",
                        use_container_width=True,
                        on_click=_go,
                        args=(page,),
                    )


# ---------------------------------------------------------------------
# SECTION 1: Upload & Analyze
# ---------------------------------------------------------------------

def _clause_expander(analysis: ContractAnalysis, slot_lower: str) -> None:
    with st.expander(f"View extracted clauses ({len(analysis.clauses)})", expanded=False):
        for clause in analysis.clauses:
            page_info = f"Page {clause.page_start}" if clause.page_start is not None else "Text"
            st.markdown(
                f"**[{clause.clause_id}] {clause.number or ''} {clause.heading or ''}** "
                f"({clause.category.upper()} | {page_info})"
            )
            st.text_area(
                "Verbatim clause",
                value=clause.text,
                height=80,
                disabled=True,
                key=f"clause_{slot_lower}_{clause.clause_id}",
            )


def page_upload() -> None:
    _page_header(PAGE_UPLOAD)

    col_up_a, col_up_b = st.columns(2)

    # Contract A
    with col_up_a:
        with st.container(border=True):
            st.markdown(
                "<div class='ca-panel'><span class='ca-dot' style='background:#3b82f6'></span>"
                "Primary Agreement (Contract A)</div>",
                unsafe_allow_html=True,
            )
            input_mode_a = st.radio(
                "Source for Contract A:",
                ["Synthetic Sample", "Upload PDF/DOCX", "Paste Text"],
                key="input_mode_a",
                horizontal=True,
            )

            if input_mode_a == "Synthetic Sample":
                samples_a = {
                    "contract_a.pdf": "Sample A: Standard Freelancer Contract (PDF)",
                    "contract_b.pdf": "Sample B: Client-Favourable / High Risk Contract (PDF)",
                    "contract_c.docx": "Sample C: Freelance Services Agreement (DOCX)",
                    "scanned.pdf": "Sample D: Scanned Document Demo (PDF)",
                }
                chosen_a = st.selectbox(
                    "Select sample contract:",
                    list(samples_a.keys()),
                    format_func=lambda x: samples_a[x],
                    key="sample_select_a",
                )
                if st.button("🚀 Load & Analyze Contract A", key="btn_load_a", type="primary", use_container_width=True):
                    with st.spinner("Analyzing Contract A..."):
                        doc, clauses, analysis = _process_contract_data(
                            "A", "Sample", sample_choice=chosen_a
                        )
                        if doc is not None and analysis is not None:
                            st.session_state["doc_A"] = doc
                            st.session_state["clauses_A"] = clauses
                            st.session_state["analysis_A"] = analysis
                            _clear_summary_for_contract("A")
                            st.success(f"Loaded {doc.filename} successfully!")
                            st.rerun()

            elif input_mode_a == "Upload PDF/DOCX":
                file_a = st.file_uploader(
                    "Upload file for Contract A:",
                    type=["pdf", "docx"],
                    key="uploader_a",
                )
                if file_a and st.button("🚀 Analyze Uploaded Contract A", key="btn_upload_a", type="primary", use_container_width=True):
                    with st.spinner("Extracting and analyzing Contract A..."):
                        doc, clauses, analysis = _process_contract_data(
                            "A", "Upload", file_obj=file_a
                        )
                        if doc is not None and analysis is not None:
                            st.session_state["doc_A"] = doc
                            st.session_state["clauses_A"] = clauses
                            st.session_state["analysis_A"] = analysis
                            _clear_summary_for_contract("A")
                            st.success(f"Processed {file_a.name} successfully!")
                            st.rerun()

            elif input_mode_a == "Paste Text":
                pasted_a = st.text_area(
                    "Paste contract text for Contract A:",
                    height=160,
                    key="pasted_text_a",
                    placeholder="Paste contract clauses here...",
                )
                if st.button("🚀 Analyze Pasted Text A", key="btn_paste_a", type="primary", use_container_width=True):
                    if pasted_a.strip():
                        with st.spinner("Analyzing pasted text..."):
                            doc, clauses, analysis = _process_contract_data(
                                "A", "Paste", text_content=pasted_a
                            )
                            if doc is not None and analysis is not None:
                                st.session_state["doc_A"] = doc
                                st.session_state["clauses_A"] = clauses
                                st.session_state["analysis_A"] = analysis
                                _clear_summary_for_contract("A")
                                st.success("Analyzed pasted contract successfully!")
                                st.rerun()
                    else:
                        st.warning("Please paste contract text before analyzing.")

            if doc_a_active is not None and analysis_a is not None:
                st.markdown("---")
                st.success(f"Analysis complete: `{doc_a_active.filename}`")
                m1, m2, m3 = st.columns(3)
                m1.metric("Format", doc_a_active.file_type.upper())
                m2.metric("Clauses", len(analysis_a.clauses))
                m3.metric("Characters", f"{len(doc_a_active.full_text):,}")

                for warning in doc_a_active.warnings:
                    st.warning(f"⚠️ {warning}")

                st.button(
                    "View risk analysis",
                    key="btn_view_risk_after_upload",
                    type="primary",
                    on_click=_go,
                    args=(PAGE_RISKS,),
                )
                _clause_expander(analysis_a, "a")

    # Contract B (optional)
    with col_up_b:
        with st.container(border=True):
            st.markdown(
                "<div class='ca-panel'><span class='ca-dot' style='background:#1b2a4a'></span>"
                "Comparison Agreement (Contract B, optional)</div>",
                unsafe_allow_html=True,
            )
            input_mode_b = st.radio(
                "Source for Contract B:",
                ["Synthetic Sample", "Upload PDF/DOCX", "Paste Text"],
                key="input_mode_b",
                horizontal=True,
            )

            if input_mode_b == "Synthetic Sample":
                samples_b = {
                    "contract_b.pdf": "Sample B: Client-Favourable / High Risk Contract (PDF)",
                    "contract_a.pdf": "Sample A: Standard Freelancer Contract (PDF)",
                    "contract_c.docx": "Sample C: Freelance Services Agreement (DOCX)",
                    "scanned.pdf": "Sample D: Scanned Document Demo (PDF)",
                }
                chosen_b = st.selectbox(
                    "Select comparison sample contract:",
                    list(samples_b.keys()),
                    format_func=lambda x: samples_b[x],
                    key="sample_select_b",
                )
                if st.button("🚀 Load & Analyze Contract B", key="btn_load_b", use_container_width=True):
                    with st.spinner("Analyzing Contract B..."):
                        doc, clauses, analysis = _process_contract_data(
                            "B", "Sample", sample_choice=chosen_b
                        )
                        if doc is not None and analysis is not None:
                            st.session_state["doc_B"] = doc
                            st.session_state["clauses_B"] = clauses
                            st.session_state["analysis_B"] = analysis
                            _clear_summary_for_contract("B")
                            st.success(f"Loaded {doc.filename} successfully!")
                            st.rerun()

            elif input_mode_b == "Upload PDF/DOCX":
                file_b = st.file_uploader(
                    "Upload file for Contract B:",
                    type=["pdf", "docx"],
                    key="uploader_b",
                )
                if file_b and st.button("🚀 Analyze Uploaded Contract B", key="btn_upload_b", use_container_width=True):
                    with st.spinner("Extracting and analyzing Contract B..."):
                        doc, clauses, analysis = _process_contract_data(
                            "B", "Upload", file_obj=file_b
                        )
                        if doc is not None and analysis is not None:
                            st.session_state["doc_B"] = doc
                            st.session_state["clauses_B"] = clauses
                            st.session_state["analysis_B"] = analysis
                            _clear_summary_for_contract("B")
                            st.success(f"Processed {file_b.name} successfully!")
                            st.rerun()

            elif input_mode_b == "Paste Text":
                pasted_b = st.text_area(
                    "Paste contract text for Contract B:",
                    height=160,
                    key="pasted_text_b",
                    placeholder="Paste contract clauses here...",
                )
                if st.button("🚀 Analyze Pasted Text B", key="btn_paste_b", use_container_width=True):
                    if pasted_b.strip():
                        with st.spinner("Analyzing pasted text..."):
                            doc, clauses, analysis = _process_contract_data(
                                "B", "Paste", text_content=pasted_b
                            )
                            if doc is not None and analysis is not None:
                                st.session_state["doc_B"] = doc
                                st.session_state["clauses_B"] = clauses
                                st.session_state["analysis_B"] = analysis
                                _clear_summary_for_contract("B")
                                st.success("Analyzed pasted contract successfully!")
                                st.rerun()
                    else:
                        st.warning("Please paste contract text before analyzing.")

            if doc_b_active is not None and analysis_b is not None:
                st.markdown("---")
                st.success(f"Analysis complete: `{doc_b_active.filename}`")
                m1, m2, m3 = st.columns(3)
                m1.metric("Format", doc_b_active.file_type.upper())
                m2.metric("Clauses", len(analysis_b.clauses))
                m3.metric("Characters", f"{len(doc_b_active.full_text):,}")

                for warning in doc_b_active.warnings:
                    st.warning(f"⚠️ {warning}")

                if analysis_a is not None:
                    st.button(
                        "Compare the two offers",
                        key="btn_go_compare_after_upload",
                        type="primary",
                        on_click=_go,
                        args=(PAGE_COMPARE,),
                    )
                _clause_expander(analysis_b, "b")


# ---------------------------------------------------------------------
# SECTION 2: Understand Your Risks
# ---------------------------------------------------------------------

def page_risks() -> None:
    _page_header(PAGE_RISKS)

    if not analysis_a and not analysis_b:
        _empty_state(PAGE_RISKS)
        return

    risk_contract_options = []
    if analysis_a:
        risk_contract_options.append("Contract A")
    if analysis_b:
        risk_contract_options.append("Contract B")

    selected_risk_contract = st.radio(
        "Select contract to inspect:",
        risk_contract_options,
        horizontal=True,
        key="risk_contract_selector",
    )
    active_risk_analysis = analysis_a if selected_risk_contract == "Contract A" else analysis_b

    if active_risk_analysis is None:
        return

    score = active_risk_analysis.overall_risk_score

    if score <= 30:
        score_badge = "🟢 LOW RISK"
        score_desc = "Favourable freelancer terms with balanced commitments."
    elif score <= 60:
        score_badge = "🟡 MODERATE RISK"
        score_desc = "Standard terms with several clauses that warrant review or negotiation."
    else:
        score_badge = "🔴 HIGH CONCERN"
        score_desc = "Aggressive client-favouring clauses or critical protections absent."

    with st.container(border=True):
        c_score1, c_score2, c_score3 = st.columns([1, 1, 2])
        with c_score1:
            st.metric(
                "Overall Risk Score",
                f"{score} / 100",
                delta=score_badge,
                delta_color="inverse",
            )
        with c_score2:
            st.metric("Findings", len(active_risk_analysis.findings))
        with c_score3:
            st.markdown(f"**Assessment:** {score_desc}")
            st.caption(
                "Methodology: Transparent rules evaluate payment deadlines, IP assignment, "
                "termination notice, and liability caps. Missing essential clauses receive "
                "mandatory caution penalties and are never marked green."
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    render_heatmap_summary(active_risk_analysis)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    render_findings_details(active_risk_analysis)


# ---------------------------------------------------------------------
# SECTION 3: Explain in My Language
# ---------------------------------------------------------------------

def page_explain() -> None:
    _page_header(PAGE_EXPLAIN)

    if not analysis_a and not analysis_b:
        _empty_state(PAGE_EXPLAIN)
        return

    contract_options = []
    if analysis_a:
        contract_options.append("Contract A")
    if analysis_b:
        contract_options.append("Contract B")

    selected_contract_label = st.radio(
        "Select contract to explain:",
        contract_options,
        horizontal=True,
        key="summary_contract_selector",
    )
    active_analysis = analysis_a if selected_contract_label == "Contract A" else analysis_b

    if active_analysis is None:
        return

    lang_key = f"lang_{active_analysis.doc_id}"
    consent_key = f"consent_{active_analysis.doc_id}"
    _restore_widget(lang_key)
    _restore_widget(consent_key)

    with st.container(border=True):
        col_lang, col_consent = st.columns([1, 2])

        with col_lang:
            language_map = {
                "English": "English",
                "Hindi": "Hindi (हिंदी)",
                "Telugu": "Telugu (తెలుగు)",
                "Tamil": "Tamil (தமிழ்)",
            }
            selected_lang = st.selectbox(
                "Choose Language:",
                list(language_map.keys()),
                format_func=lambda x: language_map[x],
                key=lang_key,
            )

        with col_consent:
            st.markdown("**🔒 Privacy Guardrail & AI Consent:**")
            st.caption(
                "Plain-language summaries require external AI processing via Google Gemini. "
                "No contract text is sent without your explicit consent. "
                "Deterministic risk analysis and comparisons do not require AI."
            )
            consent_given = st.checkbox(
                "I consent to sending contract text to Gemini for generating this summary.",
                key=consent_key,
            )

        # Truthful AI Provider Notice
        ai_ready = is_ai_available()
        if not ai_ready:
            st.warning(
                "⚠️ **AI Summary Generation Unavailable:** `GEMINI_API_KEY` is not configured in `.streamlit/secrets.toml` or `.env`. "
                "Plain-language summaries in English, Hindi, Telugu, and Tamil require an active AI provider. "
                "Deterministic risk analysis in **Understand Your Risks** and contract comparison in **Compare Two Offers** "
                "remain fully functional locally without an API key."
            )

        summary_cache_key = _summary_cache_key(active_analysis.doc_id, selected_lang)

        btn_disabled = not ai_ready
        if st.button(
            "✨ Generate Plain-Language Breakdown",
            key=f"btn_gen_summary_{active_analysis.doc_id}",
            type="primary",
            disabled=btn_disabled,
            help="Configure GEMINI_API_KEY and grant consent to enable" if btn_disabled else "Generate summary using Gemini AI",
        ):
            if not consent_given:
                st.warning("⚠️ Please provide explicit consent above before sending contract text to Gemini.")
            else:
                st.session_state[f"summary_requested_{active_analysis.doc_id}"] = True
                summary_clauses, facts_and_findings = _build_summary_inputs(active_analysis)
                with st.spinner(f"Generating summary in {selected_lang}..."):
                    try:
                        summary_res = summarize_contract(
                            clauses=summary_clauses,
                            facts_and_findings=facts_and_findings,
                            language=selected_lang,
                            consent_given=True,
                        )
                        st.session_state[summary_cache_key] = summary_res
                    except Exception as exc:
                        st.error(f"Could not generate summary: {exc}")

    _remember_widget(lang_key)
    _remember_widget(consent_key)

    summary_result = st.session_state.get(summary_cache_key)

    # Auto-regenerate when switching language (only if consent and AI are active)
    if summary_result is None and st.session_state.get(f"summary_requested_{active_analysis.doc_id}") and consent_given and ai_ready:
        summary_clauses, facts_and_findings = _build_summary_inputs(active_analysis)
        with st.spinner(f"Switching language to {selected_lang}..."):
            try:
                summary_res = summarize_contract(
                    clauses=summary_clauses,
                    facts_and_findings=facts_and_findings,
                    language=selected_lang,
                    consent_given=True,
                )
                st.session_state[summary_cache_key] = summary_res
                summary_result = summary_res
            except Exception as exc:
                st.error(f"Could not generate summary in {selected_lang}: {exc}")

    if not consent_given and summary_result is not None and summary_result.generated_by == "llm":
        st.session_state.pop(summary_cache_key, None)
        summary_result = None

    if summary_result is None:
        return

    st.markdown("---")

    if summary_result.generated_by == "llm":
        st.success("🤖 **AI-Generated Summary** (Schema-validated against contract text)")
    else:
        st.info("🛡️ **Deterministic Rule-Derived Facts** (Structured from contract analysis without external AI)")

    stated_items = [it for it in summary_result.items if it.status in ("Stated", "Inferred")]
    missing_items = [it for it in summary_result.items if it.status == "Missing"]
    unclear_items = [it for it in summary_result.items if it.status == "Unclear"]

    subtab_all, subtab_stated, subtab_missing, subtab_unclear = st.tabs(
        [
            f"All Key Terms ({len(summary_result.items)})",
            f"Stated Terms ({len(stated_items)})",
            f"Missing Protections ({len(missing_items)})",
            f"Unclear Terms ({len(unclear_items)})",
        ]
    )

    def _render_summary_list(items):
        if not items:
            st.caption("No points found in this section.")
            return

        for item in items:
            badge_kind = {
                "Stated": "green",
                "Inferred": "neutral",
                "Missing": "amber",
                "Unclear": "red",
            }.get(item.status, "neutral")

            category_label = translate_category(item.category, selected_lang)
            status_label = translate_status(item.status, selected_lang)

            with st.container(border=True):
                st.markdown(
                    f"<div class='ca-item-head'><span class='ca-item-title'>{_esc(category_label)}</span>"
                    f"{_badge(status_label, badge_kind)}</div>",
                    unsafe_allow_html=True,
                )
                st.write(item.explanation)
                if item.evidence_quote:
                    st.markdown("**Original verbatim quote from agreement:**")
                    st.code(item.evidence_quote, language="text")
                    page_str = f"Page {item.page}" if item.page is not None else "Text"
                    st.caption(f"Source: Clause `{item.clause_id}` | {page_str}")

    with subtab_all:
        _render_summary_list(summary_result.items)
    with subtab_stated:
        _render_summary_list(stated_items)
    with subtab_missing:
        _render_summary_list(missing_items)
    with subtab_unclear:
        _render_summary_list(unclear_items)

    if hasattr(summary_result, "disclaimer") and summary_result.disclaimer:
        st.caption(f"ℹ️ {summary_result.disclaimer}")


# ---------------------------------------------------------------------
# SECTION 4: Negotiate Better Terms
# ---------------------------------------------------------------------

def page_negotiate() -> None:
    _page_header(PAGE_NEGOTIATE)

    if not analysis_a and not analysis_b:
        _empty_state(PAGE_NEGOTIATE)
        return

    neg_contract_options = []
    if analysis_a:
        neg_contract_options.append("Contract A")
    if analysis_b:
        neg_contract_options.append("Contract B")

    selected_neg_contract = st.radio(
        "Select contract for negotiation preparation:",
        neg_contract_options,
        horizontal=True,
        key="neg_contract_selector",
    )
    active_neg_analysis = analysis_a if selected_neg_contract == "Contract A" else analysis_b

    if active_neg_analysis is None:
        return

    findings_list = active_neg_analysis.findings

    if not findings_list:
        st.info("No risk findings identified for this contract.")
        return

    finding_labels = {}
    for f in findings_list:
        cat_name = get_category_display_name(f.category)
        badge = RATING_ICONS.get(f.rating, f.rating)
        lbl = f"{cat_name} — {badge} ({f.plain_reason[:75]}...)"
        finding_labels[lbl] = f

    chosen_label = st.selectbox(
        "Choose an issue to address in negotiation:",
        list(finding_labels.keys()),
        key=f"neg_finding_{active_neg_analysis.doc_id}",
    )
    selected_finding = finding_labels[chosen_label]

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### 1. Current Contract Wording")
            if selected_finding.evidence:
                for ev in selected_finding.evidence:
                    page_str = f"Page {ev.page}" if ev.page is not None else "Text"
                    _quote_block(ev.quote, f"Clause {ev.clause_id} | {page_str}")
            else:
                st.markdown(
                    "<div class='ca-gap'>⚠️ This protection is currently missing from the contract.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### 2. Why Review This Term?")
            st.info(f"💡 {selected_finding.plain_reason}")

            if selected_finding.triggered_rules:
                rule_tags = [f"`{r.rule_id}` ({r.points:+d} pts)" for r in selected_finding.triggered_rules]
                st.caption(f"Triggered Rubric Rules: {' | '.join(rule_tags)}")

    with right:
        with st.container(border=True):
            st.markdown("#### 3. Suggested Replacement Clause")
            borrow_ref = st.session_state.get("borrow_ref_quote")
            if borrow_ref:
                st.info(f"📎 **Attached Reference Clause from Comparison Offer:**\n> \"{borrow_ref[:140]}...\"")
                if st.button("❌ Remove Attached Reference", key="btn_clear_borrow_ref"):
                    del st.session_state["borrow_ref_quote"]
                    st.rerun()

            if st.button(
                "✍️ Generate Counter-Clause Draft",
                key=f"btn_counter_{active_neg_analysis.doc_id}_{selected_finding.finding_id}",
                type="primary",
            ):
                try:
                    draft = generate_counter_clause(selected_finding, reference_quote=borrow_ref)
                    st.session_state["active_counter_draft"] = draft
                    st.session_state["active_counter_fid"] = selected_finding.finding_id
                except Exception as exc:
                    st.error(f"Error drafting counter-clause: {exc}")

            active_draft = st.session_state.get("active_counter_draft")
            active_fid = st.session_state.get("active_counter_fid")

            if active_draft is not None and active_fid == selected_finding.finding_id:
                if hasattr(active_draft, "applicability_note"):
                    st.caption(f"📌 {active_draft.applicability_note}")

                st.text_area(
                    "Edit or copy this replacement clause before sending to your client:",
                    value=active_draft.proposed_clause,
                    height=170,
                    key=f"editor_{active_neg_analysis.doc_id}_{selected_finding.finding_id}",
                )
            else:
                st.caption("Generate a draft to see alternative wording here.")

    active_draft = st.session_state.get("active_counter_draft")
    active_fid = st.session_state.get("active_counter_fid")
    if active_draft is not None and active_fid == selected_finding.finding_id:
        with st.container(border=True):
            st.markdown("#### 4. Points to Discuss With Client")
            for pt in active_draft.negotiation_points:
                st.markdown(f"- 💬 {pt}")

            st.markdown("#### 5. Legal Notice & Guidance")
            st.caption(active_draft.disclaimer)

    # 3-position framework (static guidance)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🎯 3-Position Negotiation Strategy Framework")
    st.caption(
        "General guidance (not generated from your contract). "
        "Use this 3-position boundary strategy to prepare before speaking with the client. "
        "*(Note: The Stage 4 Negotiation Engine dynamically computes clause-specific positions once configured. "
        "Use the active counter-clause draft above for your immediate negotiation wording).* "
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        with st.container(border=True):
            st.markdown("**Position 1: Target (Preferred)**")
            st.caption("Your ideal terms ensuring optimal protection, balanced commitments, and fair compensation.")
            st.markdown(
                "• **Objective:** Maximum freelancer safeguards\n"
                "• **Strategy:** State your standard preferred wording directly\n"
                "• **Rationale:** Aligns with industry standards for independent work"
            )
    with p_col2:
        with st.container(border=True):
            st.markdown("**Position 2: Compromise (Realistic)**")
            st.caption("A balanced fallback position that addresses client concerns while maintaining core protections.")
            st.markdown(
                "• **Objective:** Reasonable middle ground\n"
                "• **Strategy:** Offer procedural concessions without sacrificing rights\n"
                "• **Rationale:** Demonstrates flexibility while safeguarding core interests"
            )
    with p_col3:
        with st.container(border=True):
            st.markdown("**Position 3: Walk-Away Floor (Minimum Acceptable)**")
            st.caption("Your non-negotiable minimum threshold. If the client refuses this position, reassess or walk away.")
            st.markdown(
                "• **Objective:** Absolute downside protection\n"
                "• **Strategy:** Set firm, non-negotiable boundary criteria\n"
                "• **Rationale:** Prevents signing agreements that create unmanageable liabilities"
            )


# ---------------------------------------------------------------------
# SECTION 5: Compare Two Offers
# ---------------------------------------------------------------------

def page_compare() -> None:
    _page_header(PAGE_COMPARE)

    if not analysis_a and not analysis_b:
        st.info("👋 Please upload or select Contract A and Contract B in **Upload & Analyze**.")
        st.button(
            "Go to Upload & Analyze",
            key="cmp_go_upload_none",
            type="primary",
            on_click=_go,
            args=(PAGE_UPLOAD,),
        )
        return

    if not analysis_b:
        st.warning(
            "Contract A is loaded. Please load **Contract B** in **Upload & Analyze** "
            "to perform side-by-side comparison."
        )
        st.button(
            "Go to Upload & Analyze",
            key="cmp_go_upload_b",
            on_click=_go,
            args=(PAGE_UPLOAD,),
        )
        if st.button("⚡ Quick Load Sample B into Contract B", key="quick_load_sample_b"):
            with st.spinner("Loading Sample B for comparison..."):
                doc, clauses, analysis = _process_contract_data("B", "Sample", sample_choice="contract_b.pdf")
                if doc and analysis:
                    st.session_state["doc_B"] = doc
                    st.session_state["clauses_B"] = clauses
                    st.session_state["analysis_B"] = analysis
                    st.success("Sample B loaded as Contract B!")
                    st.rerun()
        return

    if not analysis_a:
        st.warning("Please load **Contract A** in **Upload & Analyze** to compare.")
        st.button(
            "Go to Upload & Analyze",
            key="cmp_go_upload_a",
            on_click=_go,
            args=(PAGE_UPLOAD,),
        )
        return

    doc_a = st.session_state.get("doc_A")
    doc_b = st.session_state.get("doc_B")

    col_id_a, col_id_b = st.columns(2)
    with col_id_a:
        with st.container(border=True):
            st.markdown(
                "<div class='ca-panel'><span class='ca-dot' style='background:#3b82f6'></span>Contract A</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"`{doc_a.filename if doc_a else 'Contract A'}`")
            st.caption(f"{len(analysis_a.clauses)} clauses | Overall Risk Score: **{analysis_a.overall_risk_score} / 100**")
    with col_id_b:
        with st.container(border=True):
            st.markdown(
                "<div class='ca-panel'><span class='ca-dot' style='background:#1b2a4a'></span>Contract B</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"`{doc_b.filename if doc_b else 'Contract B'}`")
            st.caption(f"{len(analysis_b.clauses)} clauses | Overall Risk Score: **{analysis_b.overall_risk_score} / 100**")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 1. Set Your Priorities (Importance Weights)")
    st.caption("Adjust the weight for each category: 0 = Ignore, 1 = Low, 2 = Medium (Default), 3 = High.")

    if "compare_weights" not in st.session_state:
        st.session_state["compare_weights"] = {
            "payment": 2,
            "ip": 2,
            "termination": 2,
            "scope": 2,
            "liability": 2,
            "confidentiality": 2,
        }

    priority_labels = {
        "payment": "💰 Payment",
        "ip": "🛡️ IP Ownership",
        "termination": "🚪 Termination",
        "scope": "📋 Scope",
        "liability": "⚖️ Liability",
        "confidentiality": "🔒 Confidentiality",
    }

    w_cols = st.columns(6)
    updated_weights: dict[str, int] = {}
    for idx, (cat_key, cat_label) in enumerate(priority_labels.items()):
        with w_cols[idx]:
            curr_w = st.session_state["compare_weights"].get(cat_key, 2)
            new_w = st.selectbox(
                cat_label,
                options=[0, 1, 2, 3],
                index=curr_w,
                format_func=lambda x: {0: "0 (Ignore)", 1: "1 (Low)", 2: "2 (Med)", 3: "3 (High)"}[x],
                key=f"weight_select_{cat_key}",
            )
            updated_weights[cat_key] = new_w

    st.session_state["compare_weights"] = updated_weights

    # Deterministic calculations
    fav_a = calculate_category_favourability(analysis_a)
    fav_b = calculate_category_favourability(analysis_b)
    scores_data = calculate_weighted_scores(fav_a, fav_b, updated_weights)
    comparison_rows = match_contracts(analysis_a, analysis_b)
    dealbreakers = find_dealbreakers(analysis_a, updated_weights, "Contract A") + find_dealbreakers(analysis_b, updated_weights, "Contract B")

    det_verdict = build_deterministic_verdict(
        scores_data=scores_data,
        fav_a=fav_a,
        fav_b=fav_b,
        rows=comparison_rows,
        dealbreakers=dealbreakers,
        weights=updated_weights,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 2. Weighted Score & Coverage Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    score_a = scores_data["score_a"]
    score_b = scores_data["score_b"]
    margin = scores_data["margin"]
    coverage = scores_data["coverage"]

    m_col1.metric("Contract A Score", f"{score_a:.1f} / 100")
    m_col2.metric("Contract B Score", f"{score_b:.1f} / 100")
    m_col3.metric("Score Margin (A - B)", f"{margin:+.1f} pts")
    m_col4.metric("Priority Coverage", f"{int(coverage * 100)}%")

    # Clamped coverage progress bar
    clamped_coverage = min(1.0, max(0.0, float(coverage)))
    st.progress(
        clamped_coverage,
        text=f"Coverage: {int(clamped_coverage * 100)}% of prioritized categories have comparable evidence (Threshold: {int(COVERAGE_THRESHOLD * 100)}%)",
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 3. Recommendation Verdict")

    if dealbreakers:
        st.error("🚨 **CRITICAL DEALBREAKER WARNINGS:**\n" + "\n".join(f"- {d}" for d in dealbreakers))

    active_verdict = st.session_state.get("comparison_ai_verdict", det_verdict)
    if st.session_state.get("last_verdict_weights") != updated_weights:
        active_verdict = det_verdict
        st.session_state["comparison_ai_verdict"] = det_verdict
        st.session_state["last_verdict_weights"] = dict(updated_weights)

    with st.container(border=True):
        status_emoji = {
            "leans_a": "🏆",
            "leans_b": "🏆",
            "no_clear_winner": "⚖️",
            "insufficient_evidence": "⚠️",
        }.get(active_verdict.status, "⚖️")

        st.markdown(f"### {status_emoji} {active_verdict.headline}")
        st.markdown(active_verdict.full_text)

        badge_text = "Generated by: Deterministic Rule Engine" if active_verdict.generated_by == "deterministic" else "Generated by: Gemini AI (Validated)"
        st.caption(f"📌 {badge_text}")

        with st.expander("✨ Plain-Language AI Rewrite (Readability Only)", expanded=False):
            st.caption(
                "The AI may rewrite the structured verdict into smoother prose for readability. "
                "It cannot alter the leaning, invent numbers, or modify terms."
            )
            ai_consent = st.checkbox(
                "I consent to sending contract summary text to Gemini for rewriting.",
                key="comp_ai_consent",
            )
            col_btn_rw, col_stat_rw = st.columns([1, 2])
            with col_btn_rw:
                if st.button("Generate AI Rewrite", key="btn_rewrite_verdict"):
                    if not ai_consent:
                        st.warning("Please check the consent box above.")
                    elif not is_ai_available():
                        st.warning("AI provider is not configured (`GEMINI_API_KEY` not set). Displaying deterministic template verdict.")
                        st.session_state["comparison_ai_verdict"] = det_verdict
                        st.rerun()
                    else:
                        with st.spinner("Rewriting verdict with Gemini..."):
                            rewritten = rewrite_verdict_with_ai(det_verdict, has_consent=True)
                            st.session_state["comparison_ai_verdict"] = rewritten
                            st.success("Verdict rewritten and validated!")
                            st.rerun()
            with col_stat_rw:
                if active_verdict.generated_by == "llm_rewritten":
                    if st.button("Reset to Deterministic Template", key="btn_reset_verdict"):
                        st.session_state["comparison_ai_verdict"] = det_verdict
                        st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 4. Side-by-Side Structured Terms Table")

    cat_filter = st.selectbox(
        "Filter comparison by category:",
        ["All Categories"] + [get_category_display_name(c) for c in ALL_PRIORITY_CATEGORIES],
        key="comp_table_filter",
    )

    table_data = []
    for r in comparison_rows:
        cat_display = get_category_display_name(r.category)
        if cat_filter != "All Categories" and cat_display != cat_filter:
            continue

        status_badge = {
            "matched": "🟢 Matched" if r.comparable else "⚠️ Incompatible",
            "missing_a": "🟡 Missing in A",
            "missing_b": "🟡 Missing in B",
            "missing_both": "⚪ Not in either",
            "not_comparable": "❌ Incompatible",
            "uncertain": "❓ Uncertain match",
        }.get(r.status, r.status)

        favours_badge = {
            "A": "👈 Favours A",
            "B": "👉 Favours B",
            "tie": "⚖️ Comparable",
            "none": "—",
        }.get(r.favours, "—")

        table_data.append({
            "Category": cat_display,
            "Contract Term": r.label,
            "Contract A": r.display_a,
            "Contract B": r.display_b,
            "Match Status": status_badge,
            "Favourability": favours_badge,
            "Note / Details": r.note or "—",
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)

    with st.expander("🔍 View Verbatim Source Quotes from Matched Slots", expanded=False):
        for r in comparison_rows:
            if r.quote_a or r.quote_b:
                st.markdown(f"**{r.label} ({get_category_display_name(r.category)})**")
                c_qa, c_qb = st.columns(2)
                with c_qa:
                    p_str_a = f" (Page {r.page_a})" if r.page_a else ""
                    st.caption(f"Contract A {r.clause_id_a or ''}{p_str_a}:")
                    st.text_area(f"Quote A ({r.slot})", value=r.quote_a or "Not found", height=70, disabled=True, key=f"qa_{r.slot}")
                with c_qb:
                    p_str_b = f" (Page {r.page_b})" if r.page_b else ""
                    st.caption(f"Contract B {r.clause_id_b or ''}{p_str_b}:")
                    st.text_area(f"Quote B ({r.slot})", value=r.quote_b or "Not found", height=70, disabled=True, key=f"qb_{r.slot}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 5. Cross-Contract Borrow Terms (Feature 7)")
    st.caption(
        "Where one contract offers substantially more favourable terms (>= 25 pts) or covers a missing protection, "
        "borrow that wording as a starting reference to negotiate the other contract."
    )

    borrow_suggestions = generate_borrow_suggestions(
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        rows=comparison_rows,
        weights=updated_weights,
    )

    if not borrow_suggestions:
        st.info("ℹ️ No supported borrow suggestions found based on current priority weights and contract terms.")
    else:
        for b_idx, sug in enumerate(borrow_suggestions):
            with st.container(border=True):
                tag = "⚡ Favourability Gap" if sug.suggestion_type == "favourability_gap" else "🛡️ Missing Clause Protection"
                st.markdown(f"**Suggestion #{b_idx + 1}: {tag} for {get_category_display_name(sug.category)}**")
                st.write(f"Borrow from **{sug.source_contract}** to improve **{sug.target_contract}**.")
                st.markdown(f"💡 *Why this helps:* {sug.reason}")

                loc = f" (Page {sug.page})" if sug.page else ""
                st.caption(f"Verbatim clause from {sug.source_contract} `{sug.clause_id or ''}`{loc}:")
                st.text_area(
                    "Verbatim source quotation",
                    value=sug.verbatim_quote,
                    height=80,
                    disabled=True,
                    key=f"borrow_quote_{b_idx}",
                )

                st.warning(f"⚠️ **Mandatory Caveat:** {sug.caveat}")

                if st.button("📋 Use as Reference in Negotiation", key=f"btn_borrow_use_{b_idx}"):
                    st.session_state["borrow_ref_quote"] = sug.verbatim_quote
                    st.success("Attached quotation to Negotiation workspace! Open **Negotiate Better Terms** to draft a counter-clause with this reference.")


# ---------------------------------------------------------------------
# Footer: diagnostics and legal notice (relocated from sidebar)
# ---------------------------------------------------------------------

def render_footer() -> None:
    st.divider()

    with st.expander("⚙️ Technical Details & Diagnostics", expanded=False):
        st.caption(f"App Version: `{APP_VERSION}` | Python Runtime: `{sys.version.split()[0]}`")

        ai_status_str = get_ai_status()
        if is_ai_available():
            st.success(f"AI Provider: **{ai_status_str}**")
        else:
            st.warning(f"AI Provider: **{ai_status_str}**")

        st.markdown(f"**Gemini Model:** `{GEMINI_MODEL}` | **API Key:** `{mask_key(GEMINI_API_KEY)}`")
        st.markdown(f"**Groq Backup Model:** `{GROQ_MODEL}`")
        st.markdown(f"**Similarity Threshold (BM25):** `{BM25_SIMILARITY_THRESHOLD}` | **Favourability Gap:** `{FAVOURABILITY_GAP_THRESHOLD}`")
        st.markdown("**OCR Dependency Status:**")
        st.caption(get_ocr_status_message())

        st.markdown("**Legal Knowledge Base (Stage 6):**")
        try:
            _kb_status = get_kb_status()
            st.caption(
                f"Reference records loaded: {_kb_status['total']} | "
                f"Verified for retrieval: {_kb_status['verified']}. "
                "Case law and regulatory guidance are not yet integrated."
            )
        except Exception:
            st.caption("Knowledge base status unavailable.")

    st.markdown(
        "<div class='ca-footer'>🔒 <b>Legal & Privacy Notice:</b> "
        + _esc(DISCLAIMER_TEXT)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------

render_top_bar()

_current_page = st.session_state.get("nav_page", PAGE_HOME)

if _current_page == PAGE_UPLOAD:
    page_upload()
elif _current_page == PAGE_RISKS:
    page_risks()
elif _current_page == PAGE_EXPLAIN:
    page_explain()
elif _current_page == PAGE_NEGOTIATE:
    page_negotiate()
elif _current_page == PAGE_COMPARE:
    page_compare()
else:
    page_home()

render_footer()