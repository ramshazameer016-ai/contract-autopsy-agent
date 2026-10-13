"""
Contract Autopsy Agent - Streamlit Application Entry Point.
Stage 2: Facts, Rule Engine, YAML Rubric, and Risk Heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

from config.settings import (
    AI_MODE,
    GEMINI_MODEL,
    GROQ_MODEL,
    GEMINI_API_KEY,
    FAVOURABILITY_GAP_THRESHOLD,
    BM25_SIMILARITY_THRESHOLD,
    DISCLAIMER_TEXT,
    APP_VERSION,
    mask_key,
)
from src.ca.extraction.loader import load_document, load_pasted_text
from src.ca.extraction.ocr_reader import get_ocr_status_message
from src.ca.segmentation.segmenter import segment_document
from src.ca.segmentation.tagger import tag_clauses
from src.ca.analysis.pipeline import analyse_contract
from src.ca.ui.components import render_heatmap_summary, render_findings_details
from src.ca.schemas import ExtractedDocument, Clause, ContractAnalysis

# Page configuration
st.set_page_config(
    page_title="Contract Autopsy Agent",
    page_icon="⚖️",
    layout="wide",
)

# Sidebar - Environment & Configuration Diagnostics (Stage 0 Preserved)
with st.sidebar:
    st.header("⚙️ Environment & Config")
    st.markdown(f"**Python Runtime:** `{sys.version.split()[0]}`")

    if AI_MODE == "offline":
        st.warning("Mode: **Honest Offline Mode** (No external AI calls)")
    else:
        st.success("Mode: **Live AI Mode**")

    st.markdown(f"**Gemini Model:** `{GEMINI_MODEL}`")
    st.markdown(f"**Gemini API Key:** `{mask_key(GEMINI_API_KEY)}`")
    st.markdown(f"**Groq Backup:** `{GROQ_MODEL}`")
    st.markdown(f"**BM25 Threshold:** `{BM25_SIMILARITY_THRESHOLD}`")
    st.markdown(f"**Favourability Gap:** `{FAVOURABILITY_GAP_THRESHOLD}`")

    st.divider()
    st.markdown("**OCR Status:**")
    st.caption(get_ocr_status_message())

    st.divider()
    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()

# Header
st.title("⚖️ Contract Autopsy Agent")
st.caption(f"Version: {APP_VERSION} | Stage 2: Facts, Rule Engine & Risk Heatmap")

# Main Interface Tabs
tab_upload, tab_analysis, tab_counter, tab_negotiate, tab_compare = st.tabs(
    ["📁 Upload & Ingestion", "🔍 Risk Analysis & Heatmap", "✍️ Counter-Clause", "🌳 Negotiate", "⚖️ Compare"]
)


def _process_contract_data(
    doc_id: str,
    source_type: str,
    file_obj=None,
    text_content: str = "",
    sample_choice: str = "",
) -> tuple[ExtractedDocument | None, list[Clause], ContractAnalysis | None]:
    """Uniform processing function executing extraction, segmentation, and Stage 2 analysis."""
    try:
        doc: ExtractedDocument
        if source_type == "Sample":
            sample_path = Path("tests/sample_contracts") / sample_choice
            doc = load_document(sample_path, doc_id=doc_id)
        elif source_type == "Upload" and file_obj is not None:
            doc = load_document(file_obj.getvalue(), filename=file_obj.name, doc_id=doc_id)
        elif source_type == "Paste" and text_content.strip():
            doc = load_pasted_text(text_content, doc_id=doc_id, filename=f"pasted_contract_{doc_id}.txt")
        else:
            return None, [], None

        clauses = segment_document(doc)
        tag_clauses(clauses)
        analysis = analyse_contract(doc, clauses=clauses)
        return doc, clauses, analysis
    except Exception as e:
        st.error(f"Error processing Contract {doc_id}: {e}")
        return None, [], None


def _render_contract_column(doc_id: str, label: str):
    """Render ingestion and segmented clause view for a single contract."""
    st.subheader(f"Contract {doc_id}: {label}")

    input_mode = st.radio(
        f"Source for Contract {doc_id}:",
        ["Synthetic Sample", "Upload File (PDF/DOCX)", "Paste Text"],
        key=f"mode_{doc_id}",
        horizontal=True,
    )

    if input_mode == "Synthetic Sample":
        samples = ["contract_a.pdf", "contract_b.pdf", "contract_c.docx", "scanned.pdf"]
        default_idx = 0 if doc_id == "A" else 1
        chosen_sample = st.selectbox(
            f"Select synthetic contract for Contract {doc_id}:",
            samples,
            index=default_idx,
            key=f"sample_{doc_id}",
        )
        if st.button(f"Load & Process Contract {doc_id}", key=f"btn_sample_{doc_id}"):
            doc, clauses, analysis = _process_contract_data(doc_id, "Sample", sample_choice=chosen_sample)
            st.session_state[f"doc_{doc_id}"] = doc
            st.session_state[f"clauses_{doc_id}"] = clauses
            st.session_state[f"analysis_{doc_id}"] = analysis

    elif input_mode == "Upload File (PDF/DOCX)":
        uploaded_file = st.file_uploader(
            f"Upload PDF or DOCX for Contract {doc_id}",
            type=["pdf", "docx"],
            key=f"upload_{doc_id}",
        )
        if uploaded_file and st.button(f"Process Uploaded File {doc_id}", key=f"btn_upload_{doc_id}"):
            doc, clauses, analysis = _process_contract_data(doc_id, "Upload", file_obj=uploaded_file)
            st.session_state[f"doc_{doc_id}"] = doc
            st.session_state[f"clauses_{doc_id}"] = clauses
            st.session_state[f"analysis_{doc_id}"] = analysis

    elif input_mode == "Paste Text":
        pasted_text = st.text_area(
            f"Paste contract text for Contract {doc_id}:",
            height=200,
            key=f"paste_{doc_id}",
            placeholder="Paste contract clauses here...",
        )
        if st.button(f"Process Pasted Text {doc_id}", key=f"btn_paste_{doc_id}"):
            doc, clauses, analysis = _process_contract_data(doc_id, "Paste", text_content=pasted_text)
            st.session_state[f"doc_{doc_id}"] = doc
            st.session_state[f"clauses_{doc_id}"] = clauses
            st.session_state[f"analysis_{doc_id}"] = analysis

    # Display results from session state if available
    doc: ExtractedDocument | None = st.session_state.get(f"doc_{doc_id}")
    clauses: list[Clause] = st.session_state.get(f"clauses_{doc_id}", [])

    if doc is not None:
        # Display extraction warnings (e.g. Scanned PDF warnings)
        for warn in doc.warnings:
            st.warning(f"⚠️ {warn}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File Type", doc.file_type.upper())
        col2.metric("Pages / Blocks", len(doc.pages))
        col3.metric("Characters", f"{len(doc.full_text):,}")
        col4.metric("Clauses Found", len(clauses))

        st.caption(f"**SHA-256:** `{doc.sha256[:16]}...` | **Filename:** `{doc.filename}`")

        if clauses:
            category_counts: dict[str, int] = {}
            for c in clauses:
                category_counts[c.category] = category_counts.get(c.category, 0) + 1

            badge_str = " | ".join(f"**{cat}**: {count}" for cat, count in sorted(category_counts.items()))
            st.markdown(f"🏷️ **Categories Tagged:** {badge_str}")

            with st.expander(f"📋 View Segmented Clauses ({len(clauses)})", expanded=False):
                for c in clauses:
                    cat_color = {
                        "payment": "🟢",
                        "ip": "🟣",
                        "termination": "🟠",
                        "scope": "🔵",
                        "liability": "🔴",
                        "confidentiality": "🟤",
                        "other": "⚪",
                    }.get(c.category, "⚪")

                    page_info = f"Page {c.page_start}" if c.page_start is not None else "Text"
                    if c.page_end and c.page_end != c.page_start:
                        page_info += f"-{c.page_end}"

                    header_label = f"{cat_color} [{c.clause_id}] {c.number or ''} {c.heading or ''} ({c.category.upper()} - {int(c.category_confidence*100)}% conf | {page_info})"

                    with st.expander(header_label, expanded=False):
                        st.markdown(f"**Offsets:** `{c.char_start}` to `{c.char_end}` ({len(c.text)} chars)")
                        st.text_area("Verbatim Clause Text", value=c.text, height=120, disabled=True, key=f"txt_{c.clause_id}")


with tab_upload:
    st.info("💡 Process Contract A and Contract B independently. Identical extraction and analysis pipeline is executed on both.")
    col_a, col_b = st.columns(2)
    with col_a:
        _render_contract_column("A", "First Agreement")
    with col_b:
        _render_contract_column("B", "Second Agreement")

with tab_analysis:
    st.subheader("🔍 Explainable Risk Analysis & Heatmap")

    analysis_a: ContractAnalysis | None = st.session_state.get("analysis_A")
    analysis_b: ContractAnalysis | None = st.session_state.get("analysis_B")

    if not analysis_a and not analysis_b:
        st.info("Please load or upload a contract in the **Upload & Ingestion** tab to see its risk analysis and heatmap.")
    else:
        available_contracts = []
        if analysis_a:
            available_contracts.append("Contract A")
        if analysis_b:
            available_contracts.append("Contract B")

        selected_view = st.radio("Select Contract to Inspect:", available_contracts, horizontal=True)
        active_analysis = analysis_a if selected_view == "Contract A" else analysis_b

        if active_analysis is not None:
            # Overall Risk Banner
            risk_color = (
                "#FF4B4B" if active_analysis.overall_risk_score >= 60
                else "#FFAA00" if active_analysis.overall_risk_score >= 35
                else "#09AB3B"
            )
            col_m1, col_m2 = st.columns([1, 3])
            with col_m1:
                st.metric("Overall Contract Risk Score", f"{active_analysis.overall_risk_score} / 100")
            with col_m2:
                st.caption(
                    "Risk score ranges from 0 (lowest risk, highly favourable terms) to 100 (highest risk, aggressive client terms). "
                    "Absence of mandatory categories incurs a caution penalty. Rating colors are derived deterministically from the YAML rubric."
                )

            st.divider()
            render_heatmap_summary(active_analysis)
            st.divider()
            render_findings_details(active_analysis)

with tab_counter:
    st.subheader("✍️ Counter-Clause Drafter (Stage 4)")
    st.info("Pending Stage 4.")

with tab_negotiate:
    st.subheader("🌳 Negotiation-Tree Simulator (Stage 4)")
    st.info("Pending Stage 4.")

with tab_compare:
    st.subheader("⚖️ Dual-Contract Comparison (Stage 5)")
    st.info("Pending Stage 5 (Slot Matching, Preference Weighting & Deterministic Verdict).")

# Mandatory Disclaimer Footer (F10 Privacy and Responsible Use)
st.divider()
st.caption(f"🔒 **Privacy & Legal Notice:** {DISCLAIMER_TEXT}")
