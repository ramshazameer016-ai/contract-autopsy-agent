"""
Contract Autopsy Agent - Streamlit Application Entry Point.
Clean, professional frontend designed for Indian freelancers and judges.
Features:
1. Upload Your Contract
2. Understand Your Contract (Plain language in EN, HI, TE, TA)
3. View Your Risk Map (Scorecard & explainable findings)
4. Prepare for Negotiation (Counter-clauses & talking points)
5. Compare Two Contracts (Side-by-side evaluation)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from src.ca.llm.client import clear_session_cache

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


# ---------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Contract Autopsy Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling for high readability and professional presentation
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-stated {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .badge-missing {
        background-color: #fff3e0;
        color: #e65100;
    }
    .badge-unclear {
        background-color: #fff8e1;
        color: #f57f17;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Backend data helpers (Unchanged deterministic logic)
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
# Sidebar - Friendly status & diagnostics
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚖️ Contract Autopsy")
    st.caption("AI-powered contract intelligence for Indian freelancers.")

    doc_a = st.session_state.get("doc_A")
    doc_b = st.session_state.get("doc_B")

    st.markdown("---")
    st.markdown("**Active Contracts:**")
    if doc_a:
        st.success(f"**Contract A:** `{doc_a.filename}`")
    else:
        st.info("Contract A: *Not loaded*")

    if doc_b:
        st.success(f"**Contract B:** `{doc_b.filename}`")
    else:
        st.caption("Contract B: *Not loaded (optional for comparison)*")

    st.markdown("---")

   
if st.button("🔄 Reset / Clear Session", use_container_width=True):
    clear_session_cache()
    st.session_state.clear()
    st.rerun()


    st.markdown("---")
    with st.expander("⚙️ Technical Details & Diagnostics", expanded=False):
        st.caption(f"App Version: `{APP_VERSION}`")
        st.markdown(f"**Python Runtime:** `{sys.version.split()[0]}`")

        ai_status_str = get_ai_status()
        if is_offline():
            st.warning(f"AI Provider: **{ai_status_str}**")
        else:
            st.success(f"AI Provider: **{ai_status_str}**")

        st.markdown(f"**Gemini Model:** `{GEMINI_MODEL}`")
        st.markdown(f"**Gemini API Key:** `{mask_key(GEMINI_API_KEY)}`")
        st.markdown(f"**Groq Backup Model:** `{GROQ_MODEL}`")
        st.markdown(f"**BM25 Threshold:** `{BM25_SIMILARITY_THRESHOLD}`")
        st.markdown(f"**Favourability Gap:** `{FAVOURABILITY_GAP_THRESHOLD}`")
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

    st.markdown("---")
    st.caption("🔒 **Privacy Notice:** Contract text is processed in memory only for this session. AI features require explicit consent before sending data to an external provider.")


# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------

st.title("⚖️ Contract Autopsy Agent")
st.subheader("Understand your contract before you sign.")
st.markdown(
    "Contract Autopsy Agent protects Indian freelancers by identifying unfair terms, "
    "spotting missing protections, explaining clauses in plain language (English, Hindi, Telugu, Tamil), "
    "and drafting ready-to-use counter-clauses."
)

tab_upload, tab_understand, tab_risk, tab_negotiate, tab_compare = st.tabs(
    [
        "📁 1. Upload Your Contract",
        "📖 2. Understand Your Contract",
        "📊 3. View Your Risk Map",
        "🤝 4. Prepare for Negotiation",
        "⚖️ 5. Compare Two Contracts",
    ]
)


# ---------------------------------------------------------------------
# TASK 1: Upload Your Contract
# ---------------------------------------------------------------------

with tab_upload:
    st.markdown("### 📁 Upload or Select Contracts")
    st.write(
        "Upload a contract as PDF or DOCX, paste text directly, or test immediately "
        "with one of our pre-built synthetic contracts."
    )

    col_upload_a, col_upload_b = st.columns(2)

    # Contract A Ingestion
    with col_upload_a:
        st.markdown("#### Primary Agreement (Contract A)")
        input_mode_a = st.radio(
            "Select input method for Contract A:",
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
                "Select a sample contract:",
                list(samples_a.keys()),
                format_func=lambda x: samples_a[x],
                key="sample_select_a",
            )
            if st.button("🚀 Load & Analyze Contract A", key="btn_load_a", use_container_width=True):
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
                "Upload Contract A file:",
                type=["pdf", "docx"],
                key="uploader_a",
            )
            if file_a and st.button("🚀 Analyze Uploaded Contract A", key="btn_upload_a", use_container_width=True):
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
                height=180,
                key="pasted_text_a",
                placeholder="Paste contract clauses here...",
            )
            if st.button("🚀 Analyze Pasted Text A", key="btn_paste_a", use_container_width=True):
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

        # Ingestion status card for Contract A
        doc_a_active = st.session_state.get("doc_A")
        clauses_a_active = st.session_state.get("clauses_A", [])
        if doc_a_active is not None:
            st.markdown("---")
            st.markdown(f"**Contract A Status:** `{doc_a_active.filename}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Format", doc_a_active.file_type.upper())
            m2.metric("Clauses", len(clauses_a_active))
            m3.metric("Characters", f"{len(doc_a_active.full_text):,}")

            for warning in doc_a_active.warnings:
                st.warning(f"⚠️ {warning}")

            with st.expander(f"📋 View Segmented Clauses ({len(clauses_a_active)})", expanded=False):
                for clause in clauses_a_active:
                    page_info = f"Page {clause.page_start}" if clause.page_start is not None else "Text"
                    st.markdown(f"**[{clause.clause_id}] {clause.number or ''} {clause.heading or ''}** ({clause.category.upper()} | {page_info})")
                    st.text_area(
                        "Verbatim clause",
                        value=clause.text,
                        height=90,
                        disabled=True,
                        key=f"clause_a_{clause.clause_id}",
                    )

    # Contract B Ingestion (For comparison)
    with col_upload_b:
        st.markdown("#### Comparison Agreement (Contract B - Optional)")
        input_mode_b = st.radio(
            "Select input method for Contract B:",
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
                "Select a comparison sample contract:",
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
                "Upload Contract B file:",
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
                height=180,
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

        # Ingestion status card for Contract B
        doc_b_active = st.session_state.get("doc_B")
        clauses_b_active = st.session_state.get("clauses_B", [])
        if doc_b_active is not None:
            st.markdown("---")
            st.markdown(f"**Contract B Status:** `{doc_b_active.filename}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Format", doc_b_active.file_type.upper())
            m2.metric("Clauses", len(clauses_b_active))
            m3.metric("Characters", f"{len(doc_b_active.full_text):,}")

            for warning in doc_b_active.warnings:
                st.warning(f"⚠️ {warning}")

            with st.expander(f"📋 View Segmented Clauses ({len(clauses_b_active)})", expanded=False):
                for clause in clauses_b_active:
                    page_info = f"Page {clause.page_start}" if clause.page_start is not None else "Text"
                    st.markdown(f"**[{clause.clause_id}] {clause.number or ''} {clause.heading or ''}** ({clause.category.upper()} | {page_info})")
                    st.text_area(
                        "Verbatim clause",
                        value=clause.text,
                        height=90,
                        disabled=True,
                        key=f"clause_b_{clause.clause_id}",
                    )


# ---------------------------------------------------------------------
# TASK 2: Understand Your Contract
# ---------------------------------------------------------------------

with tab_understand:
    st.markdown("### 📖 Understand Your Contract in Plain Language")
    st.write(
        "Complex legalese translated into simple, straightforward language for freelancers. "
        "Choose between English, Hindi, Telugu, or Tamil. Every claim is validated against contract quotes."
    )

    analysis_a: ContractAnalysis | None = st.session_state.get("analysis_A")
    analysis_b: ContractAnalysis | None = st.session_state.get("analysis_B")

    if not analysis_a and not analysis_b:
        st.info("👋 Please upload or load a contract in the **Upload Your Contract** tab first.")
    else:
        # Choose active contract
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

        if active_analysis is not None:
            col_lang, col_consent = st.columns([1, 2])

            with col_lang:
                language_map = {
                    "English": "English",
                    "Hindi": "Hindi (हिंदी)",
                    "Telugu": "Telugu (తెలుగు)",
                    "Tamil": "Tamil (தமிழ்)",
                }
                selected_lang = st.selectbox(
                    "Language for summary:",
                    list(language_map.keys()),
                    format_func=lambda x: language_map[x],
                    key=f"lang_{active_analysis.doc_id}",
                )

            with col_consent:
                st.markdown("**🔒 AI Privacy Notice & Consent:**")
                st.caption(
                    "When consent is given and live AI is enabled, contract text and findings are "
                    "processed with Google Gemini. Without consent, the app uses its verified local "
                    "template fallback without sending data over the network."
                )
                consent_given = st.checkbox(
                    "I consent to sending contract text to the configured AI provider for this summary.",
                    key=f"consent_{active_analysis.doc_id}",
                )

            summary_cache_key = _summary_cache_key(active_analysis.doc_id, selected_lang)

            if st.button("✨ Generate Plain-Language Summary", key=f"btn_gen_summary_{active_analysis.doc_id}"):
                st.session_state[f"summary_requested_{active_analysis.doc_id}"] = True
                summary_clauses, facts_and_findings = _build_summary_inputs(active_analysis)
                with st.spinner(f"Generating plain-language breakdown in {selected_lang}..."):
                    try:
                        summary_res = summarize_contract(
                            clauses=summary_clauses,
                            facts_and_findings=facts_and_findings,
                            language=selected_lang,
                            consent_given=consent_given,
                        )
                        st.session_state[summary_cache_key] = summary_res
                    except Exception as exc:
                        st.error(f"Could not generate summary: {exc}")

            summary_result = st.session_state.get(summary_cache_key)

            # Automatically regenerate when the user changes language after requesting a summary
            if summary_result is None and st.session_state.get(f"summary_requested_{active_analysis.doc_id}"):
                summary_clauses, facts_and_findings = _build_summary_inputs(active_analysis)
                with st.spinner(f"Switching language to {selected_lang}..."):
                    try:
                        summary_res = summarize_contract(
                            clauses=summary_clauses,
                            facts_and_findings=facts_and_findings,
                            language=selected_lang,
                            consent_given=consent_given,
                        )
                        st.session_state[summary_cache_key] = summary_res
                        summary_result = summary_res
                    except Exception as exc:
                        st.error(f"Could not generate summary in {selected_lang}: {exc}")

            # If user revokes consent, clear live LLM summary
            if not consent_given and summary_result is not None and summary_result.generated_by == "llm":
                st.session_state.pop(summary_cache_key, None)
                summary_result = None

            if summary_result is not None:
                st.markdown("---")

                # Honest Fallback Badge
                if summary_result.generated_by == "template":
                    st.info("🛡️ **Mode: Local Deterministic Template** (Generated from contract structure without sending data to an external AI provider)")
                else:
                    st.success("🤖 **Mode: AI-Generated & Schema-Validated** (Validated against source contract text)")

                # Organize items into Stated, Missing, and Unclear
                stated_items = [it for it in summary_result.items if it.status in ("Stated", "Inferred")]
                missing_items = [it for it in summary_result.items if it.status == "Missing"]
                unclear_items = [it for it in summary_result.items if it.status == "Unclear"]

                subtab_all, subtab_stated, subtab_missing, subtab_unclear = st.tabs(
                    [
                        f"All Points ({len(summary_result.items)})",
                        f"✅ Stated Terms ({len(stated_items)})",
                        f"⚠️ Missing Terms ({len(missing_items)})",
                        f"❓ Unclear Terms ({len(unclear_items)})",
                    ]
                )

                def _render_summary_list(items):
                    if not items:
                        st.caption("No points found in this section.")
                        return

                    for item in items:
                        badge_color = {
                            "Stated": "🟢",
                            "Inferred": "ℹ️",
                            "Missing": "⚠️",
                            "Unclear": "❓",
                        }.get(item.status, "⚪")

                        category_label = translate_category(item.category, selected_lang)
                        status_label = translate_status(item.status, selected_lang)
                        with st.expander(f"{badge_color} **{category_label}** — {status_label}", expanded=True):
                            st.write(item.explanation)
                            if item.evidence_quote:
                                st.markdown("**Original wording from contract (verbatim):**")
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
# TASK 3: View Your Risk Map
# ---------------------------------------------------------------------

with tab_risk:
    st.markdown("### 📊 Explainable Risk Map & Scorecard")
    st.write(
        "Deterministic risk ratings derived directly from transparent rules. "
        "Risk scores range from 0 (safest freelance terms) to 100 (highest risk / client-leaning). "
        "Missing essential clauses automatically incur caution points."
    )

    if not analysis_a and not analysis_b:
        st.info("👋 Please upload or load a contract in the **Upload Your Contract** tab first.")
    else:
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

        if active_risk_analysis is not None:
            # Overall Score Gauge
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

            c_score1, c_score2 = st.columns([1, 2])
            with c_score1:
                st.metric(
                    "Overall Contract Risk Score",
                    f"{score} / 100",
                    delta=score_badge,
                    delta_color="inverse",
                )
            with c_score2:
                st.markdown(f"**Interpretation:** {score_desc}")
                st.caption(
                    "Score methodology: Deterministic YAML rubric assigns points per triggered rule. "
                    "Absence of mandatory freelancer safeguards (payment terms, IP clarity, termination notice) "
                    "incurs a penalty so missing protections are never rated green."
                )

            st.markdown("---")
            render_heatmap_summary(active_risk_analysis)

            st.markdown("---")
            render_findings_details(active_risk_analysis)


# ---------------------------------------------------------------------
# TASK 4: Prepare for Negotiation
# ---------------------------------------------------------------------

with tab_negotiate:
    st.markdown("### 🤝 Prepare for Client Negotiation")
    st.write(
        "Turn risky clauses into fair, professional counter-clauses with concrete talking points. "
        "Generated drafts serve as negotiation starting points, not formal legal advice."
    )

    if not analysis_a and not analysis_b:
        st.info("👋 Please upload or load a contract in the **Upload Your Contract** tab first.")
    else:
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

        if active_neg_analysis is not None:
            # Filter findings that have high or moderate risk, or any findings
            findings_list = active_neg_analysis.findings

            if not findings_list:
                st.info("No risk findings identified for this contract.")
            else:
                # Finding selector
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

                # 5-step structured briefing
                st.markdown("---")
                st.markdown("#### 1. Current Contract Wording")
                if selected_finding.evidence:
                    for ev in selected_finding.evidence:
                        st.markdown(f"> *\"{ev.quote}\"*")
                        page_str = f"Page {ev.page}" if ev.page is not None else "Text"
                        st.caption(f"Clause {ev.clause_id} | {page_str}")
                else:
                    st.warning("⚠️ This protection is currently missing from the contract.")

                st.markdown("#### 2. Why Review This Term?")
                st.info(f"💡 {selected_finding.plain_reason}")
                if selected_finding.triggered_rules:
                    rule_tags = [f"`{r.rule_id}` ({r.points:+d} pts)" for r in selected_finding.triggered_rules]
                    st.caption(f"Triggered Rubric Rules: {' | '.join(rule_tags)}")

                st.markdown("#### 3. Suggested Replacement Clause")
                borrow_ref = st.session_state.get("borrow_ref_quote")
                if borrow_ref:
                    st.info(f"📎 **Attached Reference Clause from Comparison Offer:**\n> \"{borrow_ref[:140]}...\"")
                    if st.button("❌ Remove Attached Reference", key="btn_clear_borrow_ref"):
                        del st.session_state["borrow_ref_quote"]
                        st.rerun()

                if st.button("✍️ Generate Counter-Clause Draft", key=f"btn_counter_{active_neg_analysis.doc_id}_{selected_finding.finding_id}"):
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
                        height=140,
                        key=f"editor_{active_neg_analysis.doc_id}_{selected_finding.finding_id}",
                    )

                    st.markdown("#### 4. Points to Discuss With Client")
                    for pt in active_draft.negotiation_points:
                        st.markdown(f"- 💬 {pt}")

                    st.markdown("#### 5. Legal Notice & Guidance")
                    st.caption(active_draft.disclaimer)

                # Honest Status Note regarding Feature 5 (Negotiation Decision Tree)
                st.markdown("---")
                st.info(
                    "ℹ️ **Feature Status Note:** The full 3-position negotiation tree "
                    "(target, fallback, and walkaway boundary simulator) is currently in development. "
                    "Stage 4 counter-clause drafting with talking points and rationale is fully active above."
                )


# ---------------------------------------------------------------------
# TASK 5: Compare Two Contracts
# ---------------------------------------------------------------------

with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Contract Comparison")
    st.write(
        "Compare two contracts side-by-side to evaluate risk scores, category ratings, "
        "and key terms before choosing which client agreement to sign."
    )

    if not analysis_a and not analysis_b:
        st.info("👋 Please upload or select Contract A and Contract B in the **Upload Your Contract** tab.")
    elif not analysis_b:
        st.warning(
            "Contract A is loaded. Please load **Contract B** in the **Upload Your Contract** tab "
            "to perform a side-by-side comparison."
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
    elif not analysis_a:
        st.warning("Please load **Contract A** in the **Upload Your Contract** tab to compare.")
    else:
        # Both contracts are loaded!
        doc_a = st.session_state.get("doc_A")
        doc_b = st.session_state.get("doc_B")

        # Top Contract Identification Banner
        col_id_a, col_id_b = st.columns(2)
        with col_id_a:
            st.markdown(f"**Contract A:** `{doc_a.filename if doc_a else 'Contract A'}` ({len(analysis_a.clauses)} clauses, Risk {analysis_a.overall_risk_score}/100)")
        with col_id_b:
            st.markdown(f"**Contract B:** `{doc_b.filename if doc_b else 'Contract B'}` ({len(analysis_b.clauses)} clauses, Risk {analysis_b.overall_risk_score}/100)")

        st.markdown("---")
        # 1. User Priority Controls (Architecture G.2)
        st.markdown("#### 1. Set Your Priorities (Importance Weights)")
        st.caption("Adjust the importance of each contract aspect. 0 = Ignore, 1 = Low, 2 = Medium (Default), 3 = High.")

        # Ensure session state weights exist
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
            "payment": "💰 Payment Security",
            "ip": "🛡️ IP Ownership",
            "termination": "🚪 Termination",
            "scope": "📋 Scope Clarity",
            "liability": "⚖️ Liability Protection",
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
                    format_func=lambda x: {0: "0 (Ignore)", 1: "1 (Low)", 2: "2 (Medium)", 3: "3 (High)"}[x],
                    key=f"weight_select_{cat_key}",
                )
                updated_weights[cat_key] = new_w

        st.session_state["compare_weights"] = updated_weights

        # 2. Deterministic Calculation
        fav_a = calculate_category_favourability(analysis_a)
        fav_b = calculate_category_favourability(analysis_b)
        scores_data = calculate_weighted_scores(fav_a, fav_b, updated_weights)

        # Match slots
        comparison_rows = match_contracts(analysis_a, analysis_b)

        # Find dealbreakers (RED findings in weight 3 categories)
        dealbreakers = find_dealbreakers(analysis_a, updated_weights, "Contract A") + find_dealbreakers(analysis_b, updated_weights, "Contract B")

        # Build initial deterministic verdict
        det_verdict = build_deterministic_verdict(
            scores_data=scores_data,
            fav_a=fav_a,
            fav_b=fav_b,
            rows=comparison_rows,
            dealbreakers=dealbreakers,
            weights=updated_weights,
        )

        st.markdown("---")
        # 2. Overall Scores and Coverage Meter
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

        st.progress(coverage, text=f"Coverage Meter: {int(coverage * 100)}% of prioritized terms have evidence in both contracts (Threshold: {int(COVERAGE_THRESHOLD * 100)}%)")
        st.caption(f"ℹ️ Decision Rules: Insufficient evidence if coverage < {int(COVERAGE_THRESHOLD * 100)}% | No clear winner if |margin| < {MARGIN_THRESHOLD:.1f} pts.")

        st.markdown("---")
        # 3. Verdict Card and Dealbreaker Guard
        st.markdown("#### 3. Recommendation Verdict")

        # Display dealbreakers prominently if any
        if dealbreakers:
            st.error("🚨 **CRITICAL DEALBREAKER WARNINGS:**\n" + "\n".join(f"- {d}" for d in dealbreakers))

        active_verdict = st.session_state.get("comparison_ai_verdict", det_verdict)

        # If user changed weights, active_verdict updates deterministically
        if st.session_state.get("last_verdict_weights") != updated_weights:
            active_verdict = det_verdict
            st.session_state["comparison_ai_verdict"] = det_verdict
            st.session_state["last_verdict_weights"] = dict(updated_weights)

        # Verdict card container
        with st.container(border=True):
            status_emoji = {
                "leans_a": "🏆",
                "leans_b": "🏆",
                "no_clear_winner": "⚖️",
                "insufficient_evidence": "⚠️",
            }.get(active_verdict.status, "⚖️")

            st.markdown(f"### {status_emoji} {active_verdict.headline}")
            st.markdown(active_verdict.full_text)

            badge_text = "Mode: Deterministic Rubric" if active_verdict.generated_by == "deterministic" else "Mode: AI Rewritten (Validated)"
            st.caption(f"📌 {badge_text}")

            # Optional AI Rewrite Controls
            with st.expander("✨ Plain-Language AI Rewrite (Readability Only)", expanded=False):
                st.caption(
                    "The AI may rewrite the structured verdict into smoother prose for readability. "
                    "It is strictly forbidden from changing the leaning, inventing numbers, or altering terms."
                )
                ai_consent = st.checkbox(
                    "I consent to sending contract summary text to Google Gemini for rewriting.",
                    key="comp_ai_consent",
                )
                col_btn_rw, col_stat_rw = st.columns([1, 2])
                with col_btn_rw:
                    if st.button("Generate AI Rewrite", key="btn_rewrite_verdict"):
                        if not ai_consent:
                            st.warning("Please check the consent box above.")
                        elif is_offline():
                            st.warning(
                                "AI provider is not configured (GEMINI_API_KEY not set). "
                                "Displaying deterministic template verdict."
                            )
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

        st.markdown("---")
        # 4. Side-by-Side Structured Comparison Table
        st.markdown("#### 4. Side-by-Side Structured Terms Table")
        st.write("Examine extracted slot values, explicit gaps, comparability, and direction rules.")

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

        st.markdown("---")
        # 5. Cross-Contract Borrow Terms (Feature 7)
        st.markdown("#### 5. Terms You Could Ask For (Borrow Terms — Feature 7)")
        st.write(
            "Where one contract offers substantially more favourable terms (>= 25 pts) or covers a missing priority, "
            "you can borrow that wording as a starting reference to negotiate the other contract."
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
                        height=90,
                        disabled=True,
                        key=f"borrow_quote_{b_idx}",
                    )

                    st.warning(f"⚠️ **Mandatory Caveat:** {sug.caveat}")

                    if st.button("📋 Use as Reference in Negotiation", key=f"btn_borrow_use_{b_idx}"):
                        st.session_state["borrow_ref_quote"] = sug.verbatim_quote
                        st.success("Attached quotation to Negotiation workspace! Switch to the **Prepare for Negotiation** tab to generate a counter-clause with this reference term.")


# ---------------------------------------------------------------------
# Mandatory legal disclaimer footer
# ---------------------------------------------------------------------

st.divider()
st.caption(f"🔒 **Legal & Privacy Notice:** {DISCLAIMER_TEXT}")
