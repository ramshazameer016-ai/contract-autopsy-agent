
"""
Contract Autopsy Agent - Streamlit Application Entry Point.
Stages 0-3: Extraction, deterministic analysis, and plain-language summary.
Stage 4: Counter-Clause drafting UI.
"""

from __future__ import annotations

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
    DISCLAIMER_TEXT,
    APP_VERSION,
    mask_key,
)
from src.ca.extraction.loader import load_document, load_pasted_text
from src.ca.extraction.ocr_reader import get_ocr_status_message
from src.ca.segmentation.segmenter import segment_document
from src.ca.segmentation.tagger import tag_clauses
from src.ca.analysis.pipeline import analyse_contract
from src.ca.ui.components import (
    render_heatmap_summary,
    render_findings_details,
)
from src.ca.schemas import ExtractedDocument, Clause, ContractAnalysis
from src.ca.generation.summary import summarize_contract
from src.ca.generation.counter_clause import generate_counter_clause


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Contract Autopsy Agent",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------
# Helpers
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

    Evidence is included only when:
    1. The referenced clause exists.
    2. The exact quote occurs in that clause's text.
    3. The clause category matches the finding category.

    This safeguard does not change findings, ratings, or risk scores.
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

    # Preserve extracted facts.
    for clause in clauses:
        for fact in clause.facts:
            context_lines.append(
                f"FACT | Clause: {clause.clause_id} | "
                f"Slot: {fact.slot} | Value: {fact.value} | "
                f"Confidence: {fact.confidence}"
            )

    # Preserve deterministic findings and verify their evidence.
    for finding in analysis.findings:
        finding_category = _normalise_category(finding.category)

        context_lines.append(
            f"FINDING | Category: {finding.category} | "
            f"Rating: {finding.rating} | "
            f"Evidence status: {finding.evidence_status} | "
            f"Reason: {finding.plain_reason}"
        )

        for evidence in finding.evidence:
            evidence_clause_id = str(
                evidence.clause_id or ""
            ).strip()
            evidence_quote = str(evidence.quote or "")

            if not evidence_clause_id or not evidence_quote:
                continue

            candidate_clauses = clauses_by_id.get(
                evidence_clause_id, []
            )

            verified_clause: Clause | None = None

            for clause in candidate_clauses:
                source_text = str(clause.text or "")
                clause_category = _normalise_category(
                    clause.category
                )

                if clause_category != finding_category:
                    continue

                # Exact substring check: never invent or rewrite a quotation.
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
    prefixes = (
        f"summary_result_{doc_id}_",
        f"summary_result_{doc_id}",
    )

    for key in list(st.session_state.keys()):
        if any(
            key == prefix or key.startswith(prefix)
            for prefix in prefixes
        ):
            st.session_state.pop(key, None)


def _summary_cache_key(doc_id: str, language: str) -> str:
    """Keep summaries for different languages separate."""
    return f"summary_result_{doc_id}_{language.casefold()}"


def _render_summary_debug(
    summary_clauses: list[dict[str, Any]],
    facts_and_findings: str,
) -> None:
    """Display the exact summary inputs for debugging."""
    with st.expander("Debug: Summary input", expanded=False):
        evidence_lines = [
            line
            for line in facts_and_findings.splitlines()
            if line.startswith("EVIDENCE |")
        ]

        finding_lines = [
            line
            for line in facts_and_findings.splitlines()
            if line.startswith("FINDING |")
        ]

        st.write(f"Summary clauses supplied: {len(summary_clauses)}")
        st.write(
            f"Verified evidence lines supplied: {len(evidence_lines)}"
        )

        if evidence_lines:
            st.code("\n".join(evidence_lines), language="text")
        else:
            st.warning(
                "No verified evidence lines were passed to the summary "
                "generator. Check clause IDs, categories, and exact quotes."
            )

        with st.expander(
            "Deterministic findings supplied",
            expanded=False,
        ):
            if finding_lines:
                st.code("\n".join(finding_lines), language="text")
            else:
                st.caption(
                    "No deterministic finding lines were supplied."
                )


# ---------------------------------------------------------------------
# Sidebar - Environment & Configuration Diagnostics
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Environment & Config")
    st.markdown(f"**Python Runtime:** `{sys.version.split()[0]}`")

    if AI_MODE == "offline":
        st.warning(
            "Mode: **Honest Offline Mode** (No external AI calls)"
        )
    else:
        st.success("Mode: **Live AI Mode**")

    st.markdown(f"**Gemini Model:** `{GEMINI_MODEL}`")
    st.markdown(f"**Gemini API Key:** `{mask_key(GEMINI_API_KEY)}`")
    st.markdown(f"**Groq Backup:** `{GROQ_MODEL}`")
    st.markdown(
        f"**BM25 Threshold:** `{BM25_SIMILARITY_THRESHOLD}`"
    )
    st.markdown(
        f"**Favourability Gap:** `{FAVOURABILITY_GAP_THRESHOLD}`"
    )

    st.divider()
    st.markdown("**OCR Status:**")
    st.caption(get_ocr_status_message())

    st.divider()

    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()


# ---------------------------------------------------------------------
# Header and tabs
# ---------------------------------------------------------------------

st.title("⚖️ Contract Autopsy Agent")
st.caption(
    f"Version: {APP_VERSION} | "
    "Stages 0-3: Extraction, Rule Engine, Risk Heatmap & Summary"
)

tab_upload, tab_analysis, tab_counter, tab_negotiate, tab_compare = st.tabs(
    [
        "📁 Upload & Ingestion",
        "🔍 Risk Analysis & Heatmap",
        "✍️ Counter-Clause",
        "🌳 Negotiate",
        "⚖️ Compare",
    ]
)


# ---------------------------------------------------------------------
# Contract processing
# ---------------------------------------------------------------------

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
    """Execute extraction, segmentation, tagging, and analysis."""
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


# ---------------------------------------------------------------------
# Contract upload / ingestion UI
# ---------------------------------------------------------------------

def _render_contract_column(doc_id: str, label: str) -> None:
    """Render ingestion and segmented clause views for one contract."""
    st.subheader(f"Contract {doc_id}: {label}")

    input_mode = st.radio(
        f"Source for Contract {doc_id}:",
        ["Synthetic Sample", "Upload File (PDF/DOCX)", "Paste Text"],
        key=f"mode_{doc_id}",
        horizontal=True,
    )

    if input_mode == "Synthetic Sample":
        samples = [
            "contract_a.pdf",
            "contract_b.pdf",
            "contract_c.docx",
            "scanned.pdf",
        ]
        default_idx = 0 if doc_id == "A" else 1

        chosen_sample = st.selectbox(
            f"Select synthetic contract for Contract {doc_id}:",
            samples,
            index=default_idx,
            key=f"sample_{doc_id}",
        )

        if st.button(
            f"Load & Process Contract {doc_id}",
            key=f"btn_sample_{doc_id}",
        ):
            doc, clauses, analysis = _process_contract_data(
                doc_id,
                "Sample",
                sample_choice=chosen_sample,
            )

            if doc is not None and analysis is not None:
                st.session_state[f"doc_{doc_id}"] = doc
                st.session_state[f"clauses_{doc_id}"] = clauses
                st.session_state[f"analysis_{doc_id}"] = analysis
                _clear_summary_for_contract(doc_id)

    elif input_mode == "Upload File (PDF/DOCX)":
        uploaded_file = st.file_uploader(
            f"Upload PDF or DOCX for Contract {doc_id}",
            type=["pdf", "docx"],
            key=f"upload_{doc_id}",
        )

        if uploaded_file and st.button(
            f"Process Uploaded File {doc_id}",
            key=f"btn_upload_{doc_id}",
        ):
            doc, clauses, analysis = _process_contract_data(
                doc_id,
                "Upload",
                file_obj=uploaded_file,
            )

            if doc is not None and analysis is not None:
                st.session_state[f"doc_{doc_id}"] = doc
                st.session_state[f"clauses_{doc_id}"] = clauses
                st.session_state[f"analysis_{doc_id}"] = analysis
                _clear_summary_for_contract(doc_id)

    elif input_mode == "Paste Text":
        pasted_text = st.text_area(
            f"Paste contract text for Contract {doc_id}",
            height=200,
            key=f"paste_{doc_id}",
            placeholder="Paste contract clauses here...",
        )

        if st.button(
            f"Process Pasted Text {doc_id}",
            key=f"btn_paste_{doc_id}",
        ):
            doc, clauses, analysis = _process_contract_data(
                doc_id,
                "Paste",
                text_content=pasted_text,
            )

            if doc is not None and analysis is not None:
                st.session_state[f"doc_{doc_id}"] = doc
                st.session_state[f"clauses_{doc_id}"] = clauses
                st.session_state[f"analysis_{doc_id}"] = analysis
                _clear_summary_for_contract(doc_id)

    # Display previously processed results.
    doc: ExtractedDocument | None = st.session_state.get(
        f"doc_{doc_id}"
    )
    clauses: list[Clause] = st.session_state.get(
        f"clauses_{doc_id}", []
    )

    if doc is not None:
        for warning in doc.warnings:
            st.warning(f"⚠️ {warning}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File Type", doc.file_type.upper())
        col2.metric("Pages / Blocks", len(doc.pages))
        col3.metric("Characters", f"{len(doc.full_text):,}")
        col4.metric("Clauses Found", len(clauses))

        st.caption(
            f"**SHA-256:** `{doc.sha256[:16]}...` | "
            f"**Filename:** `{doc.filename}`"
        )

        if clauses:
            category_counts: dict[str, int] = {}

            for clause in clauses:
                category_counts[clause.category] = (
                    category_counts.get(clause.category, 0) + 1
                )

            badge_str = " | ".join(
                f"**{category}**: {count}"
                for category, count in sorted(category_counts.items())
            )
            st.markdown(f"🏷️ **Categories Tagged:** {badge_str}")

            with st.expander(
                f"📋 View Segmented Clauses ({len(clauses)})",
                expanded=False,
            ):
                for clause in clauses:
                    cat_color = {
                        "payment": "🟢",
                        "ip": "🟣",
                        "termination": "🟠",
                        "scope": "🔵",
                        "liability": "🔴",
                        "confidentiality": "🟤",
                        "other": "⚪",
                    }.get(clause.category, "⚪")

                    page_info = (
                        f"Page {clause.page_start}"
                        if clause.page_start is not None
                        else "Text"
                    )

                    if (
                        clause.page_end
                        and clause.page_end != clause.page_start
                    ):
                        page_info += f"-{clause.page_end}"

                    header_label = (
                        f"{cat_color} [{clause.clause_id}] "
                        f"{clause.number or ''} {clause.heading or ''} "
                        f"({clause.category.upper()} - "
                        f"{int(clause.category_confidence * 100)}% conf | "
                        f"{page_info})"
                    )

                    with st.expander(header_label, expanded=False):
                        st.markdown(
                            f"**Offsets:** `{clause.char_start}` to "
                            f"`{clause.char_end}` "
                            f"({len(clause.text)} chars)"
                        )
                        st.text_area(
                            "Verbatim Clause Text",
                            value=clause.text,
                            height=120,
                            disabled=True,
                            key=f"txt_{clause.clause_id}",
                        )


with tab_upload:
    st.info(
        "💡 Process Contract A and Contract B independently. "
        "Identical extraction and analysis pipeline is executed on both."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        _render_contract_column("A", "First Agreement")

    with col_b:
        _render_contract_column("B", "Second Agreement")


# ---------------------------------------------------------------------
# Risk analysis and plain-language summary
# ---------------------------------------------------------------------

with tab_analysis:
    st.subheader("🔍 Explainable Risk Analysis & Heatmap")

    analysis_a: ContractAnalysis | None = st.session_state.get("analysis_A")
    analysis_b: ContractAnalysis | None = st.session_state.get("analysis_B")

    if not analysis_a and not analysis_b:
        st.info(
            "Please load or upload a contract in the **Upload & Ingestion** "
            "tab to see its risk analysis and heatmap."
        )

    else:
        available_contracts: list[str] = []

        if analysis_a:
            available_contracts.append("Contract A")

        if analysis_b:
            available_contracts.append("Contract B")

        selected_view = st.radio(
            "Select Contract to Inspect:",
            available_contracts,
            horizontal=True,
        )

        active_analysis = (
            analysis_a if selected_view == "Contract A" else analysis_b
        )

        if active_analysis is not None:
            col_m1, col_m2 = st.columns([1, 3])

            with col_m1:
                st.metric(
                    "Overall Contract Risk Score",
                    f"{active_analysis.overall_risk_score} / 100",
                )

            with col_m2:
                st.caption(
                    "Risk score ranges from 0 (lowest risk, highly favourable "
                    "terms) to 100 (highest risk, aggressive client terms). "
                    "Absence of mandatory categories incurs a caution penalty. "
                    "Rating colors are derived deterministically from the "
                    "YAML rubric."
                )

            st.divider()
            render_heatmap_summary(active_analysis)
            st.divider()
            render_findings_details(active_analysis)

            # ---------------------------------------------------------
            # Stage 3: Plain-language summary
            # ---------------------------------------------------------

            st.divider()
            st.subheader("🧾 Plain-language Contract Summary")

            st.caption(
                "This summary explains existing extracted clauses and "
                "deterministic findings. It does not replace the original "
                "contract or change risk scores."
            )

            summary_language = st.selectbox(
                "Summary language",
                ["English", "Hindi", "Telugu", "Tamil"],
                key=f"summary_language_{active_analysis.doc_id}",
            )

            st.info(
                "Privacy notice: If you give consent below and live AI is "
                "available, contract text and findings may be sent to "
                "Google's Gemini service. Do not submit confidential or "
                "sensitive contracts unless you are authorised to share "
                "them with that provider. Without consent, the app uses "
                "its local template fallback."
            )

            consent_given = st.checkbox(
                "I understand the disclosure and consent to sending contract "
                "text and findings to the configured AI provider for this "
                "summary.",
                key=f"summary_consent_{active_analysis.doc_id}",
            )

            summary_key = _summary_cache_key(
                active_analysis.doc_id,
                summary_language,
            )

            if st.button(
                "Generate Plain-language Summary",
                key=f"generate_summary_{active_analysis.doc_id}",
            ):
                summary_clauses, facts_and_findings = _build_summary_inputs(
                    active_analysis
                )

                _render_summary_debug(
                    summary_clauses,
                    facts_and_findings,
                )

                with st.spinner("Preparing the summary..."):
                    try:
                        result = summarize_contract(
                            clauses=summary_clauses,
                            facts_and_findings=facts_and_findings,
                            language=summary_language,
                            consent_given=consent_given,
                        )

                        st.session_state[summary_key] = result

                    except Exception as exc:
                        st.error(
                            "The summary could not be generated. "
                            "Your deterministic risk analysis remains available."
                        )
                        st.caption(f"Error type: {type(exc).__name__}")

            summary_result = st.session_state.get(summary_key)

            # Do not display a previously generated live AI summary if
            # the user withdraws consent.
            if (
                not consent_given
                and summary_result is not None
                and summary_result.generated_by == "llm"
            ):
                st.session_state.pop(summary_key, None)
                summary_result = None

            if summary_result is not None:
                if summary_result.generated_by == "template":
                    st.caption("Summary mode: Local template fallback")
                else:
                    st.caption(
                        "Summary mode: AI-generated; validated against the schema"
                    )

                for item in summary_result.items:
                    with st.expander(
                        f"{item.category} — {item.status}",
                        expanded=True,
                    ):
                        st.write(item.explanation)

                        if item.evidence_quote:
                            st.markdown("**Evidence from the contract**")
                            st.code(item.evidence_quote)

                        if item.clause_id:
                            st.caption(f"Clause ID: {item.clause_id}")

                        if item.page is not None:
                            st.caption(f"Page: {item.page}")


# ---------------------------------------------------------------------
# Stage 4: Counter-Clause Drafter
# ---------------------------------------------------------------------

with tab_counter:
    st.subheader("Counter-Clause Drafter (Stage 4)")

    st.caption(
        "Generate a negotiation starting point from an existing "
        "deterministic risk finding. Drafts are not legal advice."
    )

    available_analyses: dict[str, ContractAnalysis] = {}

    for doc_id, label in [
        ("A", "Contract A"),
        ("B", "Contract B"),
    ]:
        analysis = st.session_state.get(f"analysis_{doc_id}")
        if analysis is not None:
            available_analyses[label] = analysis

    if not available_analyses:
        st.info(
            "First process a contract in the Upload & Ingestion tab, "
            "then return here to draft a counter-clause."
        )

    else:
        selected_contract = st.selectbox(
            "Choose a contract",
            list(available_analyses.keys()),
            key="counter_contract",
        )

        active_counter_analysis = available_analyses[selected_contract]

        if not active_counter_analysis.findings:
            st.warning("No risk findings are available for this contract.")

        else:
            finding_options = {
                (
                    f"{finding.category.upper()} | "
                    f"{finding.rating} | "
                    f"{finding.finding_id} | "
                    f"{finding.plain_reason[:90]}"
                ): finding
                for finding in active_counter_analysis.findings
            }

            selected_finding_label = st.selectbox(
                "Choose the finding to address",
                list(finding_options.keys()),
                key=f"counter_finding_{active_counter_analysis.doc_id}",
            )

            selected_finding = finding_options[selected_finding_label]

            st.markdown("### Original risk finding")
            col1, col2, col3 = st.columns(3)

            col1.metric("Original rating", selected_finding.rating)
            col2.metric(
                "Evidence status",
                selected_finding.evidence_status,
            )
            col3.metric(
                "Risk score",
                (
                    str(selected_finding.score)
                    if selected_finding.score is not None
                    else "Not scored"
                ),
            )

            st.write("**Reason identified by the rule engine**")
            st.write(selected_finding.plain_reason)

            if selected_finding.evidence:
                st.write("**Evidence from the original contract**")

                for evidence in selected_finding.evidence:
                    if evidence.quote:
                        st.code(evidence.quote, language="text")
                        st.caption(
                            f"Clause: {evidence.clause_id} | "
                            f"Page: {evidence.page if evidence.page is not None else 'Not available'}"
                        )
            else:
                st.warning(
                    "No supporting quotation is attached to this finding. "
                    "Review the original contract before considering any draft."
                )

            st.divider()

            if st.button(
                "Generate Counter-Clause Draft",
                key=f"generate_counter_{active_counter_analysis.doc_id}",
            ):
                try:
                    draft = generate_counter_clause(selected_finding)
                    st.session_state["counter_clause_draft"] = draft
                    st.session_state["counter_clause_finding_id"] = (
                        selected_finding.finding_id
                    )
                    st.session_state["counter_clause_doc_id"] = (
                        active_counter_analysis.doc_id
                    )

                except ValueError as exc:
                    st.warning(str(exc))

                except Exception as exc:
                    st.error(
                        "The counter-clause draft could not be generated. "
                        f"Error type: {type(exc).__name__}"
                    )

            draft = st.session_state.get("counter_clause_draft")
            draft_finding_id = st.session_state.get(
                "counter_clause_finding_id"
            )
            draft_doc_id = st.session_state.get("counter_clause_doc_id")

            if (
                draft is not None
                and draft_finding_id == selected_finding.finding_id
                and draft_doc_id == active_counter_analysis.doc_id
            ):
                st.markdown("### Proposed replacement clause")

                st.warning(
                    "This is a template proposal, not a legally validated "
                    "solution. It does not change the original risk rating."
                )

                # Stage 4 safety improvement: explain whether the generic
                # template is suitable to consider based on finding metadata.
                if hasattr(draft, "applicability_note"):
                    st.info(draft.applicability_note)
                else:
                    st.info(
                        "This is a generic category-level template. "
                        "It has not been tailored to the original clause. "
                        "Compare both texts before considering any replacement."
                    )

                st.markdown("**Proposed clause text**")
                st.code(draft.proposed_clause, language="text")

                st.text_area(
                    "Draft clause — review and edit before use",
                    value=draft.proposed_clause,
                    height=220,
                    key=(
                        f"counter_draft_editor_"
                        f"{active_counter_analysis.doc_id}_"
                        f"{selected_finding.finding_id}"
                    ),
                )

                st.markdown("### Negotiation points")

                for point in draft.negotiation_points:
                    st.markdown(f"- {point}")

                st.caption(draft.disclaimer)

                st.caption(
                    f"Source finding: {draft.finding_id} | "
                    f"Original rating: {draft.original_rating} | "
                    f"Generator: {draft.generated_by}"
                )


# ---------------------------------------------------------------------
# Stage 4: Negotiation Simulator placeholder
# ---------------------------------------------------------------------

with tab_negotiate:
    st.subheader("🌳 Negotiation-Tree Simulator (Stage 4)")
    st.info("Pending Stage 4.")


# ---------------------------------------------------------------------
# Stage 5: Dual-Contract Comparison placeholder
# ---------------------------------------------------------------------

with tab_compare:
    st.subheader("⚖️ Dual-Contract Comparison (Stage 5)")
    st.info(
        "Pending Stage 5 (Slot Matching, Preference Weighting & "
        "Deterministic Verdict)."
    )


# ---------------------------------------------------------------------
# Mandatory disclaimer footer
# ---------------------------------------------------------------------

st.divider()
st.caption(f"🔒 **Privacy & Legal Notice:** {DISCLAIMER_TEXT}")
