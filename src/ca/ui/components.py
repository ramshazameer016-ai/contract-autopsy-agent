
"""
UI components for Contract Autopsy Agent.
Provides explainable risk heatmap, evidence cards, rule explanations,
and absence chips for Streamlit. Stage 6 adds the Legal context expander.
"""

from __future__ import annotations

import streamlit as st
from src.ca.schemas import ContractAnalysis, RiskFinding, Rating

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "payment": "Payment terms",
    "ip": "Intellectual property",
    "termination": "Ending the contract",
    "scope": "Work and deliverables",
    "liability": "Responsibility and liability",
    "confidentiality": "Confidentiality",
}

RATING_COLORS: dict[str, str] = {
    "RED": "#FF4B4B",
    "YELLOW": "#FFAA00",
    "GREEN": "#09AB3B",
    "UNRATED": "#808495",
}

RATING_ICONS: dict[str, str] = {
    "RED": "🔴 RED — High concern",
    "YELLOW": "🟡 YELLOW — Needs review",
    "GREEN": "🟢 GREEN — Favourable indicator",
    "UNRATED": "⚪ UNRATED — Needs more information",
}

# Categories for which legal context retrieval is attempted (IP and Payment
# are the Stage 6 priority categories; others are supported if passages exist).
_KB_CATEGORIES = {"payment", "ip"}


def get_category_display_name(category_raw: str) -> str:
    """Return friendly category title."""
    return CATEGORY_DISPLAY_NAMES.get(category_raw.lower(), category_raw.title())


def render_heatmap_summary(analysis: ContractAnalysis):
    """Render a visual risk scorecard / heatmap of all contract categories."""
    st.markdown("### 📊 Category Risk Scorecard")

    categories = ["payment", "ip", "termination", "scope", "liability", "confidentiality"]
    cols = st.columns(len(categories))

    for idx, cat in enumerate(categories):
        with cols[idx]:
            rating = analysis.category_ratings.get(cat, "UNRATED")
            score = analysis.category_scores.get(cat, 0)
            cat_title = get_category_display_name(cat)

            # Check if category is absent
            cat_findings = [f for f in analysis.findings if f.category == cat]
            is_absent = any(f.evidence_status == "absent" for f in cat_findings)

            if is_absent:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255, 170, 0, 0.12); border: 2px dashed #FFAA00;
                                border-radius: 8px; padding: 12px 8px; text-align: center; min-height: 125px;">
                        <div style="font-weight: 600; font-size: 0.95rem; color: #FFAA00;">{cat_title}</div>
                        <div style="margin: 8px 0; font-weight: bold; color: #FFAA00; font-size: 0.85rem;">⚠️ Not found</div>
                        <div style="color: #666; font-size: 0.75rem;">Absence Penalty: {score} pts</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                bg_color = {
                    "RED": "rgba(255, 75, 75, 0.12)",
                    "YELLOW": "rgba(255, 170, 0, 0.12)",
                    "GREEN": "rgba(9, 171, 59, 0.12)",
                    "UNRATED": "rgba(128, 132, 149, 0.12)",
                }.get(rating, "#f0f2f6")

                border_color = RATING_COLORS.get(rating, "#808495")
                badge_label = RATING_ICONS.get(rating, rating)

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; border: 2px solid {border_color};
                                border-radius: 8px; padding: 12px 8px; text-align: center; min-height: 125px;">
                        <div style="font-weight: 600; font-size: 0.95rem; color: {border_color};">{cat_title}</div>
                        <div style="margin: 8px 0; font-weight: bold; color: {border_color}; font-size: 0.85rem;">{badge_label}</div>
                        <div style="color: #666; font-size: 0.75rem;">Risk Points: {score} pts</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_findings_details(analysis: ContractAnalysis):
    """Render explainable rule findings, verbatim evidence, and conflict warnings."""
    st.markdown("### 🔍 Detailed Risk Findings & Contract Evidence")

    categories = sorted(list({f.category for f in analysis.findings}))

    if not categories:
        st.info("No risk findings detected.")
        return

    for cat in categories:
        cat_findings = [f for f in analysis.findings if f.category == cat]
        cat_rating = analysis.category_ratings.get(cat, "UNRATED")
        cat_title = get_category_display_name(cat)
        badge = RATING_ICONS.get(cat_rating, cat_rating)

        with st.expander(f"📁 {cat_title} — {badge} ({len(cat_findings)} finding{'s' if len(cat_findings) != 1 else ''})", expanded=(cat_rating == "RED")):
            for f in cat_findings:
                _render_single_finding(f)


def _render_single_finding(finding: RiskFinding):
    """Render details of one finding with stable rule IDs, reason, quotes, and legal context."""
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        if finding.clause_id:
            st.markdown(f"**Clause {finding.clause_id}** — Status: `{finding.evidence_status.upper()}`")
        else:
            st.markdown("**Category Status: ⚠️ Not found in contract** (Absent protection)")

    with col_hdr2:
        rating_color = RATING_COLORS.get(finding.rating, "#808495")
        badge_text = RATING_ICONS.get(finding.rating, finding.rating)
        st.markdown(f"<span style='color:{rating_color}; font-weight:bold;'>{badge_text}</span> ({finding.score} pts)", unsafe_allow_html=True)

    # Plain reason
    st.info(f"💡 **Why review this:** {finding.plain_reason}")

    # Triggered rules
    if finding.triggered_rules:
        rule_tags = [f"`{r.rule_id}` ({r.severity}, {r.points:+d} pts)" for r in finding.triggered_rules]
        st.caption(f"**Triggered Rules:** {' | '.join(rule_tags)}")

    # Consistency / Conflicts
    if finding.evidence_status == "conflicting" and finding.conflicts_with:
        st.error(f"⚡ **Cross-Clause Conflict:** Values in this clause conflict with clause(s): {', '.join(finding.conflicts_with)}")

    # Contract Evidence Spans (verbatim from the uploaded contract)
    if finding.evidence:
        with st.expander("🔍 Contract evidence — original contract wording", expanded=False):
            st.caption("The following quotations are taken verbatim from the uploaded contract.")
            for ev in finding.evidence:
                st.markdown(f"> *\"{ev.quote}\"*")
                page_str = f"Page {ev.page}" if ev.page is not None else "Text"
                st.caption(f"Clause {ev.clause_id} | {page_str} | Offsets: `{ev.start}:{ev.end}`")
    elif finding.evidence_status == "absent":
        st.caption("No contract text found for this category. Rated with absence penalty.")

    # Legal context — Stage 6 KB retrieval (payment and IP only, verified passages)
    if finding.category in _KB_CATEGORIES:
        _render_legal_context(finding)

    st.divider()


def _render_legal_context(finding: RiskFinding) -> None:
    """Render the Stage 6 'Legal context' expander for a finding.

    Shows only verified passages.  Clearly distinguishes legal context from
    contract evidence.  Reports no-match honestly.
    Does not send data to an external AI service.
    """
    try:
        from src.ca.kb.retriever import retrieve, get_kb_status
    except ImportError:
        return  # KB module not available — silently skip

    try:
        result = retrieve(finding.category, verified_only=True)
    except Exception:
        return  # Retrieval error — silently skip rather than crash UI

    cat_display = get_category_display_name(finding.category)

    with st.expander(f"📚 Legal context — {cat_display}", expanded=False):
        st.caption(
            "ℹ️ **What is this?** Legal context passages provide background "
            "information on relevant Indian statutes. They are provided for "
            "reference only and do not constitute legal advice. "
            "**This is not a conclusion about your contract's compliance or "
            "enforceability.**"
        )
        st.caption(
            "⚠️ Distinguish between: "
            "**Contract evidence** (verbatim quotations from your uploaded contract, "
            "shown above) and **Legal context** (statutory reference material shown here)."
        )

        if result.has_results:
            for passage in result.passages:
                st.markdown(f"**{passage.source_title}** — *{passage.section}*")
                st.markdown(f"*Issuing authority:* {passage.authority}")
                if passage.official_url:
                    st.markdown(f"*Official source:* [{passage.official_url}]({passage.official_url})")
                date_parts: list[str] = []
                if passage.publication_date:
                    date_parts.append(f"Published: {passage.publication_date}")
                if passage.amendment_date:
                    date_parts.append(f"Amended: {passage.amendment_date}")
                if passage.date_accessed:
                    date_parts.append(f"Accessed: {passage.date_accessed}")
                if date_parts:
                    st.caption(" | ".join(date_parts))
                st.markdown("---")
                st.markdown(passage.text)
                if passage.limitations:
                    st.caption(f"📌 *Limitation:* {passage.limitations}")
                st.caption(
                    "This passage provides contextual background only. "
                    "Do not infer that your contract is compliant or enforceable "
                    "based on this statutory reference."
                )
        else:
            st.info(
                result.no_match_reason
                or "No verified legal context found for this finding. "
                   "The knowledge base currently has no verified passages "
                   "for this category."
            )
            try:
                status = get_kb_status()
                st.caption(
                    f"Knowledge base: {status['total']} reference record(s) loaded, "
                    f"{status['verified']} verified. "
                    "Case law and regulatory guidance are not yet integrated."
                )
            except Exception:
                pass

