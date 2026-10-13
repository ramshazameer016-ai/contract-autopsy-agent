"""
UI components for Contract Autopsy Agent.
Provides explainable risk heatmap, evidence cards, rule explanations,
and absence chips for Streamlit.
"""

from __future__ import annotations

import streamlit as st
from src.ca.schemas import ContractAnalysis, RiskFinding, Rating

RATING_COLORS = {
    "RED": "#FF4B4B",
    "YELLOW": "#FFAA00",
    "GREEN": "#09AB3B",
    "UNRATED": "#808495",
}

RATING_ICONS = {
    "RED": "🔴 RED (High Risk)",
    "YELLOW": "🟡 YELLOW (Caution / Ambiguous)",
    "GREEN": "🟢 GREEN (Favourable)",
    "UNRATED": "⚪ UNRATED (Review Required)",
}


def render_heatmap_summary(analysis: ContractAnalysis):
    """Render a visual risk scorecard / heatmap of all contract categories."""
    st.markdown("### 📊 Category Risk Heatmap")

    # Render category tiles in columns
    categories = ["payment", "ip", "termination", "scope", "liability", "confidentiality"]
    cols = st.columns(len(categories))

    for idx, cat in enumerate(categories):
        with cols[idx]:
            rating = analysis.category_ratings.get(cat, "UNRATED")
            score = analysis.category_scores.get(cat, 0)

            # Check if category is absent
            cat_findings = [f for f in analysis.findings if f.category == cat]
            is_absent = any(f.evidence_status == "absent" for f in cat_findings)

            cat_title = cat.upper()
            if is_absent:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255, 170, 0, 0.15); border: 2px dashed #FFAA00;
                                border-radius: 8px; padding: 10px; text-align: center;">
                        <h4 style="margin: 0; color: #FFAA00;">{cat_title}</h4>
                        <p style="margin: 5px 0; font-weight: bold; color: #FFAA00;">⚠️ Not found</p>
                        <small style="color: #666;">Absence Risk: {score} pts</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                bg_color = {
                    "RED": "rgba(255, 75, 75, 0.15)",
                    "YELLOW": "rgba(255, 170, 0, 0.15)",
                    "GREEN": "rgba(9, 171, 59, 0.15)",
                    "UNRATED": "rgba(128, 132, 149, 0.15)",
                }.get(rating, "#f0f2f6")

                border_color = RATING_COLORS.get(rating, "#808495")
                icon_label = {"RED": "🔴 RED", "YELLOW": "🟡 YELLOW", "GREEN": "🟢 GREEN", "UNRATED": "⚪ UNRATED"}.get(rating, rating)

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; border: 2px solid {border_color};
                                border-radius: 8px; padding: 10px; text-align: center;">
                        <h4 style="margin: 0; color: {border_color};">{cat_title}</h4>
                        <p style="margin: 5px 0; font-weight: bold; color: {border_color};">{icon_label}</p>
                        <small style="color: #666;">Risk: {score} pts</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_findings_details(analysis: ContractAnalysis):
    """Render explainable rule findings, verbatim evidence, and conflict warnings."""
    st.markdown("### 🔍 Explainable Risk Findings & Evidence")

    # Group findings by category
    categories = sorted(list({f.category for f in analysis.findings}))

    for cat in categories:
        cat_findings = [f for f in analysis.findings if f.category == cat]
        cat_rating = analysis.category_ratings.get(cat, "UNRATED")

        badge = RATING_ICONS.get(cat_rating, cat_rating)
        with st.expander(f"📁 {cat.upper()} — {badge} ({len(cat_findings)} findings)", expanded=(cat_rating == "RED")):
            for f in cat_findings:
                _render_single_finding(f)


def _render_single_finding(finding: RiskFinding):
    """Render details of one finding with stable rule IDs, reason, and quotes."""
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        if finding.clause_id:
            st.markdown(f"**Clause {finding.clause_id}** — Status: `{finding.evidence_status.upper()}`")
        else:
            st.markdown(f"**Category Status: ⚠️ Not found in contract** (Absent)")

    with col_hdr2:
        rating_color = RATING_COLORS.get(finding.rating, "#808495")
        st.markdown(f"<span style='color:{rating_color}; font-weight:bold;'>Rating: {finding.rating} ({finding.score} pts)</span>", unsafe_allow_html=True)

    # Plain reason
    st.info(f"💡 **Plain Explanation:** {finding.plain_reason}")

    # Triggered rules
    if finding.triggered_rules:
        rule_tags = [f"`{r.rule_id}` ({r.severity}, {r.points:+d} pts)" for r in finding.triggered_rules]
        st.markdown(f"**Triggered Rubric Rules:** {' | '.join(rule_tags)}")

    # Consistency / Conflicts
    if finding.evidence_status == "conflicting" and finding.conflicts_with:
        st.error(f"⚡ **Cross-Clause Conflict:** Values in this clause conflict with clause(s): {', '.join(finding.conflicts_with)}")

    # Verbatim Evidence Spans
    if finding.evidence:
        st.markdown("**Verbatim Contract Evidence:**")
        for ev in finding.evidence:
            st.markdown(f"> *\"{ev.quote}\"*  *(clause {ev.clause_id}, offsets {ev.start}:{ev.end})*")
    elif finding.evidence_status == "absent":
        st.caption("No contract text found for this category. Rated YELLOW with absence penalty.")

    st.divider()
