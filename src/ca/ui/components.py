"""
UI components for Contract Autopsy Agent.
Provides explainable risk heatmap, evidence cards, rule explanations,
and absence chips for Streamlit.
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
    """Render details of one finding with stable rule IDs, reason, and quotes."""
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

    # Verbatim Evidence Spans
    if finding.evidence:
        with st.expander("🔍 View original contract wording", expanded=False):
            for ev in finding.evidence:
                st.markdown(f"> *\"{ev.quote}\"*")
                page_str = f"Page {ev.page}" if ev.page is not None else "Text"
                st.caption(f"Clause {ev.clause_id} | {page_str} | Offsets: `{ev.start}:{ev.end}`")
    elif finding.evidence_status == "absent":
        st.caption("No contract text found for this category. Rated with absence penalty.")

    st.divider()
