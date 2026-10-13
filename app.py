"""
Contract Autopsy Agent - Streamlit Application Entry Point.
Stage 0: Setup and Day-0 Verification Hello Page.
"""

import sys
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

# Page configuration
st.set_page_state = "expanded"
st.set_page_config(
    page_title="Contract Autopsy Agent (Stage 0)",
    page_icon="⚖️",
    layout="wide",
)

# Header
st.title("⚖️ Contract Autopsy Agent")
st.caption(f"Version: {APP_VERSION} | Stage 0: Setup & Day-0 Checks")

# Sidebar - Environment & Configuration Diagnostics
with st.sidebar:
    st.header("⚙️ Environment & Config")
    st.markdown(f"**Python Runtime:** `{sys.version.split()[0]}`")
    
    if AI_MODE == "offline":
        st.warning("Mode: **Honest Offline Mode** (No external AI calls)")
    else:
        st.success(f"Mode: **Live AI Mode**")

    st.markdown(f"**Gemini Model:** `{GEMINI_MODEL}`")
    st.markdown(f"**Gemini API Key:** `{mask_key(GEMINI_API_KEY)}`")
    st.markdown(f"**Groq Model (Backup):** `{GROQ_MODEL}`")
    st.markdown(f"**BM25 Threshold:** `{BM25_SIMILARITY_THRESHOLD}`")
    st.markdown(f"**Favourability Gap:** `{FAVOURABILITY_GAP_THRESHOLD}`")

    st.divider()
    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()

# Stage 0 Welcome and Verification Status
st.success("✅ **Stage 0 Setup Verified:** Environment, dependencies, and configuration loaded successfully.")

# Tabs as specified in Architecture Section D (placeholders for upcoming stages)
tab_upload, tab_analysis, tab_counter, tab_negotiate, tab_compare = st.tabs(
    ["📁 Upload", "🔍 Analysis", "✍️ Counter-Clause", "🌳 Negotiate", "⚖️ Compare"]
)

with tab_upload:
    st.subheader("Document Ingestion (Stage 1)")
    st.info("Document upload (PDF, DOCX, Paste) and clause segmentation will be activated in Stage 1.")

with tab_analysis:
    st.subheader("Risk Heatmap & Explanations (Stage 2 & 3)")
    st.info("Explainable YAML rubric evaluation and plain-language summaries will be activated in Stages 2 & 3.")

with tab_counter:
    st.subheader("Counter-Clause Drafter (Stage 4)")
    st.info("Evidence-grounded counter-clause generation will be activated in Stage 4.")

with tab_negotiate:
    st.subheader("Negotiation-Tree Simulator (Stage 4)")
    st.info("Boundary forms and 3-position negotiation tree will be activated in Stage 4.")

with tab_compare:
    st.subheader("Dual-Contract Comparison (Stage 5)")
    st.info("Slot matching, trade-off detection, and deterministic verdict engine will be activated in Stage 5.")

# Mandatory Disclaimer Footer (F10 Privacy and Responsible Use)
st.divider()
st.caption(f"🔒 **Privacy & Legal Notice:** {DISCLAIMER_TEXT}")
