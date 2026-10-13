"""
Settings and configuration management for Contract Autopsy Agent.
Reads environment variables from .env or system environment, handles AI mode,
model names, thresholds, and privacy guardrails.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file if it exists
load_dotenv(PROJECT_ROOT / ".env")


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from Streamlit secrets (if available) or os.environ."""
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# LLM API Keys & Providers
GEMINI_API_KEY: str = get_secret("GEMINI_API_KEY", "")
GROQ_API_KEY: str = get_secret("GROQ_API_KEY", "")

# Configurable Models (never hardcoded in application logic)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# AI Mode: forced offline if no API key is available
_env_ai_mode = os.getenv("AI_MODE", "").lower()
if not GEMINI_API_KEY:
    AI_MODE: str = "offline"
elif _env_ai_mode in ("live", "offline"):
    AI_MODE = _env_ai_mode
else:
    AI_MODE = "live"

# Thresholds (configurable assumptions from Architecture doc)
FAVOURABILITY_GAP_THRESHOLD: int = int(os.getenv("FAVOURABILITY_GAP_THRESHOLD", "25"))
BM25_SIMILARITY_THRESHOLD: float = float(os.getenv("BM25_SIMILARITY_THRESHOLD", "0.35"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Privacy & Disclaimer defaults
DATA_STORAGE_POLICY: str = "session_only"  # no disk/db persistence of uploaded contracts
APP_VERSION: str = "0.1.0-stage0"
DISCLAIMER_TEXT: str = (
    "Informational screening against a configurable rubric. Not legal advice. "
    "Ratings are not legally validated. Consult a qualified lawyer for important contracts."
)


def is_offline() -> bool:
    """Return True if system is running in offline mode."""
    return AI_MODE == "offline" or not bool(GEMINI_API_KEY)


def mask_key(key: str) -> str:
    """Return masked representation of an API key for safe diagnostics."""
    if not key:
        return "[NOT SET]"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
