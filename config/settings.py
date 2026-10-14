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
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# AI Provider & Mode.
# AI features require a configured AI provider (GEMINI_API_KEY).
# AI_MODE "offline" is preserved for backward compatibility with existing tests.
AI_PROVIDER: str = "gemini"
AI_MODE: str = "live" if bool(GEMINI_API_KEY) else "offline"


# Thresholds (configurable assumptions from Architecture doc)
FAVOURABILITY_GAP_THRESHOLD: int = int(os.getenv("FAVOURABILITY_GAP_THRESHOLD", "25"))
BM25_SIMILARITY_THRESHOLD: float = float(os.getenv("BM25_SIMILARITY_THRESHOLD", "0.35"))
COVERAGE_THRESHOLD: float = float(os.getenv("COVERAGE_THRESHOLD", "0.5"))
MARGIN_THRESHOLD: float = float(os.getenv("MARGIN_THRESHOLD", "8.0"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Privacy & Disclaimer defaults
DATA_STORAGE_POLICY: str = "session_only"  # no disk/db persistence of uploaded contracts
APP_VERSION: str = "0.1.0-stage0"
DISCLAIMER_TEXT: str = (
    "Informational screening against a configurable rubric. Not legal advice. "
    "Ratings are not legally validated. Consult a qualified lawyer for important contracts."
)


def is_ai_available() -> bool:
    """Return True if an AI provider is configured with a valid API key."""
    return bool(GEMINI_API_KEY.strip()) if GEMINI_API_KEY else False


def is_offline() -> bool:
    """Returns True when no AI provider API key is configured.

    Preserved for backward compatibility with Stage 3/5 tests.
    Use is_ai_available() in new code.
    """
    return not is_ai_available()


def get_ai_status() -> str:
    """Return an honest, user-readable provider status string.

    Never exposes key values.
    Examples:
      "Gemini configured (gemini-2.5-flash)"
      "AI unavailable — GEMINI_API_KEY not set"
    """
    if is_ai_available():
        return f"Gemini configured ({GEMINI_MODEL})"
    return "AI unavailable — GEMINI_API_KEY not set"


def mask_key(key: str) -> str:
    """Return masked representation of an API key for safe diagnostics."""
    if not key:
        return "[NOT SET]"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
