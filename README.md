# Contract Autopsy Agent

Contract Autopsy Agent is a dual-contract review and negotiation agent designed to analyze freelance, consulting, and service contracts against an explainable rubric, highlighting risks, comparing options, and suggesting counter-clauses.

> **Stage 0 (Foundation & Day-0 Checks)**: Setup, dependencies, configuration, synthetic sample contracts, and environment verification.

---

## 1. Setup Instructions (Windows PowerShell)

```powershell
# 1. Clone / Navigate to project directory
cd D:\downloads\contract_autopsy_agent

# 2. Verify Python version (3.11 or 3.12 recommended)
python --version

# 3. Create virtual environment (if not already created)
python -m venv .venv

# 4. Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
# If PowerShell execution policy blocks the script, run once in the active window:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 5. Install dependencies
pip install -r requirements.txt

# 6. Configure environment variables (optional for live AI mode)
Copy-Item .env.example .env
# Edit .env and paste your GEMINI_API_KEY from Google AI Studio
# If GEMINI_API_KEY is left empty, the application runs in honest Offline mode.

# 7. Launch Streamlit
streamlit run app.py
```

---

## 2. Limits and Operational Scope

- **Informational Screening**: Evaluates contract terms against deterministic heuristic rubrics and AI interpretation models.
- **Not Legal Advice**: Ratings (RED, YELLOW, GREEN) and counter-clause suggestions are intended for guidance and risk screening, not legally binding counsel.
- **Supported Formats**: Text-based PDF and DOCX files, plus direct text paste.
- **OCR Status**: RapidOCR / Tesseract are optional add-ons; if OCR is unavailable, the application provides an explicit paste-text fallback for scanned agreements.

---

## 3. Data Handling and Privacy Policy

- **Session-Only Storage**: Contracts are held in memory during the active Streamlit session and discarded upon session clear or disconnect.
- **No Disk Logging of Contracts**: Logs record execution events, durations, prompt versions, and error codes only. Verbatim contract text and secrets are never persisted in log files.
- **Offline Fallback**: In Offline mode (when no API key is provided or offline mode is toggled), the system performs deterministic rule checks and renders pre-defined templates without making external network calls.
- **Key Hygiene**: Secrets are loaded from `.env` or system environment variables; `.env` is strictly git-ignored.
