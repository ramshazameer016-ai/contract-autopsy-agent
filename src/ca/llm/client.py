
"""Safe Gemini gateway with validated responses and privacy protection."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    is_ai_available,
    is_offline,
)

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# In-memory cache only. Prompts and contract text are never written to disk.
_CACHE: dict[str, str] = {}


class LLMUnavailable(RuntimeError):
    """Raised when the AI provider is unavailable or returns invalid data."""


class ConsentRequiredError(RuntimeError):
    """Raised when an AI operation is attempted without explicit user consent."""


def _safe_provider_error(exc: Exception) -> str:
    """Return a safe provider diagnostic without exposing request data."""
    error_name = type(exc).__name__
    status = getattr(exc, "code", None)

    if status is None:
        status = getattr(exc, "status_code", None)

    # Some SDK errors expose an HTTP status through response.status_code.
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    details = f"{error_name}"
    if isinstance(status, (int, str)):
        details += f", HTTP/status={status}"

    # Avoid logging exception messages, which may contain sensitive details.
    logger.warning("Gemini provider request failed: %s", details)
    return details


def _call_gemini(prompt: str) -> str:
    """Make one Gemini request. Call only after explicit user consent."""
    if not GEMINI_API_KEY:
        raise LLMUnavailable(
            "Gemini is not configured. Check GEMINI_API_KEY in your settings."
        )

    try:
        from google import genai
    except ImportError:
        raise LLMUnavailable(
            "The Gemini SDK is missing. Install the project's required dependencies."
        ) from None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:
        # Do not expose the raw exception, API key, prompt, or contract text.
        details = _safe_provider_error(exc)
        raise LLMUnavailable(
            f"Gemini request failed ({details}). Check API-key validity, "
            "model availability, API access, quota, and network connectivity."
        ) from None

    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise LLMUnavailable(
            "Gemini returned an empty response. Try again or check model availability."
        )

    return response_text.strip()


def generate_validated(
    prompt: str,
    schema: type[T],
    *,
    prompt_version: str,
) -> T:
    """Generate structured output and validate it against a Pydantic schema.

    This function does not itself establish consent. The calling workflow must
    check explicit consent before invoking it.
    """

     # Add this check here, before checking the API key or cache.
    if is_offline():
        raise LLMUnavailable(
            "AI provider is unavailable while the application is offline."
        )

    if not GEMINI_API_KEY:
        raise LLMUnavailable(
            "Gemini is not configured. Check GEMINI_API_KEY in your settings."
        )

    if not GEMINI_API_KEY:
        raise LLMUnavailable(
            "Gemini is not configured. Check GEMINI_API_KEY in your settings."
        )

    cache_key = hashlib.sha256(
        f"{prompt_version}\0{schema.__name__}\0{prompt}".encode("utf-8")
    ).hexdigest()

    # Cache contains only previously schema-validated responses.
    if cache_key in _CACHE:
        try:
            return schema.model_validate_json(_CACHE[cache_key])
        except (ValidationError, ValueError):
            # Discard invalid/stale cached data and regenerate.
            _CACHE.pop(cache_key, None)

    correction = ""
    last_error = "No response"

    # Initial request plus up to two correction/retry attempts.
    for attempt in range(3):
        try:
            raw = _call_gemini(prompt + correction)

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = f"Invalid JSON (line {exc.lineno}, column {exc.colno})"
                correction = (
                    "\nYour previous response was not valid JSON. "
                    "Return corrected JSON only, matching the requested schema."
                )
            else:
                try:
                    parsed = schema.model_validate(payload)
                except ValidationError as exc:
                    # Do not include response contents or contract text in logs.
                    error_types = sorted(
                        {
                            str(item.get("type", "validation_error"))
                            for item in exc.errors()
                        }
                    )
                    last_error = "Schema validation failed: " + ", ".join(
                        error_types[:8]
                    )
                    correction = (
                        "\nYour previous JSON failed schema validation. "
                        f"Error categories: {last_error}. "
                        "Return corrected JSON only, matching the requested schema."
                    )
                else:
                    # Cache only validated output.
                    _CACHE[cache_key] = parsed.model_dump_json()
                    return parsed

        except LLMUnavailable:
            # Configuration, SDK, and provider errors need actionable handling.
            # Do not retry authentication or quota failures blindly.
            raise

        except Exception as exc:
            # Unexpected internal failure: log only its type.
            last_error = f"Unexpected internal error ({type(exc).__name__})"
            logger.warning("%s", last_error)
            correction = (
                "\nReturn valid JSON only, matching the requested schema."
            )

        if attempt < 2:
            time.sleep((2, 6)[attempt])

    raise LLMUnavailable(
        "Gemini did not return a valid response after three attempts. "
        f"Last issue: {last_error}. Check the model, prompt, and schema."
    )


def clear_session_cache() -> None:
    """Clear the in-memory response cache."""
    _CACHE.clear()
