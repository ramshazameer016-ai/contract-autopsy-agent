
"""Safe Gemini gateway with offline protection and validated responses."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, is_offline

T = TypeVar("T", bound=BaseModel)

# In-memory cache only. Prompts and contract text are not written to disk.
_CACHE: dict[str, str] = {}


class LLMUnavailable(RuntimeError):
    """Raised when the AI provider is unavailable or returns invalid data."""


def _call_gemini(prompt: str) -> str:
    """Make one Gemini request, only when AI is enabled and configured."""
    if is_offline() or not GEMINI_API_KEY:
        raise LLMUnavailable("AI is offline or no API key is configured.")

    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    response_text = response.text
    if not response_text:
        raise LLMUnavailable("The provider returned an empty response.")

    return response_text.strip()


def generate_validated(
    prompt: str,
    schema: type[T],
    *,
    prompt_version: str,
) -> T:
    """Generate structured output and validate it against a Pydantic schema."""

    # IMPORTANT: reject offline requests before cache lookup or provider calls.
    if is_offline() or not GEMINI_API_KEY:
        raise LLMUnavailable("AI is offline or no API key is configured.")

    cache_key = hashlib.sha256(
        f"{prompt_version}\0{schema.__name__}\0{prompt}".encode("utf-8")
    ).hexdigest()

    if cache_key in _CACHE:
        return schema.model_validate_json(_CACHE[cache_key])

    correction = ""
    last_error = "No response"

    # Three total attempts: initial request plus up to two retries.
    for attempt in range(3):
        try:
            raw = _call_gemini(prompt + correction)
            parsed = schema.model_validate(json.loads(raw))

            # Cache only successfully validated responses.
            _CACHE[cache_key] = parsed.model_dump_json()
            return parsed

        except LLMUnavailable:
            # Do not retry explicit offline/configuration failures.
            raise

        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)[:500]
            correction = (
                "\nThe previous output failed validation. "
                f"Error: {last_error}\n"
                "Return corrected JSON only."
            )

        except Exception as exc:
            # Avoid exposing provider error details or contract content.
            last_error = type(exc).__name__

            if attempt < 2:
                time.sleep((2, 6)[attempt])

    raise LLMUnavailable(
        "AI response failed validation or provider requests failed "
        f"({last_error})."
    )


def clear_session_cache() -> None:
    """Clear the in-memory response cache."""
    _CACHE.clear()
