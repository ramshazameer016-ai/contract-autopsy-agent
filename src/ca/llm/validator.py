
from typing import Literal

from pydantic import BaseModel, Field, field_validator


SummaryStatus = Literal["Stated", "Missing", "Unclear", "Inferred"]


class SummaryItem(BaseModel):
    category: str = Field(min_length=1, max_length=100)
    status: SummaryStatus
    explanation: str = Field(min_length=1, max_length=1200)
    evidence_quote: str = ""
    clause_id: str | None = None
    page: int | None = None

    @field_validator("evidence_quote")
    @classmethod
    def strip_quote(cls, value: str) -> str:
        return value.strip()


class PlainSummary(BaseModel):
    language: Literal["English", "Hindi", "Telugu", "Tamil"]
    items: list[SummaryItem] = Field(min_length=1, max_length=40)
    generated_by: Literal["llm", "template"]
    prompt_version: str = "summary_v1"
