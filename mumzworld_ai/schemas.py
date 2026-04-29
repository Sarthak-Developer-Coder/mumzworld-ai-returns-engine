from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, confloat, model_validator


class Resolution(str, Enum):
    refund = "refund"
    exchange = "exchange"
    store_credit = "store_credit"
    escalate = "escalate"


class IssueType(str, Enum):
    damaged = "damaged"
    defective = "defective"
    wrong_item = "wrong_item"
    change_of_mind = "change_of_mind"
    late = "late"
    other = "other"


class ActionType(str, Enum):
    auto_refund = "auto_refund"
    auto_exchange = "auto_exchange"
    issue_store_credit = "issue_store_credit"
    escalate = "escalate"


class ActionPlan(BaseModel):
    model_config = {"extra": "forbid"}

    type: ActionType
    reason: str = Field(..., min_length=1)
    requires_human: bool


class EvidenceQuote(BaseModel):
    model_config = {"extra": "forbid"}

    doc_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1)
    relevance: str = Field(..., min_length=1)


class ExtractedFields(BaseModel):
    model_config = {"extra": "forbid"}

    order_id: str | None = None
    sku: str | None = None
    product_name: str | None = None
    delivered_date: date | None = None

    issue_type: IssueType | None = None
    customer_requested: Resolution | None = None


class Refusal(BaseModel):
    model_config = {"extra": "forbid"}

    is_refusal: Literal[True] = True
    reason: str = Field(..., min_length=1)
    safe_alternative: str = Field(..., min_length=1)


class TriageResult(BaseModel):
    """Validated structured output for a single return-request triage."""

    model_config = {"extra": "forbid"}

    language: Literal["en", "ar"]
    intent: Resolution = Field(
        ..., validation_alias=AliasChoices("intent", "category")
    )

    action: ActionPlan | None = None

    needs_human: bool
    confidence: confloat(ge=0, le=1)

    summary: str = Field(..., min_length=1, description="One-sentence summary of the issue")

    extracted: ExtractedFields

    follow_up_questions: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)

    customer_message: str = Field(..., min_length=1)

    evidence: list[EvidenceQuote] = Field(default_factory=list)
    refusal: Refusal | None = None

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "TriageResult":
        if self.refusal is not None:
            # A refusal always requires a human escalation for correct routing.
            if self.intent != Resolution.escalate:
                raise ValueError("If refusal is present, intent must be 'escalate'.")
            if not self.needs_human:
                raise ValueError("If refusal is present, needs_human must be true.")

        if self.action is not None:
            if self.action.type == ActionType.escalate and not self.action.requires_human:
                raise ValueError("If action.type is 'escalate', action.requires_human must be true.")
        return self
