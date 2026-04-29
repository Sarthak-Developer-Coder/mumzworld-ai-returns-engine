from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import Settings, get_settings
from .llm import BaseLLM, ChatMessage, LLMError, MockLLM, OpenRouterLLM
from .retrieval import Language, PolicyChunk, Retriever, detect_language, load_policy_chunks
from .schemas import ActionPlan, ActionType, EvidenceQuote, Resolution, TriageResult


SYSTEM_PROMPT = """You are an internal customer-support copilot for Mumzworld.

Task:
Given a customer's return request (English or Arabic) and optional order context, produce a single JSON object that follows the provided schema.

Hard rules:
- Output MUST be valid JSON and MUST match the schema exactly (no extra keys).
- If a field is unknown, use null (do not invent, do not use empty strings).
- Ground your decision in the provided policy chunks. If you cannot support a claim, lower confidence and ask follow-up questions.
- If the user asks for medical/legal advice or anything unrelated to returns, refuse safely: set `refusal` and route to `intent=escalate` with `needs_human=true`.
- Evidence quotes MUST be exact substrings from the cited policy chunk.
- The `customer_message` must be written naturally in the same language as the customer's message.

Note:
- `action` may be null; the system will compute a safe action plan.

Be concise and operational. Prefer asking 1–3 follow-up questions over guessing.
"""


@dataclass(frozen=True)
class TriageInputs:
    message: str
    order_context: dict[str, Any] | None = None
    language_hint: Language | None = None


def _schema_contract() -> str:
    # Keep this short; models are more reliable with a contract than with a huge JSON Schema dump.
    return (
        "Schema (JSON object):\n"
        "- language: 'en' | 'ar'\n"
        "- intent: 'refund' | 'exchange' | 'store_credit' | 'escalate'\n"
        "- action: null OR { type: 'auto_refund' | 'auto_exchange' | 'issue_store_credit' | 'escalate', reason: string, requires_human: boolean }\n"
        "- needs_human: boolean\n"
        "- confidence: number between 0 and 1\n"
        "- summary: string\n"
        "- extracted: { order_id, sku, product_name, delivered_date(YYYY-MM-DD or null), issue_type, customer_requested }\n"
        "- follow_up_questions: string[]\n"
        "- recommended_next_steps: string[]\n"
        "- customer_message: string\n"
        "- evidence: [{ doc_id, chunk_id, quote, relevance }]\n"
        "- refusal: null OR { is_refusal: true, reason: string, safe_alternative: string }\n"
        "- warnings: string[]\n"
    )


def _merge_known_context(result: TriageResult, order_context: dict[str, Any] | None) -> TriageResult:
    """Merge trusted structured context into `extracted`.

    This makes the pipeline more reliable (especially for action execution) and reduces
    the need for the model to re-extract already-known fields.
    """

    if not order_context:
        return result

    extracted_updates: dict[str, Any] = {}
    warnings = list(result.warnings)

    if result.extracted.order_id is None and isinstance(order_context.get("order_id"), str):
        extracted_updates["order_id"] = order_context["order_id"]

    if result.extracted.sku is None and isinstance(order_context.get("sku"), str):
        extracted_updates["sku"] = order_context["sku"]

    if result.extracted.product_name is None and isinstance(order_context.get("product_name"), str):
        extracted_updates["product_name"] = order_context["product_name"]

    if result.extracted.delivered_date is None and order_context.get("delivered_date") is not None:
        raw = order_context.get("delivered_date")
        if isinstance(raw, str):
            try:
                extracted_updates["delivered_date"] = date.fromisoformat(raw)
            except ValueError:
                warnings.append("bad_context_delivered_date")

    if result.extracted.customer_requested is None and order_context.get("customer_requested") is not None:
        raw_req = order_context.get("customer_requested")
        if isinstance(raw_req, str):
            try:
                extracted_updates["customer_requested"] = Resolution(raw_req)
            except ValueError:
                warnings.append("bad_context_customer_requested")

    if extracted_updates:
        result = result.model_copy(
            update={
                "extracted": result.extracted.model_copy(update=extracted_updates),
                "warnings": warnings,
            }
        )
    elif warnings != result.warnings:
        result = result.model_copy(update={"warnings": warnings})

    return result


def _autofill_evidence_for_mock(
    result: TriageResult,
    *,
    retrieved_chunks: list[PolicyChunk],
) -> TriageResult:
    """Make mock mode demonstrably grounded.

    In mock mode the LLM is heuristic and doesn't reliably produce citations. To keep the
    repo fully runnable without keys (and still show RAG grounding), we auto-cite one
    retrieved policy chunk when evidence is missing.
    """

    if result.refusal is not None:
        return result

    if result.evidence:
        return result

    if not retrieved_chunks:
        return result

    if result.intent == Resolution.refund:
        keywords = ["damaged", "defective", "broken", "تالف", "مكسور", "معيب"]
    elif result.intent == Resolution.exchange:
        keywords = ["wrong item", "exchange", "different", "منتج مختلف", "استبدال"]
    elif result.intent == Resolution.store_credit:
        keywords = ["store credit", "change of mind", "wrong size", "رصيد", "غير مناسب", "تغيير"]
    else:
        keywords = ["escalate", "manual", "non-returnable", "تصعيد", "غير قابلة"]

    chosen = None
    for c in retrieved_chunks:
        hay = c.text.lower()
        if any(k.lower() in hay for k in keywords):
            chosen = c
            break
    chosen = chosen or retrieved_chunks[0]

    # Prefer a line that includes a keyword; otherwise fall back to a short prefix.
    quote = ""
    for line in chosen.text.split("\n"):
        if any(k.lower() in line.lower() for k in keywords):
            quote = line.strip()
            break
    if not quote:
        quote = chosen.text[:220].strip()

    ev = EvidenceQuote(
        doc_id=chosen.doc_id,
        chunk_id=chosen.chunk_id,
        quote=quote,
        relevance="Relevant return-policy excerpt.",
    )

    return result.model_copy(update={"evidence": [ev]})


def _compute_action_plan(result: TriageResult) -> TriageResult:
    """Compute a deterministic action plan (the 'action layer').

    This turns the model output into an operational decision: either auto-execute a
    low-risk action or escalate to a human when uncertainty / missing prerequisites exist.
    """

    warnings = list(result.warnings)
    follow_ups = list(result.follow_up_questions)

    # If structured context filled a field, remove redundant follow-up questions.
    if result.extracted.order_id is not None and follow_ups:
        cleaned: list[str] = []
        for q in follow_ups:
            ql = q.lower()
            is_order_question = ("order" in ql and ("number" in ql or "id" in ql)) or ("رقم الطلب" in q)
            if not is_order_question:
                cleaned.append(q)
        follow_ups = cleaned

    # Minimum requirements to auto-execute.
    min_conf = 0.75
    blocking_warnings = {
        "no_policy_retrieved",
        "no_evidence_from_policy",
    }

    blockers: list[str] = []

    if result.refusal is not None:
        blockers.append("refusal")

    if result.intent == Resolution.escalate:
        blockers.append("intent_escalate")

    if result.needs_human:
        blockers.append("needs_human")

    if follow_ups:
        blockers.append("follow_up_questions")

    if result.extracted.order_id is None:
        blockers.append("missing_order_id")
        q = "Can you share your order number?" if result.language == "en" else "هل يمكنك مشاركة رقم الطلب؟"
        has_order_question = False
        for existing in follow_ups:
            el = existing.lower()
            if ("order" in el and ("number" in el or "id" in el)) or ("رقم الطلب" in existing):
                has_order_question = True
                break
        if not has_order_question and q not in follow_ups:
            follow_ups.append(q)

    if result.confidence < min_conf:
        blockers.append("low_confidence")

    if not result.evidence:
        blockers.append("missing_evidence")

    if any(w in blocking_warnings for w in warnings):
        blockers.append("grounding_missing")

    safe_to_auto = not blockers and result.intent in {
        Resolution.refund,
        Resolution.exchange,
        Resolution.store_credit,
    }

    if safe_to_auto:
        if result.intent == Resolution.refund:
            action_type = ActionType.auto_refund
        elif result.intent == Resolution.exchange:
            action_type = ActionType.auto_exchange
        else:
            action_type = ActionType.issue_store_credit

        requires_human = False
        reason = f"Auto-execute {action_type.value}: grounded policy + confidence={result.confidence:.2f}."
    else:
        action_type = ActionType.escalate
        requires_human = True

        # Keep the recommended resolution visible via `intent`, but route execution to a human.
        short = ", ".join(blockers[:3]) + ("…" if len(blockers) > 3 else "")
        reason = f"Escalate: cannot safely auto-execute intent={result.intent.value}. Blockers: {short}."

        if not result.needs_human:
            warnings.append("escalated_by_action_gate")
            result = result.model_copy(update={"needs_human": True, "warnings": warnings})

    action = ActionPlan(type=action_type, reason=reason, requires_human=requires_human)

    # Update follow-ups if we added any.
    if follow_ups != result.follow_up_questions:
        result = result.model_copy(update={"follow_up_questions": follow_ups})

    return result.model_copy(update={"action": action})


def _render_policy_chunks(chunks: list[PolicyChunk]) -> str:
    if not chunks:
        return "(no policy chunks retrieved)"

    parts: list[str] = []
    for c in chunks:
        parts.append(
            "\n".join(
                [
                    f"[doc_id={c.doc_id} chunk_id={c.chunk_id}]",
                    c.text,
                ]
            )
        )

    return "\n\n---\n\n".join(parts)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    text = (text or "").strip()

    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()

    # Best-effort: take the first full JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    return text


def _build_llm(settings: Settings) -> BaseLLM:
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise LLMError(
                "MW_LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY. "
                "Set it in .env or environment variables."
            )
        return OpenRouterLLM(api_key=settings.openrouter_api_key)

    if settings.llm_provider == "mock":
        return MockLLM()

    raise LLMError(f"Unknown MW_LLM_PROVIDER: {settings.llm_provider}")


@lru_cache(maxsize=1)
def _get_retriever(policies_dir: str) -> Retriever:
    chunks = load_policy_chunks(Path(policies_dir))
    return Retriever(chunks)


def _grounding_check(
    result: TriageResult,
    *,
    retriever: Retriever,
    retrieved_chunks: list[PolicyChunk],
    expected_language: Language,
) -> TriageResult:
    warnings = list(result.warnings)

    # Normalize language (don't silently hide; keep a warning).
    if result.language != expected_language:
        warnings.append(f"language_mismatch:{result.language}->{expected_language}")
        result = result.model_copy(update={"language": expected_language, "warnings": warnings})

    # Encourage grounded outputs: if we retrieved policy but got no citations, cap confidence.
    if result.refusal is None and not result.evidence:
        if retrieved_chunks:
            warnings.append("no_evidence_from_policy")
        else:
            warnings.append("no_policy_retrieved")

        if result.confidence > 0.6:
            result = result.model_copy(update={"confidence": 0.6, "warnings": warnings})
        else:
            result = result.model_copy(update={"warnings": warnings})

        return result

    kept = []

    for ev in result.evidence:
        chunk = retriever.get_chunk(ev.doc_id, ev.chunk_id)
        if chunk is None:
            warnings.append(f"evidence_missing_chunk:{ev.doc_id}:{ev.chunk_id}")
            continue
        if ev.quote not in chunk.text:
            warnings.append(f"evidence_quote_not_substring:{ev.doc_id}:{ev.chunk_id}")
            continue
        kept.append(ev)

    # If we dropped evidence, reflect that in warnings.
    if len(kept) != len(result.evidence):
        result = result.model_copy(update={"evidence": kept, "warnings": warnings})

    return result


def triage_return_request(
    inputs: TriageInputs,
    *,
    settings: Settings | None = None,
) -> TriageResult:
    settings = settings or get_settings()

    message = (inputs.message or "").strip()
    if not message:
        raise ValueError("message is required")

    language: Language = inputs.language_hint or detect_language(message)

    retriever = _get_retriever(str(settings.policies_dir))

    # Retrieval query: customer text plus any structured context values.
    query_parts = [message]
    if inputs.order_context:
        for k, v in inputs.order_context.items():
            if v is None:
                continue
            query_parts.append(f"{k}: {v}")

    query = "\n".join(query_parts)
    policy_chunks = retriever.search(query, language=language, top_k=settings.top_k)

    user_prompt = "\n\n".join(
        [
            _schema_contract(),
            f"Customer message (language={language}):\n{message}",
            f"Order context (JSON or null):\n{json.dumps(inputs.order_context, ensure_ascii=False)}",
            "Relevant policy chunks:\n" + _render_policy_chunks(policy_chunks),
        ]
    )

    llm = _build_llm(settings)

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]

    last_error: str | None = None

    for attempt in range(settings.max_retries + 1):
        raw = llm.chat(model=settings.openrouter_model, messages=messages, temperature=settings.temperature)
        candidate = _extract_json(raw)

        try:
            data = json.loads(candidate)
            parsed = TriageResult.model_validate(data)

            parsed = _merge_known_context(parsed, inputs.order_context)

            if settings.llm_provider == "mock":
                parsed = _autofill_evidence_for_mock(parsed, retrieved_chunks=policy_chunks)

            parsed = _grounding_check(
                parsed,
                retriever=retriever,
                retrieved_chunks=policy_chunks,
                expected_language=language,
            )

            parsed = _compute_action_plan(parsed)
            return parsed
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)

            if attempt >= settings.max_retries:
                break

            # Ask the model to repair its output.
            repair_prompt = "\n\n".join(
                [
                    "Your previous output was invalid. Fix it.",
                    "Return ONLY valid JSON that matches the schema exactly.",
                    "Do not add extra keys.",
                    f"Error: {last_error}",
                    "Previous output:",
                    raw,
                ]
            )
            messages = [
                ChatMessage(role="system", content="You are a JSON repair tool."),
                ChatMessage(role="user", content=repair_prompt),
            ]

    raise LLMError(f"Failed to produce valid schema output after retries. Last error: {last_error}")
