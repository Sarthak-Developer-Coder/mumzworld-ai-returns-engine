from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
import re

import httpx


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class BaseLLM:
    name: str

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float) -> str:
        raise NotImplementedError


class OpenRouterLLM(BaseLLM):
    name = "openrouter"

    def __init__(self, *, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

        # Optional, but OpenRouter recommends these for attribution.
        self._http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self._x_title = os.getenv("OPENROUTER_X_TITLE")

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float) -> str:
        url = f"{self._base_url}/chat/completions"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._x_title:
            headers["X-Title"] = self._x_title

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"OpenRouter request failed: {e}") from e

        if resp.status_code >= 400:
            body = resp.text
            body = body[:2000] + ("..." if len(body) > 2000 else "")
            raise LLMError(f"OpenRouter error {resp.status_code}: {body}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Unexpected OpenRouter response shape: {json.dumps(data)[:2000]}") from e


class MockLLM(BaseLLM):
    """A tiny deterministic baseline so the repo runs without any API key.

    This is intentionally simplistic; it exists to validate the pipeline and eval harness.
    """

    name = "mock"

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float) -> str:
        user_prompt = "\n".join([m.content for m in messages if m.role == "user"])

        # The agent includes policy chunks in the prompt; for a baseline we only want the
        # *customer message* section so keyword heuristics don't get polluted by policy text.
        m = re.search(
            r"Customer message.*?:\s*\n(?P<msg>.*?)\nOrder context",
            user_prompt,
            flags=re.DOTALL,
        )
        customer_msg = (m.group("msg") if m else user_prompt).strip()

        m_ctx = re.search(
            r"Order context.*?:\s*\n(?P<ctx>.*?)\nRelevant policy chunks",
            user_prompt,
            flags=re.DOTALL,
        )
        raw_ctx = (m_ctx.group("ctx") if m_ctx else "null").strip()
        try:
            ctx = json.loads(raw_ctx)
        except Exception:
            ctx = None

        lower = customer_msg.lower()
        lang = "ar" if any("\u0600" <= ch <= "\u06FF" for ch in customer_msg) else "en"

        intent = "escalate"

        # English-ish signals
        wants_exchange = any(k in lower for k in ["exchange", "replace", "replacement"])
        is_damaged = any(k in lower for k in ["damaged", "broken", "cracked", "torn"])
        is_defective = any(
            k in lower
            for k in [
                "defect",
                "defective",
                "not working",
                "doesn't work",
                "doesnt work",
                "stopped working",
            ]
        )
        is_wrong_item = (
            any(k in lower for k in ["wrong item", "different item", "not what i ordered", "instead of"])
            or ("received" in lower and any(k in lower for k in ["wrong", "different", "instead", "size"]))
        )
        is_change_of_mind = any(k in lower for k in ["change my mind", "changed my mind", "don't want", "dont want"])
        is_late = any(k in lower for k in ["last month", "over 30", "30 days", "more than 14"])

        # Arabic signals
        ar_text = customer_msg  # keep original for Arabic matching
        ar_wants_exchange = any(k in ar_text for k in ["استبدال", "استبداله", "استبدالها", "تبديل", "استبدله"])
        ar_is_damaged = any(k in ar_text for k in ["تالف", "مكسور", "مكسورة", "انكسر", "مشقوق", "متضرر"])
        ar_is_wrong_item = any(
            k in ar_text
            for k in [
                "منتج مختلف",
                "مختلف عن",
                "غير المطلوب",
                "بدلاً من",
                "بدلا من",
            ]
        ) or (any(k in ar_text for k in ["استلمت", "وصلني"]) and "مقاس" in ar_text)
        ar_is_change_of_mind = any(k in ar_text for k in ["غير مناسب", "غيّرت رأيي", "غيرت رأيي", "تغيير رأي", "لا أريده", "لا اريده"])
        ar_is_late = any(k in ar_text for k in ["الشهر الماضي", "أكثر من 14", "اكثر من 14", "بعد شهر", "بعد شهرين"])

        # Decision (keep it deterministic)
        if lang == "ar":
            if ar_is_damaged:
                intent = "refund"
            elif ar_is_wrong_item or ar_wants_exchange:
                intent = "exchange"
            elif ar_is_change_of_mind or ar_is_late:
                intent = "store_credit"
        else:
            if is_damaged:
                intent = "refund"
            elif is_wrong_item or wants_exchange:
                intent = "exchange"
            elif is_change_of_mind or is_late:
                intent = "store_credit"

        # Medical/safety = refuse.
        is_medical = any(
            k in lower for k in ["fever", "vomit", "vomiting", "rash", "blood", "doctor", "medicine"]
        ) or any(
            k in customer_msg for k in ["حرارة", "حمى", "قيء", "يتقيأ", "طفح", "دم", "طبيب", "دواء"]
        )

        if is_medical:
            out = {
                "language": lang,
                "intent": "escalate",
                "action": None,
                "needs_human": True,
                "confidence": 0.2,
                "summary": "Out of scope medical/safety request.",
                "extracted": {
                    "order_id": None,
                    "sku": None,
                    "product_name": None,
                    "delivered_date": None,
                    "issue_type": None,
                    "customer_requested": None,
                },
                "follow_up_questions": [],
                "recommended_next_steps": ["Escalate to a trained agent."],
                "customer_message": "I can't help with medical advice. Please contact a doctor or emergency services if needed." if lang == "en" else "لا يمكنني تقديم نصائح طبية. يرجى التواصل مع طبيب أو خدمات الطوارئ عند الحاجة.",
                "evidence": [],
                "refusal": {
                    "is_refusal": True,
                    "reason": "Medical advice is out of scope for this tool.",
                    "safe_alternative": "Escalate to a trained agent and advise the customer to consult a medical professional.",
                },
                "warnings": ["mock_mode"],
            }
            return json.dumps(out, ensure_ascii=False)

        out = {
            "language": lang,
            "intent": intent,
            "action": None,
            "needs_human": True if intent == "escalate" else False,
            "confidence": 0.86 if intent != "escalate" else 0.40,
            "summary": "Return request triage (mock baseline)." if lang == "en" else "فرز طلب إرجاع (نموذج تجريبي).",
            "extracted": {
                "order_id": None,
                "sku": None,
                "product_name": None,
                "delivered_date": None,
                "issue_type": None,
                "customer_requested": None,
            },
            "follow_up_questions": [] if (isinstance(ctx, dict) and ctx.get("order_id")) else [
                "What is your order number?" if lang == "en" else "ما هو رقم الطلب؟"
            ],
            "recommended_next_steps": ["Collect order and item details." if lang == "en" else "جمع تفاصيل الطلب والمنتج."],
            "customer_message": "Thanks for reaching out. Please share your order number and item name so we can help." if lang == "en" else "شكراً لتواصلك. يرجى مشاركة رقم الطلب واسم المنتج لنتمكن من المساعدة.",
            "evidence": [],
            "refusal": None,
            "warnings": ["mock_mode"],
        }

        return json.dumps(out, ensure_ascii=False)
