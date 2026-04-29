from __future__ import annotations

from rich.console import Console

from .agent import TriageInputs, triage_return_request


DEMO_CASES = [
    {
        "title": "EN — Damaged on arrival (refund)",
        "message": "Hi, my baby bottle sterilizer arrived broken and the plastic is cracked. I want a refund. Order #MW-10483. Delivered yesterday.",
        "context": {
            "order_id": "MW-10483",
            "product_name": "Baby bottle sterilizer",
            "delivered_date": "2026-04-27",
            "customer_requested": "refund",
        },
        "language": "en",
    },
    {
        "title": "EN — Wrong item shipped (exchange)",
        "message": "I ordered size 4 diapers but received size 3. Can you exchange it? Order MW-99211.",
        "context": {"order_id": "MW-99211", "product_name": "Diapers size 4"},
        "language": "en",
    },
    {
        "title": "AR — تغيير مقاس / رصيد متجر", 
        "message": "مرحباً، استلمت فستان أمومة لكن المقاس غير مناسب ولم أفتحه. أريد إرجاعه. رقم الطلب MW-77810.",
        "context": {"order_id": "MW-77810", "product_name": "فستان أمومة"},
        "language": "ar",
    },
    {
        "title": "AR — منتج غير قابل للإرجاع بعد الفتح (تصعيد + أسئلة)",
        "message": "فتحت علبة حليب أطفال لكن طفلي لم يتقبله. هل يمكنني إرجاعه واسترداد المبلغ؟ رقم الطلب MW-65001.",
        "context": {"order_id": "MW-65001", "product_name": "حليب أطفال"},
        "language": "ar",
    },
    {
        "title": "EN — Out of scope medical (refusal)",
        "message": "My baby has a rash after using this lotion. What medicine should I give?",
        "context": None,
        "language": "en",
    },
]


def main() -> int:
    console = Console()
    console.rule("Mumzworld Returns Triage Copilot — Demo")

    for i, case in enumerate(DEMO_CASES, start=1):
        console.rule(f"Case {i}/5 — {case['title']}")
        console.print("Input message:")
        console.print(case["message"])

        result = triage_return_request(
            TriageInputs(
                message=case["message"],
                order_context=case.get("context"),
                language_hint=case.get("language"),
            )
        )

        console.print("\nOutput JSON:")
        console.print_json(result.model_dump_json(indent=2, ensure_ascii=False))

    console.rule("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
