from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ..agent import TriageInputs, triage_return_request
from ..config import get_settings


@dataclass(frozen=True)
class Case:
    id: str
    language: str
    message: str
    context: dict[str, Any] | None
    expected_intent: str
    expected_action_type: str | None
    expected_requires_human: bool | None
    expected_refusal: bool


def _load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        expected = obj.get("expected", {})

        expected_intent = expected.get("intent") or expected.get("category")
        if not expected_intent:
            raise ValueError(f"Case {obj.get('id')} missing expected.intent")

        cases.append(
            Case(
                id=obj["id"],
                language=obj.get("language") or "en",
                message=obj["message"],
                context=obj.get("context"),
                expected_intent=str(expected_intent),
                expected_action_type=expected.get("action_type"),
                expected_requires_human=expected.get("requires_human"),
                expected_refusal=bool(expected.get("refusal", False)),
            )
        )

    return cases


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _confusion_matrix_md(labels: list[str], cm: dict[str, dict[str, int]]) -> list[str]:
    lines: list[str] = []
    lines.append("| expected\\pred | " + " | ".join(labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
    for e in labels:
        lines.append("| " + e + " | " + " | ".join(str(cm[e][p]) for p in labels) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run evals for the returns triage copilot.")
    parser.add_argument(
        "--cases",
        type=str,
        default=str(Path(__file__).with_name("cases.jsonl")),
        help="Path to cases.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).with_name("out")),
        help="Directory for outputs (report + JSON results)",
    )

    args = parser.parse_args(argv)

    cases_path = Path(args.cases)
    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    cases = _load_cases(cases_path)

    console = Console()
    console.rule("Evals")
    console.print(f"Provider: {settings.llm_provider}")

    rows: list[dict[str, Any]] = []

    # Summary stats.
    schema_valid = 0
    correct_intent = 0
    correct_refusal = 0

    scored_action_type = 0
    correct_action_type = 0

    scored_requires_human = 0
    correct_requires_human = 0

    # Confusion matrix for intent.
    intent_labels = ["refund", "exchange", "store_credit", "escalate"]
    cm_intent: dict[str, dict[str, int]] = {e: {p: 0 for p in intent_labels} for e in intent_labels}

    # Per-language breakdown (intent).
    by_lang_total: Counter[str] = Counter()
    by_lang_ok_intent: Counter[str] = Counter()

    # Warning stats.
    warning_counts: Counter[str] = Counter()

    for case in cases:
        try:
            result = triage_return_request(
                TriageInputs(
                    message=case.message,
                    order_context=case.context,
                    language_hint=case.language,
                )
            )

            schema_valid += 1

            pred_intent = result.intent.value
            pred_refusal = result.refusal is not None

            pred_action_type = result.action.type.value if result.action is not None else None
            pred_requires_human = result.action.requires_human if result.action is not None else None

            is_intent_ok = pred_intent == case.expected_intent
            is_refusal_ok = pred_refusal == case.expected_refusal

            is_action_type_ok = True
            if case.expected_action_type is not None:
                scored_action_type += 1
                is_action_type_ok = pred_action_type == case.expected_action_type
                if is_action_type_ok:
                    correct_action_type += 1

            is_requires_human_ok = True
            if case.expected_requires_human is not None:
                scored_requires_human += 1
                is_requires_human_ok = pred_requires_human == case.expected_requires_human
                if is_requires_human_ok:
                    correct_requires_human += 1

            if is_intent_ok:
                correct_intent += 1

            if is_refusal_ok:
                correct_refusal += 1

            if case.expected_intent in cm_intent and pred_intent in cm_intent[case.expected_intent]:
                cm_intent[case.expected_intent][pred_intent] += 1

            by_lang_total[case.language] += 1
            if is_intent_ok:
                by_lang_ok_intent[case.language] += 1

            for w in result.warnings or []:
                warning_counts[str(w)] += 1

            ok = is_intent_ok and is_refusal_ok and is_action_type_ok and is_requires_human_ok

            rows.append(
                {
                    "id": case.id,
                    "language": case.language,
                    "expected_intent": case.expected_intent,
                    "pred_intent": pred_intent,
                    "expected_action_type": case.expected_action_type,
                    "pred_action_type": pred_action_type,
                    "expected_requires_human": case.expected_requires_human,
                    "pred_requires_human": pred_requires_human,
                    "expected_refusal": case.expected_refusal,
                    "pred_refusal": pred_refusal,
                    "warnings": result.warnings,
                    "ok": ok,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "id": case.id,
                    "language": case.language,
                    "expected_intent": case.expected_intent,
                    "pred_intent": None,
                    "expected_action_type": case.expected_action_type,
                    "pred_action_type": None,
                    "expected_requires_human": case.expected_requires_human,
                    "pred_requires_human": None,
                    "expected_refusal": case.expected_refusal,
                    "pred_refusal": None,
                    "warnings": [f"exception:{type(e).__name__}", str(e)],
                    "ok": False,
                }
            )

    total = len(cases)
    schema_success_rate = schema_valid / total if total else 0.0

    intent_acc = correct_intent / schema_valid if schema_valid else 0.0
    refusal_acc = correct_refusal / schema_valid if schema_valid else 0.0

    action_type_acc = (correct_action_type / scored_action_type) if scored_action_type else 0.0
    requires_human_acc = (correct_requires_human / scored_requires_human) if scored_requires_human else 0.0

    # Console output.
    table = Table(show_lines=False)
    table.add_column("case")
    table.add_column("exp_intent")
    table.add_column("pred_intent")
    table.add_column("exp_action")
    table.add_column("pred_action")
    table.add_column("exp_ref")
    table.add_column("pred_ref")
    table.add_column("ok")

    for r in rows:
        table.add_row(
            str(r["id"]),
            str(r["expected_intent"]),
            str(r["pred_intent"]),
            str(r["expected_action_type"]),
            str(r["pred_action_type"]),
            str(r["expected_refusal"]),
            str(r["pred_refusal"]),
            "✅" if r["ok"] else "❌",
        )

    console.print(table)

    summary_lines = [
        "Summary:",
        f"- Cases: {total}",
        f"- Schema-valid outputs: {schema_valid}/{total} ({schema_success_rate:.0%})",
        f"- Intent accuracy (on valid): {correct_intent}/{schema_valid} ({intent_acc:.0%})" if schema_valid else "- Intent accuracy: n/a",
        f"- Action.type accuracy (scored): {correct_action_type}/{scored_action_type} ({action_type_acc:.0%})" if scored_action_type else "- Action.type accuracy: n/a",
        f"- Action.requires_human accuracy (scored): {correct_requires_human}/{scored_requires_human} ({requires_human_acc:.0%})" if scored_requires_human else "- Action.requires_human accuracy: n/a",
        f"- Refusal accuracy (on valid): {correct_refusal}/{schema_valid} ({refusal_acc:.0%})" if schema_valid else "- Refusal accuracy: n/a",
    ]
    console.print("\n".join(summary_lines))

    results_json = {
        "provider": settings.llm_provider,
        "total": total,
        "schema_valid": schema_valid,
        "schema_success_rate": schema_success_rate,
        "intent_accuracy": intent_acc,
        "action_type_accuracy": action_type_acc,
        "requires_human_accuracy": requires_human_acc,
        "refusal_accuracy": refusal_acc,
        "confusion_matrix_intent": cm_intent,
        "warning_counts": dict(warning_counts),
        "cases": rows,
    }

    (out_dir / "results.json").write_text(json.dumps(results_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report.
    report_md: list[str] = [
        "# Eval Report",
        "",
        f"- Provider: `{settings.llm_provider}`",
        f"- Total cases: {total}",
        f"- Schema-valid outputs: {schema_valid}/{total} ({schema_success_rate:.0%})",
        f"- Intent accuracy (on valid): {correct_intent}/{schema_valid} ({intent_acc:.0%})" if schema_valid else "- Intent accuracy: n/a",
        f"- Action.type accuracy (scored): {correct_action_type}/{scored_action_type} ({action_type_acc:.0%})" if scored_action_type else "- Action.type accuracy: n/a",
        f"- Action.requires_human accuracy (scored): {correct_requires_human}/{scored_requires_human} ({requires_human_acc:.0%})" if scored_requires_human else "- Action.requires_human accuracy: n/a",
        f"- Refusal accuracy (on valid): {correct_refusal}/{schema_valid} ({refusal_acc:.0%})" if schema_valid else "- Refusal accuracy: n/a",
        "",
        "## Confusion matrix (intent)",
        "",
        "Rows = expected, columns = predicted.",
        "",
        *_confusion_matrix_md(intent_labels, cm_intent),
    ]

    report_md.extend(["", "## Breakdown", ""])
    for lang in sorted(by_lang_total.keys()):
        n = by_lang_total[lang]
        ok = by_lang_ok_intent[lang]
        report_md.append(f"- `{lang}` intent accuracy: {ok}/{n} ({(ok / n if n else 0):.0%})")

    if warning_counts:
        report_md.extend(["", "## Warning stats", "", "Top warnings:", ""])
        for w, c in warning_counts.most_common(10):
            report_md.append(f"- {w}: {c}")

    failures = [r for r in rows if not r.get("ok")]
    report_md.extend(["", "## Failure analysis", ""])
    if not failures:
        report_md.append("- No failures in this run.")
    else:
        report_md.append("Key failing cases (first 10):")
        report_md.append("")
        report_md.append(
            "| Case | Expected intent | Predicted intent | Expected action | Predicted action | Refusal exp/pred | Notes |"
        )
        report_md.append("|---|---|---|---|---|---|---|")
        for r in failures[:10]:
            notes = "; ".join((r.get("warnings") or [])[:2])
            report_md.append(
                f"| {r['id']} | {r.get('expected_intent')} | {r.get('pred_intent')} | {r.get('expected_action_type')} | {r.get('pred_action_type')} | {r.get('expected_refusal')}/{r.get('pred_refusal')} | {notes} |"
            )

    report_md.extend(
        [
            "",
            "## Per-case",
            "",
            "| Case | Expected intent | Predicted intent | Action (exp/pred) | Refusal (exp/pred) | OK |",
            "|---|---|---|---|---|---|",
        ]
    )
    for r in rows:
        report_md.append(
            f"| {r['id']} | {r.get('expected_intent')} | {r.get('pred_intent')} | {r.get('expected_action_type')}/{r.get('pred_action_type')} | {r.get('expected_refusal')}/{r.get('pred_refusal')} | {'OK' if r.get('ok') else 'FAIL'} |"
        )

    (out_dir / "report.md").write_text("\n".join(report_md), encoding="utf-8")

    console.print(f"\nWrote: {out_dir / 'report.md'}")
    console.print(f"Wrote: {out_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
