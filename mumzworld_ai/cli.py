from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console

from .agent import TriageInputs, triage_return_request


def _load_context(raw: str | None) -> dict | None:
    if raw is None:
        return None

    raw = raw.strip()
    if raw == "":
        return None

    # If it's a file path, load it.
    if raw.endswith(".json"):
        with open(raw, "r", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mumzworld-ai",
        description="Multilingual returns triage copilot (prototype).",
    )

    parser.add_argument("--message", "-m", type=str, help="Customer message (EN or AR). If omitted, reads stdin.")
    parser.add_argument(
        "--context",
        "-c",
        type=str,
        help="Order context JSON (string) or path to .json file.",
    )
    parser.add_argument("--language", type=str, choices=["en", "ar"], help="Optional language hint.")
    parser.add_argument("--json", action="store_true", help="Print JSON only (no pretty formatting).")

    args = parser.parse_args(argv)

    message = args.message
    if not message:
        message = sys.stdin.read().strip()

    if not message:
        parser.error("Provide --message or pipe a message via stdin")

    context = _load_context(args.context)

    result = triage_return_request(TriageInputs(message=message, order_context=context, language_hint=args.language))

    if args.json:
        print(result.model_dump_json(indent=2, ensure_ascii=False))
        return 0

    console = Console()
    console.rule("Triage Result")
    console.print_json(result.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
