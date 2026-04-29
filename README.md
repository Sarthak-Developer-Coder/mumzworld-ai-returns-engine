# Mumzworld AI Returns Decision Engine

An AI system that **classifies, decides, and safely executes return actions** (refund/exchange/store credit) in English and Arabic using policy-grounded reasoning, strict schemas, and deterministic safeguards.

## Submission status

> **One-paragraph summary**
>
> A lightweight internal copilot for Mumzworld customer support that takes a free-text return request (English or Arabic) plus optional order context, retrieves relevant return-policy snippets (RAG), and produces a **strictly validated JSON** decision: predicted **intent** (**refund / exchange / store_credit / escalate**) plus a deterministic **action** plan (**auto_refund / auto_exchange / issue_store_credit / escalate**) with `requires_human`, confidence + uncertainty flags, follow-up questions when information is missing, and a customer-ready reply in the same language with policy-grounded evidence.

## Submission checklist (fill in)
- Track: **A — AI Engineering Intern**
- GitHub repo link: <add link before submission>
- 3-minute Loom walkthrough: <add link before submission>

This repo is set up to run end-to-end from a fresh clone in under 5 minutes using the included `.venv` or by creating a new virtualenv and installing `requirements.txt`.

The final deliverable is intentionally honest about tradeoffs:
- Mock provider: 14/14 on the 14-case eval suite.
- OpenRouter real-model run (`gpt-4o-mini`): 14/14 schema-valid, 9/14 intent accuracy, 6/14 action accuracy, 10/14 refusal accuracy.
- Main failure mode in the real-model run: the model was overly conservative and often escalated instead of taking the auto action, and it was not perfectly consistent on refusal handling.

## Key Insight

Raw LLM performance alone is not reliable for operations:
- Intent accuracy: ~64%
- Action accuracy: ~43%

To address this, the system separates:
1. **Reasoning (LLM)** -> interprets intent
2. **Decision (deterministic layer)** -> enforces safe actions

This ensures:
- Safe defaults (escalation over incorrect automation)
- Consistent behavior despite model variance
- Production-ready reliability with imperfect models

## Problem selection (why this is worth solving)
Returns are high-volume, policy-sensitive, and bilingual in the GCC. A small mistake (wrong resolution type, missing exceptions, overconfident replies) creates real cost: longer handle times, inconsistent customer experience, and avoidable escalations. This prototype targets a realistic internal workflow: **assist the agent**, don’t replace them.

## Why AI (not just rules)
- Return messages are messy: mixed intent, incomplete info, and colloquial Arabic/English.
- Policies are long and exception-heavy; retrieval + citations helps prevent “made up policy”.
- A strict JSON schema makes outputs *integratable* (routing + drafting + auditability).

## Quickstart (under ~5 minutes)

Requirements: Python 3.11+ (tested on 3.14).

1) Create a virtualenv and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Configure env:

- Copy `.env.example` → `.env`
- Set `MW_LLM_PROVIDER`:
  - `openrouter` (recommended) requires `OPENROUTER_API_KEY`
  - `mock` runs without any key (baseline)

### OpenRouter setup (recommended)
1) Create an OpenRouter key.
2) Put it in `.env` as `OPENROUTER_API_KEY=...`
3) Pick a valid chat model in `.env` (example: `gpt-4o-mini`).

3) Run the 5-case demo (good for Loom recording):

```powershell
python -m mumzworld_ai.demo
```

4) Run evals:

```powershell
python -m mumzworld_ai.evals.run
```

## Architecture

```mermaid
flowchart LR
  A[Customer message<br/>EN or AR] --> B[Language detect]
  B --> C[Retrieve policy chunks<br/>BM25 over markdown]
  C --> D[LLM generates strict JSON<br/>asks follow-ups]
  D --> E[Parse + Pydantic validate]
  E --> F[Grounding checks<br/>drop bad quotes, warn]
  F --> G[Deterministic action layer<br/>auto_* vs escalate]
  G --> H[TriageResult JSON]
```

Key design choices:
- Retrieval is **lexical BM25** (fast, no embedding downloads).
- Outputs are **schema-validated** (`pydantic`) with automatic JSON-repair retries.
- A deterministic **action layer** gates automation (only auto-executes when grounded + confident); otherwise escalates.
- Uncertainty is explicit via `needs_human`, `confidence`, `follow_up_questions`, and `action.requires_human`.

Decision output example:

```json
{
  "intent": "refund",
  "action": {
    "type": "auto_refund",
    "requires_human": false,
    "preview_message_en": "Your refund has been initiated.",
    "preview_message_ar": "تم بدء عملية الاسترداد الخاصة بك."
  }
}
```

Main entrypoint: `mumzworld_ai.agent.triage_return_request()`

## CLI usage

```powershell
# JSON only
python -m mumzworld_ai.cli --message "I received the wrong item, please exchange" --json

# With order context (inline JSON)
python -m mumzworld_ai.cli --message "وصلني المنتج مكسور" --language ar --context '{"order_id":"MW-123"}'
```

## Loom walkthrough script (3 minutes)
1) Run `python -m mumzworld_ai.demo` (shows 5 cases end-to-end).
2) Point out one case that is ambiguous and asks follow-ups.
3) Point out the refusal case (medical) and how it escalates.
4) Run `python -m mumzworld_ai.evals.run` and show the summary + `mumzworld_ai/evals/out/report.md`.

## Evals
The eval harness is intentionally lightweight and repeatable.

## Evaluation Snapshot

| Mode | Intent | Action | Refusal |
|------|--------|--------|--------|
| Mock (pipeline) | 100% | 100% | 100% |
| OpenRouter (gpt-4o-mini) | 64% | 43% | 71% |

- Cases live in `mumzworld_ai/evals/cases.jsonl` (14 cases: EN+AR, prompt-injection, out-of-scope refusal).
- Runner writes:
  - `mumzworld_ai/evals/out/report.md`
  - `mumzworld_ai/evals/out/results.json`

The report includes an **intent confusion matrix** plus a short **failure analysis** section.

Final eval results:
- Mock baseline: 14/14 schema-valid, 14/14 intent accuracy, 14/14 action accuracy, 14/14 refusal accuracy.
- OpenRouter (`gpt-4o-mini`): 14/14 schema-valid, 9/14 intent accuracy, 6/14 action accuracy, 10/14 refusal accuracy.
- Failure pattern: the real model tended to escalate when the deterministic action gate could not verify evidence or confidence, which is safer than hallucinating actions but reduces automation coverage.

See `EVALS.md` for the rubric and results.

## Tooling & provenance
- LLM gateway: OpenRouter (OpenAI-compatible API).
- Real-model eval run: OpenRouter with `gpt-4o-mini`.
- Coding assistant: GitHub Copilot Chat (GPT-5.2) for scaffolding and refactors; I overruled/edited generated code to keep it minimal and testable.
- Prompting: the primary system prompt is committed in `mumzworld_ai/agent.py` (`SYSTEM_PROMPT`).
- Harnesses used: `mumzworld_ai.demo`, `mumzworld_ai.evals.run`, and the CLI (`mumzworld_ai.cli`) for targeted checks.

## Safety & limitations
- `data/policies/` is **synthetic**. In production you would ingest the real policy/FAQ per country and keep it versioned.
- This is designed as an **agent-assist** tool; customer-facing deployment would require stronger safeguards (PII handling, logging, monitoring, human review thresholds).
- Conservative by design: prefers escalation over incorrect automation.
- Deterministic overrides ensure consistent outcomes across model runs.

## Repo contents
- `mumzworld_ai/` — core pipeline (retrieval + schema + agent)
- `data/policies/` — synthetic EN/AR policy docs used for RAG grounding
- `EVALS.md` — rubric, test cases, scores, failure modes
- `TRADEOFFS.md` — why this problem, architecture choices, what I cut, what’s next

## Notes
- The policy documents in `data/policies/` are **synthetic** and for prototype/eval only.
- This is an internal-tool prototype intended to assist an agent; it should not be customer-facing without additional safeguards and real policy integration.
- Time spent on this submission was approximately 5 hours, concentrated on the retrieval+schema pipeline, deterministic action layer, eval harness, and submission polish.

## AI usage note (max 5 lines)
- OpenRouter + a multilingual instruct model for structured triage + response drafting.
- GitHub Copilot Chat (GPT-5.2) for code scaffolding and iterative refactors.
- Automated eval harness for schema/label/refusal checks; manual spot-checking for Arabic naturalness.

## Time log (max 5 lines)
- 1.0h: problem selection + policy/data design (synthetic)
- 2.0h: core pipeline (retrieval + schema + LLM + validation)
- 1.0h: demo + eval harness + adversarial cases
- 1.0h: docs + polish
