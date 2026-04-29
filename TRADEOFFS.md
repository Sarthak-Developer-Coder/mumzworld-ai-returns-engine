# Tradeoffs

## Why this problem
Returns are high-volume, policy-sensitive, and bilingual in the GCC. A copilot that reliably routes to the right resolution type (refund/exchange/store-credit/escalate) and drafts a policy-consistent reply can reduce handle time and improve consistency.

This also maps cleanly to how AI value shows up in e-commerce ops: fewer exceptions, faster resolution, fewer policy mistakes, and better auditability.

### What I considered but didn’t pick
- **Gift finder / product recommendations**: compelling, but needs a realistic product catalog and ranking signals to evaluate properly without scraping.
- **Review summarization (“Moms Verdict”)**: great UX, but harder to evaluate “correctness” without ground truth; risk of vibe-based grading.
- **PDP generation from images**: strong multimodal demo, but typically needs paid multimodal APIs and higher risk of factual hallucinations.

## What makes it non-trivial AI engineering
- **RAG over messy policy text** (synthetic in this repo; would be real internal docs in production)
- **Strict structured output** with schema validation + retries
- **Multilingual (English + Arabic)**, including follow-up questions and customer-ready tone
- **Evals beyond vibes**: validity, routing accuracy, uncertainty/refusal checks, grounding checks

## Key choices
- **Lightweight lexical retrieval (BM25)** instead of embeddings: fast, deterministic, no model downloads, good enough for short policy docs.
- **Schema-first design** (`pydantic` + strict JSON): makes it integratable (routing, logging, human review) and prevents “pretty text” demos.
- **Deterministic action layer**: separate “what the customer wants” (`intent`) from “what we can safely do now” (`action`), with conservative escalation gates.
- **OpenRouter** as a low-friction gateway: lets reviewers swap models without code changes.

### Model choice
I optimized for models that can:
- Follow strict JSON output requirements
- Write natural Arabic and English
- Stay grounded to provided context

In practice, some models are great at Arabic but weak at strict JSON; the JSON-repair loop is there to handle this, but model selection still matters.

## Uncertainty handling
- If key details are missing (order number, delivery date, photos for damage), the system asks follow-up questions.
- If it’s out-of-scope (medical/legal), it produces a structured refusal (`refusal`) and routes to `intent=escalate` with `action.type=escalate`.
- If policy was retrieved but no citations are provided, the agent adds `no_evidence_from_policy` and caps confidence.

## What I cut (for 5-hour scope)
- Real policy ingestion from a CMS
- Agent UI (e.g., Zendesk plugin)
- Human-in-the-loop feedback loop + training data capture

Also cut:
- Per-country policy variants and regional Arabic nuance
- Tooling for attaching customer photos (would improve damage/defect handling)
- Calibration of confidence against historical outcomes

## What I’d build next
- Better retrieval (hybrid BM25 + embeddings) and per-country policy support
- Calibrated confidence with historical labels
- A/B test plan: handle time, escalation rate, CSAT

Concrete next steps:
- Integrate real Mumzworld policy docs + versioning; add country/storefront dimension
- Add a “policy citation required” gate for refund/exchange decisions
- Add agent feedback buttons (“correct/incorrect”) to collect training data
- Add monitoring: malformed JSON rate, escalation rate, refusal rate, language mismatch warnings
