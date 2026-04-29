from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    llm_provider: str

    openrouter_api_key: str | None
    openrouter_model: str

    temperature: float
    top_k: int
    max_retries: int

    project_root: Path
    policies_dir: Path


_cached: Settings | None = None


def get_settings() -> Settings:
    """Load settings once from environment variables.

    Uses `.env` if present (via python-dotenv).
    """

    global _cached
    if _cached is not None:
        return _cached

    load_dotenv(override=False)

    llm_provider = (os.getenv("MW_LLM_PROVIDER") or "mock").strip().lower()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or None
    openrouter_model = (os.getenv("OPENROUTER_MODEL") or "qwen/qwen2.5-7b-instruct").strip()

    temperature = _env_float("MW_TEMPERATURE", 0.2)
    top_k = _env_int("MW_TOP_K", 5)
    max_retries = _env_int("MW_MAX_RETRIES", 2)

    policies_dir = _PROJECT_ROOT / "data" / "policies"

    _cached = Settings(
        llm_provider=llm_provider,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        temperature=temperature,
        top_k=top_k,
        max_retries=max_retries,
        project_root=_PROJECT_ROOT,
        policies_dir=policies_dir,
    )

    return _cached
