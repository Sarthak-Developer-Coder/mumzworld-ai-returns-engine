from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from rank_bm25 import BM25Okapi


Language = Literal["en", "ar"]


_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> Language:
    return "ar" if _ARABIC_RANGE.search(text or "") else "en"


def _normalize(text: str) -> str:
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    text = _normalize(text)
    # Keep Arabic as-is; lowercase helps English matching.
    text = text.lower()
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split(" ") if t]
    return tokens


@dataclass(frozen=True)
class PolicyChunk:
    doc_id: str
    chunk_id: str
    language: Language
    text: str


def _chunk_markdown(md: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]

    merged: list[str] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        # Merge heading-only blocks with their next block to keep context.
        if re.match(r"^#{1,6}\s+", block) and i + 1 < len(blocks):
            nxt = blocks[i + 1]
            merged.append(f"{block}\n{nxt}".strip())
            i += 2
            continue

        merged.append(block)
        i += 1

    # Drop tiny chunks.
    return [m for m in merged if len(m) >= 40]


def load_policy_chunks(policies_dir: Path) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []

    for path in sorted(policies_dir.glob("*.md")):
        doc_id = path.stem
        md = path.read_text(encoding="utf-8")

        # Best-effort language routing from filename.
        if doc_id.endswith("_ar"):
            lang: Language = "ar"
        elif doc_id.endswith("_en"):
            lang = "en"
        else:
            lang = detect_language(md)

        blocks = _chunk_markdown(md)
        for idx, block in enumerate(blocks, start=1):
            chunks.append(
                PolicyChunk(
                    doc_id=doc_id,
                    chunk_id=f"{idx:03d}",
                    language=lang,
                    text=_normalize(block),
                )
            )

    return chunks


class Retriever:
    def __init__(self, chunks: Iterable[PolicyChunk]):
        self._chunks = list(chunks)

        self._by_lang: dict[Language, list[PolicyChunk]] = {"en": [], "ar": []}
        for c in self._chunks:
            self._by_lang[c.language].append(c)

        self._bm25: dict[Language, BM25Okapi] = {}
        self._tokens: dict[Language, list[list[str]]] = {}
        self._chunk_lookup: dict[tuple[str, str], PolicyChunk] = {
            (c.doc_id, c.chunk_id): c for c in self._chunks
        }

        for lang in ("en", "ar"):
            docs = self._by_lang[lang]
            tokens = [_tokenize(c.text) for c in docs]
            self._tokens[lang] = tokens
            self._bm25[lang] = BM25Okapi(tokens)

    def get_chunk(self, doc_id: str, chunk_id: str) -> PolicyChunk | None:
        return self._chunk_lookup.get((doc_id, chunk_id))

    def search(self, query: str, *, language: Language, top_k: int = 5) -> list[PolicyChunk]:
        docs = self._by_lang[language]
        if not docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25[language].get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
        out: list[PolicyChunk] = []
        for idx, score in ranked[: max(1, top_k)]:
            if score <= 0:
                continue
            out.append(docs[idx])

        return out
