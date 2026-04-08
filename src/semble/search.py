from __future__ import annotations

import hashlib
import re

import bm25s
import numpy as np
import numpy.typing as npt
from vicinity import Metric, Vicinity

from semble.types import Chunk, Encoder, SearchResult

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_DEF_RE = re.compile(r"^\s*(def |async def |class |function |func |fn |pub fn |pub async fn )")
_CONTEXT_LINES = 3


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens."""
    return _TOKEN_RE.findall(text.lower())


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score mapping to the range [0, 1]."""
    if not scores:
        return scores
    vals = np.array(list(scores.values()), dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    denom = mx - mn if mx - mn > 1e-9 else 1e-9
    return {k: float((v - mn) / denom) for k, v in scores.items()}


class BM25Index:
    """Thin wrapper around bm25s."""

    def __init__(self, texts: list[str]) -> None:
        """Build a BM25 index from texts."""
        tokens = [_tokenize(t) for t in texts]
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens, show_progress=False)

    def scores(self, query: str) -> npt.NDArray[np.float32]:
        """Return BM25 scores for all indexed chunks."""
        return self._bm25.get_scores(_tokenize(query))  # type: ignore[no-any-return]


class SemanticIndex:
    """Wrapper around vicinity for cosine similarity search."""

    def __init__(self, chunks: list[Chunk], embeddings: npt.NDArray[np.float32]) -> None:
        """Build a semantic index from chunks and embeddings."""
        self._vicinity = Vicinity.from_vectors_and_items(embeddings, chunks, metric=Metric.COSINE)

    def query(
        self, query_embedding: npt.NDArray[np.float32], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Return (chunk, cosine_distance) pairs sorted by distance ascending."""
        results = self._vicinity.query(query_embedding[None], k=top_k)
        return results[0]  # type: ignore[return-value]


def search_semantic(
    query: str,
    model: Encoder,
    semantic_index: SemanticIndex,
    top_k: int,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, top_k)
    return [
        SearchResult(chunk=chunk, score=1.0 - float(distance), source="semantic")
        for chunk, distance in hits
    ]


def search_bm25(
    query: str,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    top_k: int,
) -> list[SearchResult]:
    """Run BM25 search for a query."""
    scores = bm25_index.scores(query)
    indices = np.argsort(-scores)[:top_k]
    return [
        SearchResult(chunk=chunks[i], score=float(scores[i]), source="bm25")
        for i in indices
        if scores[i] > 0
    ]


def search_symbol(
    query: str,
    file_lines: dict[str, list[str]],
    top_k: int,
) -> list[SearchResult]:
    """Search for symbols by matching identifier tokens against file lines.

    Definition lines (def/class/function/etc.) score 1.5x higher than plain
    usages. Returns at most one result per file.

    :param query: Search query (e.g. a function or class name).
    :param file_lines: Mapping from file path to list of source lines.
    :param top_k: Maximum number of results to return.
    :returns: List of search results sorted by score descending.
    """
    tokens = list(dict.fromkeys(t.lower() for t in _TOKEN_RE.findall(query) if len(t) >= 2))
    if not tokens:
        return []

    patterns = [re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE) for tok in tokens]

    best: dict[str, tuple[float, SearchResult]] = {}

    for file_path, lines in file_lines.items():
        for lineno, line in enumerate(lines):
            matches = sum(1 for p in patterns if p.search(line))
            if matches == 0:
                continue
            is_def = bool(_DEF_RE.match(line))
            score = (matches / len(patterns)) * (1.5 if is_def else 1.0)
            if file_path in best and best[file_path][0] >= score:
                continue
            start = max(0, lineno - _CONTEXT_LINES)
            end = min(len(lines), lineno + _CONTEXT_LINES + 1)
            content = "\n".join(lines[start:end])
            chunk = Chunk(
                content=content,
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                language=None,
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            )
            best[file_path] = (score, SearchResult(chunk=chunk, score=score, source="symbol"))

    results = [sr for _, sr in best.values()]
    results.sort(key=lambda r: -r.score)
    return results[:top_k]


def search_hybrid(
    query: str,
    model: Encoder,
    semantic_index: SemanticIndex,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    top_k: int,
    alpha: float = 0.5,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted combination of semantic and BM25 scores.

    Both score sets are min-max normalized independently before combining,
    so alpha has a consistent meaning regardless of score magnitude.

    :param query: Search query string.
    :param model: Embedding model for semantic search.
    :param semantic_index: Pre-built semantic (vector) index.
    :param bm25_index: Pre-built BM25 index.
    :param chunks: All indexed chunks (parallel to BM25 index).
    :param top_k: Number of results to return.
    :param alpha: Weight for semantic score (1-alpha goes to BM25). Default 0.5.
    :returns: List of search results sorted by combined score descending.
    """
    n = top_k * 3

    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, n)
    sem_raw: dict[str, float] = {}
    cmap: dict[str, Chunk] = {}
    for chunk, distance in hits:
        sem_raw[chunk.content_hash] = 1.0 - float(distance)
        cmap[chunk.content_hash] = chunk

    bm25_scores = bm25_index.scores(query)
    bm25_raw: dict[str, float] = {}
    for idx in np.argsort(-bm25_scores)[:n]:
        if bm25_scores[idx] > 0:
            key = chunks[idx].content_hash
            bm25_raw[key] = float(bm25_scores[idx])
            cmap[key] = chunks[idx]

    sem_norm = _normalize(sem_raw)
    bm25_norm = _normalize(bm25_raw)

    combined: dict[str, float] = {}
    for key in set(sem_norm) | set(bm25_norm):
        combined[key] = alpha * sem_norm.get(key, 0.0) + (1.0 - alpha) * bm25_norm.get(key, 0.0)

    ranked = sorted(combined, key=lambda k: -combined[k])[:top_k]
    return [SearchResult(chunk=cmap[k], score=combined[k], source="hybrid") for k in ranked]
