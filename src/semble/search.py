from __future__ import annotations

import contextlib
import hashlib
import re

import bm25s
import numpy as np
import numpy.typing as npt
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.types import Chunk, SearchResult

# Stop words filtered out during symbol search
_SYMBOL_STOPS: frozenset[str] = frozenset(
    {
        "self",
        "def",
        "class",
        "return",
        "import",
        "from",
        "if",
        "else",
        "elif",
        "for",
        "in",
        "is",
        "not",
        "and",
        "or",
        "none",
        "true",
        "false",
        "try",
        "except",
        "raise",
        "with",
        "as",
        "pass",
        "the",
        "a",
        "an",
        "of",
        "to",
        "this",
        "that",
        "it",
        "how",
        "does",
        "do",
        "what",
        "where",
        "when",
        "which",
        "who",
        "are",
        "was",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "get",
        "set",
        "run",
        "main",
        "test",
        "new",
        "init",
    }
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class BM25Index:
    """Thin wrapper around bm25s."""

    def __init__(self, chunks: list[Chunk]) -> None:
        tokens = [_tokenize(c.content) for c in chunks]
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens, show_progress=False)

    def scores(self, query: str) -> npt.NDArray[np.float32]:
        """Return BM25 scores for all indexed chunks."""
        return self._bm25.get_scores(_tokenize(query))  # type: ignore[no-any-return]


class SemanticIndex:
    """Wrapper around vicinity for cosine similarity search."""

    def __init__(self, chunks: list[Chunk], embeddings: npt.NDArray[np.float32]) -> None:
        content_hashes = [c.content_hash for c in chunks]
        self._vicinity = Vicinity.from_vectors_and_items(
            embeddings, content_hashes, metric=Metric.COSINE
        )

    def query(
        self, query_embedding: npt.NDArray[np.float32], top_k: int
    ) -> list[tuple[str, float]]:
        """Return (content_hash, cosine_distance) pairs sorted by distance ascending."""
        results = self._vicinity.query(query_embedding[None], k=top_k)
        return results[0]


def search_semantic(
    query: str,
    model: StaticModel,
    semantic_index: SemanticIndex,
    hash_to_chunk: dict[str, Chunk],
    top_k: int,
) -> list[SearchResult]:
    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, top_k)
    results = []
    for content_hash, distance in hits:
        chunk = hash_to_chunk.get(content_hash)
        if chunk is not None:
            results.append(
                SearchResult(chunk=chunk, score=1.0 - float(distance), source="semantic")
            )
    return results


def search_bm25(
    query: str,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    top_k: int,
) -> list[SearchResult]:
    scores = bm25_index.scores(query)
    indices = np.argsort(-scores)[:top_k]
    return [
        SearchResult(chunk=chunks[i], score=float(scores[i]), source="bm25")
        for i in indices
        if scores[i] > 0
    ]


_DEF_RE = re.compile(r"^\s*(def |async def |class |function |func |fn |pub fn |pub async fn )")
_CONTEXT_LINES = 3


def _symbol_tokens(query: str) -> list[str]:
    """Extract identifier tokens from a query, with camelCase splitting."""
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
    tokens: list[str] = []
    for tok in raw:
        lower = tok.lower()
        if lower in _SYMBOL_STOPS or len(lower) < 2:
            continue
        tokens.append(lower)
        if "_" in tok:
            tokens.extend(p.lower() for p in tok.split("_") if len(p) >= 2)
        tokens.extend(
            p.lower() for p in re.findall(r"[A-Z]?[a-z]{2,}|[A-Z]{2,}(?=[A-Z][a-z]|\d|\b)", tok)
        )
    return list(dict.fromkeys(tokens))


def search_symbol(
    query: str,
    file_lines: dict[str, list[str]],
    top_k: int,
) -> list[SearchResult]:
    """Search for symbols using regex word-boundary matching over cached file lines.

    Extracts identifier-like tokens from the query (subword-aware), then
    searches for any of them in each file. Definition lines score higher than
    plain usages. Returns at most one result per file.

    :param query: Search query (e.g. function or class name).
    :param file_lines: Mapping from file path to list of source lines.
    :param top_k: Maximum number of results to return.
    :returns: List of search results sorted by score descending.
    """
    tokens = _symbol_tokens(query)
    if not tokens:
        return []

    patterns = []
    for tok in tokens:
        with contextlib.suppress(re.error):
            patterns.append(re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE))
    if not patterns:
        return []

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


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to [0, 1]."""
    if not scores:
        return scores
    vals = np.array(list(scores.values()), dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    denom = mx - mn if mx - mn > 1e-9 else 1e-9
    return {k: float((v - mn) / denom) for k, v in scores.items()}


def search_hybrid(
    query: str,
    model: StaticModel,
    semantic_index: SemanticIndex,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    hash_to_chunk: dict[str, Chunk],
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
    :param hash_to_chunk: Mapping from content hash to chunk.
    :param top_k: Number of results to return.
    :param alpha: Weight for semantic score (1-alpha goes to BM25). Default 0.5.
    :returns: List of search results sorted by combined score descending.
    """
    n = top_k * 3

    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, n)
    sem_raw: dict[str, float] = {}
    cmap: dict[str, Chunk] = {}
    for content_hash, distance in hits:
        chunk = hash_to_chunk.get(content_hash)
        if chunk is not None:
            sem_raw[content_hash] = 1.0 - float(distance)
            cmap[content_hash] = chunk

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
