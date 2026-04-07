"""Search strategies and hybrid fusion for semble."""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s
import numpy as np
import numpy.typing as npt
from vicinity import Metric, Vicinity

from semble._tokenizer import QUERY_STOPS, tokenize_simple, tokenize_subword
from semble._types import Chunk, SearchResult, SymbolKind

if TYPE_CHECKING:
    from model2vec import StaticModel

_RRF_K = 60
_SYMBOL_WEIGHT = 2.0
_PATH_BOOST = 0.015
_DEDUP_THRESHOLD = 0.92


class BM25Index:
    """Thin wrapper around bm25s for simple and subword tokenization."""

    def __init__(self, chunks: list[Chunk]) -> None:
        simple_tokens = [tokenize_simple(c.content) for c in chunks]
        subword_tokens = [tokenize_subword(c.content) for c in chunks]

        self._simple = bm25s.BM25()
        self._simple.index(simple_tokens, show_progress=False)

        self._subword = bm25s.BM25()
        self._subword.index(subword_tokens, show_progress=False)

    def scores_simple(self, query: str) -> npt.NDArray[np.float32]:
        return self._simple.get_scores(tokenize_simple(query))  # type: ignore[no-any-return]

    def scores_subword(self, query: str) -> npt.NDArray[np.float32]:
        return self._subword.get_scores(tokenize_subword(query))  # type: ignore[no-any-return]


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
    chunks: list[Chunk],
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
    scores = bm25_index.scores_simple(query)
    indices = np.argsort(-scores)[:top_k]
    return [
        SearchResult(chunk=chunks[i], score=float(scores[i]), source="bm25")
        for i in indices
        if scores[i] > 0
    ]


_DEF_RE = re.compile(r"^\s*(def |async def |class |function |func |fn |pub fn |pub async fn )")
_CONTEXT_LINES = 3
# Tokens that are too generic to be useful as symbol search terms
_SYMBOL_STOPS = QUERY_STOPS | frozenset({"get", "set", "run", "main", "test", "new", "init"})


def search_symbol(
    query: str,
    file_lines: dict[str, list[str]],
    top_k: int,
) -> list[SearchResult]:
    """Search for symbols using regex word-boundary matching over cached file lines.

    Extracts identifier-like tokens from the query (subword-aware), then
    searches for any of them in each file. Definition lines score higher than
    plain usages. Returns at most one result per file.
    """
    # Extract identifier tokens from the query (handles camelCase and plain words)
    raw_tokens = set(tokenize_subword(query)) - _SYMBOL_STOPS
    # Keep only tokens that look like identifiers (2+ chars, not pure stop words)
    tokens = [t for t in raw_tokens if len(t) >= 2]
    if not tokens:
        return []

    # Build per-token patterns; skip tokens that fail to compile (shouldn't happen)
    patterns = []
    for tok in tokens:
        with contextlib.suppress(re.error):
            patterns.append(re.compile(r"\b" + re.escape(tok) + r"\b", re.IGNORECASE))
    if not patterns:
        return []

    best: dict[str, tuple[float, SearchResult]] = {}

    for file_path, lines in file_lines.items():
        for lineno, line in enumerate(lines):
            # Count how many query tokens appear on this line
            matches = sum(1 for p in patterns if p.search(line))
            if matches == 0:
                continue
            is_def = bool(_DEF_RE.match(line))
            # Score: fraction of tokens matched, boosted for definition lines
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
                symbol_name=query if is_def else None,
                symbol_kind=SymbolKind.FUNCTION if is_def else SymbolKind.CHUNK,
                language=None,
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            )
            best[file_path] = (score, SearchResult(chunk=chunk, score=score, source="symbol"))

    results = [sr for _, sr in best.values()]
    results.sort(key=lambda r: -r.score)
    return results[:top_k]


def search_hybrid_rrf(
    query: str,
    model: StaticModel,
    semantic_index: SemanticIndex,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    hash_to_chunk: dict[str, Chunk],
    top_k: int,
) -> list[SearchResult]:
    n = top_k * 3
    rrf: dict[str, float] = {}
    cmap: dict[str, Chunk] = {}

    # Semantic (cosine distance — lower is better)
    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, n)
    for rank, (content_hash, _distance) in enumerate(hits):
        chunk = hash_to_chunk.get(content_hash)
        if chunk is not None:
            rrf[content_hash] = rrf.get(content_hash, 0) + 1.0 / (_RRF_K + rank + 1)
            cmap[content_hash] = chunk

    # BM25 (subword tokenizer for better recall in hybrid)
    scores = bm25_index.scores_subword(query)
    for rank, idx in enumerate(np.argsort(-scores)[:n]):
        if scores[idx] > 0:
            key = chunks[idx].content_hash
            rrf[key] = rrf.get(key, 0) + 1.0 / (_RRF_K + rank + 1)
            cmap[key] = chunks[idx]

    # Symbol (strong boost — 2x weight)
    qt = set(tokenize_subword(query)) - QUERY_STOPS
    sym_hits: list[tuple[Chunk, float]] = []
    for chunk in chunks:
        if chunk.symbol_name:
            st = set(tokenize_subword(chunk.symbol_name)) - QUERY_STOPS
            overlap = qt & st
            if overlap:
                sym_hits.append((chunk, len(overlap) / max(len(qt), 1)))
    sym_hits.sort(key=lambda x: -x[1])
    for rank, (chunk, _) in enumerate(sym_hits[:n]):
        key = chunk.content_hash
        rrf[key] = rrf.get(key, 0) + _SYMBOL_WEIGHT / (_RRF_K + rank + 1)
        cmap[key] = chunk

    # Path boost
    for key, chunk in cmap.items():
        path_terms: set[str] = set()
        for part in Path(chunk.file_path).parts:
            path_terms.update(t.lower() for t in re.findall(r"[a-zA-Z]{2,}", Path(part).stem))
        overlap = qt & path_terms
        if overlap:
            rrf[key] += len(overlap) * _PATH_BOOST

    ranked = sorted(rrf, key=lambda k: -rrf[k])[:top_k]
    return [SearchResult(chunk=cmap[k], score=rrf[k], source="hybrid_rrf") for k in ranked]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a score dict to [0, 1]."""
    if not scores:
        return scores
    vals = np.array(list(scores.values()), dtype=np.float32)
    mn, mx = float(vals.min()), float(vals.max())
    denom = mx - mn if mx - mn > 1e-9 else 1e-9
    return {k: float((v - mn) / denom) for k, v in scores.items()}


def search_hybrid_alpha(
    query: str,
    model: StaticModel,
    semantic_index: SemanticIndex,
    bm25_index: BM25Index,
    chunks: list[Chunk],
    hash_to_chunk: dict[str, Chunk],
    top_k: int,
    alpha: float = 0.7,
) -> list[SearchResult]:
    """Hybrid search using alpha-weighted linear combination of normalized scores.

    :param query: Search query string.
    :param model: Embedding model for semantic search.
    :param semantic_index: Pre-built semantic (vector) index.
    :param bm25_index: Pre-built BM25 index.
    :param chunks: All indexed chunks (parallel to BM25 index).
    :param hash_to_chunk: Mapping from content hash to chunk.
    :param top_k: Number of results to return.
    :param alpha: Weight for the semantic score (1-alpha goes to BM25).
        Higher alpha = more semantic, lower = more keyword. Default 0.7.
    :returns: List of search results sorted by combined score descending.
    """
    n = top_k * 3

    # Semantic scores: vicinity returns (hash, cosine_distance); convert to similarity
    qe = model.encode([query])[0]
    hits = semantic_index.query(qe, n)
    sem_raw: dict[str, float] = {}
    cmap: dict[str, Chunk] = {}
    for content_hash, distance in hits:
        chunk = hash_to_chunk.get(content_hash)
        if chunk is not None:
            sem_raw[content_hash] = 1.0 - float(distance)
            cmap[content_hash] = chunk

    # BM25 scores (subword tokenizer for better recall)
    bm25_scores = bm25_index.scores_subword(query)
    bm25_raw: dict[str, float] = {}
    for idx in np.argsort(-bm25_scores)[:n]:
        if bm25_scores[idx] > 0:
            key = chunks[idx].content_hash
            bm25_raw[key] = float(bm25_scores[idx])
            cmap[key] = chunks[idx]

    # Normalize each set independently, then combine
    sem_norm = _normalize(sem_raw)
    bm25_norm = _normalize(bm25_raw)

    combined: dict[str, float] = {}
    for key in set(sem_norm) | set(bm25_norm):
        combined[key] = alpha * sem_norm.get(key, 0.0) + (1.0 - alpha) * bm25_norm.get(key, 0.0)

    ranked = sorted(combined, key=lambda k: -combined[k])[:top_k]
    return [SearchResult(chunk=cmap[k], score=combined[k], source="hybrid") for k in ranked]


def dedup_results(
    results: list[SearchResult], embedding_cache: dict[str, npt.NDArray[np.float32]]
) -> list[SearchResult]:
    """Remove near-duplicate results by embedding cosine similarity."""
    if len(results) <= 1:
        return results

    kept = [results[0]]
    kept_hashes = {results[0].chunk.content_hash}

    for r in results[1:]:
        if r.chunk.content_hash in kept_hashes:
            continue
        r_emb = embedding_cache.get(r.chunk.content_hash)
        if r_emb is None:
            kept.append(r)
            kept_hashes.add(r.chunk.content_hash)
            continue
        is_dup = False
        for kr in kept:
            kr_emb = embedding_cache.get(kr.chunk.content_hash)
            if kr_emb is not None:
                sim = float(
                    np.dot(r_emb, kr_emb) / (np.linalg.norm(r_emb) * np.linalg.norm(kr_emb) + 1e-8)
                )
                if sim > _DEDUP_THRESHOLD:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(r)
            kept_hashes.add(r.chunk.content_hash)

    return kept
