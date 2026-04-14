import bm25s
import numpy as np
import numpy.typing as npt
from vicinity import Vicinity

from semble.ranking import apply_query_boost, rerank_topk, resolve_alpha
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, SearchMode, SearchResult

_RRF_K = 60


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _vicinity_query(index: Vicinity, embedding: npt.NDArray[np.float32], k: int) -> list[tuple[Chunk, float]]:
    """Query a Vicinity index, working around its current lack of generic typing."""
    return index.query(embedding[None], k=k)[0]  # type: ignore[return-value]


def search_semantic(
    query: str,
    model: Encoder,
    semantic_index: Vicinity,
    top_k: int,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    query_embedding = model.encode([query])[0]
    hits = _vicinity_query(semantic_index, query_embedding, top_k)
    # Vicinity returns cosine distance; convert to similarity so higher = better.
    return [
        SearchResult(chunk=chunk, score=1.0 - float(distance), source=SearchMode.SEMANTIC) for chunk, distance in hits
    ]


def search_bm25(
    query: str,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokenize(query))
    indices = np.argsort(-scores)[:top_k]
    # Exclude chunks with zero score, no query tokens matched.
    return [
        SearchResult(chunk=chunks[i], score=float(scores[i]), source=SearchMode.BM25) for i in indices if scores[i] > 0
    ]


def search_hybrid(
    query: str,
    model: Encoder,
    semantic_index: Vicinity,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted combination of semantic and BM25 scores.

    Both score sets are converted to RRF scores before combining, so alpha has
    a consistent meaning regardless of raw score magnitude.

    :param query: Search query string.
    :param model: Embedding model for semantic search.
    :param semantic_index: Pre-built semantic (vector) index.
    :param bm25_index: Pre-built BM25 index.
    :param chunks: All indexed chunks (parallel to BM25 index).
    :param top_k: Number of results to return.
    :param alpha: Weight for semantic score (1-alpha goes to BM25). None = auto-detect based on query type.
    :return: List of search results sorted by combined score descending.
    """
    alpha_weight = resolve_alpha(query, alpha)

    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    # 5x is sufficient; latency difference vs larger multipliers is negligible.
    candidate_count = top_k * 5

    query_embedding = model.encode([query])[0]
    hits = _vicinity_query(semantic_index, query_embedding, candidate_count)

    semantic_scores: dict[Chunk, float] = {chunk: 1.0 - float(distance) for chunk, distance in hits}

    bm25_scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokenize(query))
    bm25_result_scores: dict[Chunk, float] = {}
    for chunk_index in np.argsort(-bm25_scores)[:candidate_count]:
        if bm25_scores[chunk_index] > 0:
            bm25_result_scores[chunks[chunk_index]] = float(bm25_scores[chunk_index])

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_result_scores)

    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in set(normalized_semantic) | set(normalized_bm25)
    }

    combined_scores = apply_query_boost(combined_scores, query, chunks)

    ranked = rerank_topk(combined_scores, top_k, penalise_paths=alpha_weight < 1.0)
    return [SearchResult(chunk=chunk, score=score, source=SearchMode.HYBRID) for chunk, score in ranked]
