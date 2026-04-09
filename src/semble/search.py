import bm25s
import numpy as np
import numpy.typing as npt
from vicinity import Vicinity

from semble.types import Chunk, Encoder, SearchMode, SearchResult
from semble.utils import tokenize


def _normalize(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    values = np.array(list(scores.values()), dtype=np.float32)
    minimum_score = float(values.min())
    maximum_score = float(values.max())
    denominator = maximum_score - minimum_score if maximum_score - minimum_score > 1e-9 else 1e-9
    return {key: float((score - minimum_score) / denominator) for key, score in scores.items()}


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
    """Run BM25 search for a query."""
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
    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    candidate_count = top_k * 3

    query_embedding = model.encode([query])[0]
    hits = _vicinity_query(semantic_index, query_embedding, candidate_count)

    semantic_scores: dict[Chunk, float] = {}
    for chunk, distance in hits:
        semantic_scores[chunk] = 1.0 - float(distance)

    bm25_scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokenize(query))
    bm25_result_scores: dict[Chunk, float] = {}
    for chunk_index in np.argsort(-bm25_scores)[:candidate_count]:
        if bm25_scores[chunk_index] > 0:
            bm25_result_scores[chunks[chunk_index]] = float(bm25_scores[chunk_index])

    normalized_semantic_scores = _normalize(semantic_scores)
    normalized_bm25_scores = _normalize(bm25_result_scores)

    combined_scores: dict[Chunk, float] = {}
    for chunk in set(normalized_semantic_scores) | set(normalized_bm25_scores):
        combined_scores[chunk] = alpha * normalized_semantic_scores.get(chunk, 0.0) + (
            1.0 - alpha
        ) * normalized_bm25_scores.get(chunk, 0.0)

    ranked = sorted(combined_scores, key=lambda c: -combined_scores[c])[:top_k]
    return [SearchResult(chunk=chunk, score=combined_scores[chunk], source=SearchMode.HYBRID) for chunk in ranked]
