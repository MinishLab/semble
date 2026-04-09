import bm25s
import numpy as np
import numpy.typing as npt
from vicinity import Vicinity

from semble._utils import tokenize
from semble.types import Chunk, Encoder, SearchMode, SearchResult


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    values = np.array(list(scores.values()), dtype=np.float32)
    minimum_score = float(values.min())
    maximum_score = float(values.max())
    denominator = maximum_score - minimum_score if maximum_score - minimum_score > 1e-9 else 1e-9
    return {key: float((score - minimum_score) / denominator) for key, score in scores.items()}


def _vicinity_query(index: Vicinity, embedding: npt.NDArray[np.float32], k: int) -> list[tuple[Chunk, float]]:
    """Query a Vicinity index and return hits typed as Chunks.

    Vicinity's type stubs declare stored items as ``str``, but the actual
    objects are whatever was passed to ``from_vectors_and_items``.
    This wrapper isolates that stub inaccuracy in one place.
    TODO: remove once Vicinity adds generic support (Vicinity[T]).
    """
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
    # Exclude chunks with zero score — no query tokens matched.
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
    # Fetch more candidates than top_k from each index so the merged pool is large
    # enough to still surface top_k good results after union and re-ranking.
    candidate_count = top_k * 3

    query_embedding = model.encode([query])[0]
    hits = _vicinity_query(semantic_index, query_embedding, candidate_count)

    # Keyed by content_hash for O(1) merge with BM25 candidates.
    semantic_scores: dict[str, float] = {}
    chunks_by_hash: dict[str, Chunk] = {}
    for chunk, distance in hits:
        semantic_scores[chunk.content_hash] = 1.0 - float(distance)  # distance → similarity
        chunks_by_hash[chunk.content_hash] = chunk

    bm25_scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokenize(query))
    bm25_result_scores: dict[str, float] = {}
    for chunk_index in np.argsort(-bm25_scores)[:candidate_count]:
        if bm25_scores[chunk_index] > 0:  # exclude chunks with no matching tokens
            chunk_hash = chunks[chunk_index].content_hash
            bm25_result_scores[chunk_hash] = float(bm25_scores[chunk_index])
            chunks_by_hash[chunk_hash] = chunks[chunk_index]

    # Normalise each score set to [0, 1] independently so alpha has a consistent meaning.
    normalized_semantic_scores = _normalize(semantic_scores)
    normalized_bm25_scores = _normalize(bm25_result_scores)

    # Union of both candidate sets; a chunk absent from one index scores 0 for that signal.
    combined_scores: dict[str, float] = {}
    for chunk_hash in set(normalized_semantic_scores) | set(normalized_bm25_scores):
        combined_scores[chunk_hash] = alpha * normalized_semantic_scores.get(chunk_hash, 0.0) + (
            1.0 - alpha
        ) * normalized_bm25_scores.get(chunk_hash, 0.0)

    ranked_hashes = sorted(combined_scores, key=lambda chunk_hash: -combined_scores[chunk_hash])[:top_k]
    return [
        SearchResult(
            chunk=chunks_by_hash[chunk_hash],
            score=combined_scores[chunk_hash],
            source=SearchMode.HYBRID,
        )
        for chunk_hash in ranked_hashes
    ]
