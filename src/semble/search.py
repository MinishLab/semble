import bm25s
import numpy as np
import numpy.typing as npt
from pyversity import Strategy, diversify

from semble.index.dense import SelectableBasicBackend
from semble.index.sparse import selector_to_mask
from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, SearchResult

_RRF_K = 60
DEFAULT_DOCS_DIVERSITY = 0.3


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def _search_semantic(
    query: str,
    model: Encoder,
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    query_embedding = model.encode([query])
    indices, scores = semantic_index.query(query_embedding, k=top_k, selector=selector)[0]
    # Vicinity returns cosine distance; convert to similarity so higher = better.
    return [SearchResult(chunk=chunks[index], score=1.0 - float(distance)) for index, distance in zip(indices, scores)]


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    """Get the top k indices of an array in sort order."""
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def _search_bm25(
    query: str,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    tokens = tokenize(query)
    if not tokens:
        return []
    mask = selector_to_mask(selector, len(chunks))
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokens, weight_mask=mask)
    indices = _sort_top_k(scores, top_k)

    # Exclude chunks with zero score, no query tokens matched.
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in indices if scores[i] > 0]


def search(
    query: str,
    model: Encoder,
    semantic_index: SelectableBasicBackend,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
    rerank: bool = True,
    diversity: float | None = None,
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
    :param selector: Optional array of chunk indices to filter results by.
    :param rerank: Whether to apply code-tuned reranking (file boost, identifier boost, path penalties).
    :param diversity: DPP diversity weight in [0, 1]. When set, fetches 2× candidates, reranks,
        then re-selects with pyversity DPP. None disables diversity.
    :return: List of search results sorted by combined score descending.
    """
    alpha_weight = resolve_alpha(query, alpha)

    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    candidate_count = top_k * 5

    semantic = _search_semantic(query, model, semantic_index, chunks, candidate_count, selector)
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}
    bm25_scores = {}
    for result in _search_bm25(query, bm25_index, chunks, candidate_count, selector):
        if result.score:
            bm25_scores[result.chunk] = result.score

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)

    # Sort by the file path and start line to counteract randomness from hashing.
    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda c: c.start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }

    # Over-fetch before reranking so diversity has candidates to choose from.
    fetch_k = top_k * 2 if diversity is not None else top_k

    if rerank:
        boost_multi_chunk_files(combined_scores)
        combined_scores = apply_query_boost(combined_scores, query, chunks)
        ranked = rerank_topk(combined_scores, fetch_k, penalise_paths=alpha_weight < 1.0)
    else:
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:fetch_k]

    results = [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]

    if diversity is not None:
        return _diversify(results, top_k, diversity, semantic_index, chunks)
    return results


def _diversify(
    results: list[SearchResult],
    top_k: int,
    diversity_weight: float,
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
) -> list[SearchResult]:
    """Re-rank results with DPP to improve embedding-space diversity."""
    if len(results) <= top_k:
        return results
    chunk_index = {c: i for i, c in enumerate(chunks)}
    valid = [r for r in results if r.chunk in chunk_index]
    if len(valid) <= top_k:
        return valid
    indices = np.array([chunk_index[r.chunk] for r in valid])
    scores = np.array([r.score for r in valid], dtype=np.float32)
    result = diversify(
        embeddings=semantic_index.vectors[indices],
        scores=scores,
        k=top_k,
        strategy=Strategy.DPP,
        diversity=diversity_weight,
    )
    return sorted((valid[i] for i in result.indices), key=lambda r: -r.score)
