from collections.abc import Sequence

import bm25s
import numpy as np
import numpy.typing as npt
from model2vec import StaticModel

from semble.index.dense import SelectableBasicBackend
from semble.index.sparse import Bm25sSparseIndex, SparseIndex
from semble.ranking import apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha
from semble.types import Chunk, FilterSpec, SearchResult

_RRF_K = 60


ChunkIdentity = tuple[str, int]


def _chunk_id(chunk: Chunk, index: int) -> int:
    return chunk.chunk_id if chunk.chunk_id is not None else index


def _matches_filter_spec(filter_spec: FilterSpec, chunk: Chunk, index: int) -> bool:
    if filter_spec.file_paths is not None and chunk.file_path not in filter_spec.file_paths:
        return False
    if filter_spec.languages is not None and chunk.language not in filter_spec.languages:
        return False
    if filter_spec.chunk_ids is not None and _chunk_id(chunk, index) not in filter_spec.chunk_ids:
        return False
    return True


def _filter_spec_to_semantic_selector(
    filter_spec: FilterSpec | None, chunks: Sequence[Chunk]
) -> npt.NDArray[np.int_] | None:
    if filter_spec is None:
        return None
    if filter_spec.file_paths is None and filter_spec.languages is None and filter_spec.chunk_ids is not None:
        return np.array(sorted(filter_spec.chunk_ids), dtype=np.int_)
    if filter_spec.file_paths is not None and filter_spec.languages is None and filter_spec.chunk_ids is None:
        chunk_ids_for_paths = getattr(chunks, "chunk_ids_for_paths", None)
        if chunk_ids_for_paths is not None:
            return np.array(sorted(chunk_ids_for_paths(filter_spec.file_paths)), dtype=np.int_)
    if filter_spec.languages is not None and filter_spec.file_paths is None and filter_spec.chunk_ids is None:
        chunk_ids_for_languages = getattr(chunks, "chunk_ids_for_languages", None)
        if chunk_ids_for_languages is not None:
            chunk_ids = chunk_ids_for_languages(filter_spec.languages)
            if chunk_ids is not None:
                return np.array(sorted(chunk_ids), dtype=np.int_)
    chunk_ids = [
        _chunk_id(chunk, index) for index, chunk in enumerate(chunks) if _matches_filter_spec(filter_spec, chunk, index)
    ]
    return np.array(chunk_ids, dtype=np.int_)


def _semantic_chunks_by_id(chunks: Sequence[Chunk]) -> dict[int, Chunk]:
    return {_chunk_id(chunk, index): chunk for index, chunk in enumerate(chunks)}


def _chunk_identity(chunk: Chunk, chunk_positions: dict[Chunk, int]) -> ChunkIdentity:
    if chunk.chunk_id is not None:
        return ("chunk_id", chunk.chunk_id)
    return ("position", chunk_positions[chunk])


def _candidate_identity_maps(
    chunks: Sequence[Chunk], results: list[SearchResult]
) -> tuple[dict[Chunk, int], dict[ChunkIdentity, Chunk]]:
    chunk_lookup = getattr(chunks, "chunk_by_id", None)
    if chunk_lookup is not None:
        return {}, {_chunk_identity(result.chunk, {}): result.chunk for result in results}

    chunk_positions = {chunk: index for index, chunk in enumerate(chunks)}
    chunks_by_identity = {_chunk_identity(chunk, chunk_positions): chunk for chunk in chunks}
    for result in results:
        chunks_by_identity.setdefault(_chunk_identity(result.chunk, chunk_positions), result.chunk)
    return chunk_positions, chunks_by_identity


def _scores_by_identity(results: list[SearchResult], chunk_positions: dict[Chunk, int]) -> dict[ChunkIdentity, float]:
    scores: dict[ChunkIdentity, float] = {}
    for result in results:
        identity = _chunk_identity(result.chunk, chunk_positions)
        if identity not in scores or result.score > scores[identity]:
            scores[identity] = result.score
    return scores


def _rrf_scores(scores: dict[ChunkIdentity, float]) -> dict[ChunkIdentity, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda identity: -scores[identity])
    return {identity: 1.0 / (_RRF_K + rank) for rank, identity in enumerate(ranked, 1)}


def _max_identity_scores(items: Sequence[tuple[ChunkIdentity, float]]) -> dict[ChunkIdentity, float]:
    scores: dict[ChunkIdentity, float] = {}
    for identity, score in items:
        if identity not in scores or score > scores[identity]:
            scores[identity] = score
    return scores


def _semantic_id_scores(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None,
    query_embedding: npt.NDArray[np.float32] | None = None,
) -> list[tuple[int, float]]:
    if query_embedding is None:
        query_embedding = model.encode([query])
    selector = _filter_spec_to_semantic_selector(filter_spec, chunks)
    if selector is not None and len(selector) == 0:
        return []

    indices, distances = semantic_index.query(query_embedding, k=top_k, selector=selector)[0]
    # Vicinity returns cosine distance; convert to similarity so higher = better.
    return [(int(index), 1.0 - float(distance)) for index, distance in zip(indices, distances)]


def _search_semantic_identity_scores(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None,
    query_embedding: npt.NDArray[np.float32] | None = None,
) -> dict[ChunkIdentity, float]:
    id_scores = _semantic_id_scores(query, model, semantic_index, chunks, top_k, filter_spec, query_embedding)
    chunk_lookup = getattr(chunks, "chunk_by_id", None)
    if chunk_lookup is not None:
        return _max_identity_scores([(("chunk_id", chunk_id), score) for chunk_id, score in id_scores])

    chunk_positions = {chunk: index for index, chunk in enumerate(chunks)}
    chunks_by_id = _semantic_chunks_by_id(chunks)
    return _max_identity_scores(
        [(_chunk_identity(chunks_by_id[chunk_id], chunk_positions), score) for chunk_id, score in id_scores]
    )


def _search_semantic(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None = None,
    query_embedding: npt.NDArray[np.float32] | None = None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    id_scores = _semantic_id_scores(query, model, semantic_index, chunks, top_k, filter_spec, query_embedding)
    chunk_lookup = getattr(chunks, "chunk_by_id", None)
    if chunk_lookup is not None:
        return [SearchResult(chunk=chunk_lookup(chunk_id), score=score) for chunk_id, score in id_scores]

    chunks_by_id = _semantic_chunks_by_id(chunks)
    return [SearchResult(chunk=chunks_by_id[chunk_id], score=score) for chunk_id, score in id_scores]


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
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None = None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    return _search_sparse(query, bm25_index, chunks, top_k, filter_spec)


def _search_sparse(
    query: str,
    sparse_index: bm25s.BM25 | SparseIndex,
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None,
) -> list[SearchResult]:
    if isinstance(sparse_index, bm25s.BM25):
        sparse_index = Bm25sSparseIndex(sparse_index, chunks)
    return sparse_index.search(query, top_k, filter_spec)


def _search_sparse_identity_scores(
    query: str,
    sparse_index: bm25s.BM25 | SparseIndex,
    chunks: Sequence[Chunk],
    top_k: int,
    filter_spec: FilterSpec | None,
) -> dict[ChunkIdentity, float]:
    if isinstance(sparse_index, bm25s.BM25):
        sparse_index = Bm25sSparseIndex(sparse_index, chunks)
    search_ids = getattr(sparse_index, "search_ids", None)
    if search_ids is not None:
        return _max_identity_scores(
            [
                (("chunk_id", int(chunk_id)), float(score))
                for chunk_id, score in search_ids(query, top_k, filter_spec)
                if score
            ]
        )

    results = [result for result in sparse_index.search(query, top_k, filter_spec) if result.score]
    chunk_positions, _ = _candidate_identity_maps(chunks, results)
    return _scores_by_identity(results, chunk_positions)


def search(
    query: str,
    model: StaticModel,
    semantic_index: SelectableBasicBackend,
    bm25_index: bm25s.BM25 | SparseIndex,
    chunks: Sequence[Chunk],
    top_k: int,
    alpha: float | None = None,
    filter_spec: FilterSpec | None = None,
    rerank: bool = True,
    query_embedding: npt.NDArray[np.float32] | None = None,
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
    :param filter_spec: Optional backend-neutral filters for language, path, or chunk IDs.
    :param rerank: Whether to perform code-tuned reranking. On by default for code search, off for docs search.
    :return: List of search results sorted by combined score descending.
    """
    alpha_weight = resolve_alpha(query, alpha)

    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    candidate_count = top_k * 5

    chunk_lookup = getattr(chunks, "chunk_by_id", None)
    chunks_lookup = getattr(chunks, "chunks_by_id", None)
    if chunk_lookup is not None:
        semantic_scores = _search_semantic_identity_scores(
            query,
            model,
            semantic_index,
            chunks,
            candidate_count,
            filter_spec,
            query_embedding,
        )
        bm25_scores = _search_sparse_identity_scores(query, bm25_index, chunks, candidate_count, filter_spec)
        normalized_semantic = _rrf_scores(semantic_scores)
        normalized_bm25 = _rrf_scores(bm25_scores)
        combined_scores = {
            identity: alpha_weight * normalized_semantic.get(identity, 0.0)
            + (1.0 - alpha_weight) * normalized_bm25.get(identity, 0.0)
            for identity in {*normalized_semantic, *normalized_bm25}
        }
        if not rerank:
            ranked = [
                (identity, score)
                for identity, score in sorted(combined_scores.items(), key=lambda item: (-item[1], item[0]))
                if identity[0] == "chunk_id"
            ][:top_k]
            if chunks_lookup is not None:
                ranked_chunks = chunks_lookup([identity[1] for identity, _ in ranked])
                return [SearchResult(chunk=chunk, score=score) for chunk, (_, score) in zip(ranked_chunks, ranked)]
            return [SearchResult(chunk=chunk_lookup(identity[1]), score=score) for identity, score in ranked]

        ranked = [
            (identity, score)
            for identity, score in sorted(combined_scores.items(), key=lambda item: (-item[1], item[0]))
            if identity[0] == "chunk_id" and score > 0.0
        ][:candidate_count]
        if chunks_lookup is not None:
            ranked_chunks = chunks_lookup([identity[1] for identity, _ in ranked])
            candidate_items = list(zip(ranked_chunks, [score for _, score in ranked]))
        else:
            candidate_items = [(chunk_lookup(identity[1]), score) for identity, score in ranked]
        candidate_items.sort(key=lambda item: item[0].start_line)
        combined_scores_by_chunk = {chunk: score for chunk, score in candidate_items}
        if not combined_scores_by_chunk:
            return []
        boost_multi_chunk_files(combined_scores_by_chunk)
        combined_scores_by_chunk = apply_query_boost(
            combined_scores_by_chunk,
            query,
            list(combined_scores_by_chunk),
        )
        reranked = rerank_topk(combined_scores_by_chunk, top_k, penalise_paths=alpha_weight < 1.0)
        return [SearchResult(chunk=chunk, score=score) for chunk, score in reranked]

    semantic = _search_semantic(query, model, semantic_index, chunks, candidate_count, filter_spec, query_embedding)
    sparse = [
        result for result in _search_sparse(query, bm25_index, chunks, candidate_count, filter_spec) if result.score
    ]
    chunk_positions, chunks_by_identity = _candidate_identity_maps(chunks, [*semantic, *sparse])
    semantic_scores = _scores_by_identity(semantic, chunk_positions)
    bm25_scores = _scores_by_identity(sparse, chunk_positions)

    normalized_semantic = _rrf_scores(semantic_scores)
    normalized_bm25 = _rrf_scores(bm25_scores)

    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda identity: chunks_by_identity[identity].start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunks_by_identity[identity]: alpha_weight * normalized_semantic.get(identity, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(identity, 0.0)
        for identity in all_candidates
    }

    if rerank:
        # Boost files with multiple relevant chunks.
        boost_multi_chunk_files(combined_scores)
        # Boost queries with specific identifiers in them.
        boost_chunks = list(combined_scores) if getattr(chunks, "chunk_by_id", None) is not None else chunks
        combined_scores = apply_query_boost(combined_scores, query, boost_chunks)
        # Rerank the top-k results by applying path-based penalties.
        ranked = rerank_topk(combined_scores, top_k, penalise_paths=alpha_weight < 1.0)
    else:
        sorted_by_score = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ranked = sorted_by_score[:top_k]
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]
