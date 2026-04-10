from typing import Any

import bm25s
import numpy as np
import numpy.typing as npt
import pytest
from vicinity import Metric, Vicinity

from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, SearchMode
from semble.utils import tokenize


def _make_chunk(content: str, file_path: str = "file.py") -> Chunk:
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        language="python",
        content_hash=content[:16],  # stable stand-in; tests don't rely on hash correctness
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    """Four small code chunks covering authentication, login, user service, and utils."""
    return [
        _make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        _make_chunk("def login(username, password):\n    pass", "auth.py"),
        _make_chunk("class UserService:\n    pass", "users.py"),
        _make_chunk("def format_date(dt):\n    return str(dt)", "utils.py"),
    ]


@pytest.fixture
def embeddings(chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Deterministic random unit-norm embeddings for the chunks fixture."""
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(chunks), 256)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized: npt.NDArray[np.float32] = embs / (norms + 1e-8)
    return normalized


@pytest.fixture
def bm25(chunks: list[Chunk]) -> bm25s.BM25:
    """Pre-built BM25 index over the chunks fixture."""
    index = bm25s.BM25()
    index.index([tokenize(chunk.content) for chunk in chunks], show_progress=False)
    return index


@pytest.fixture
def semantic(chunks: list[Chunk], embeddings: npt.NDArray[np.float32]) -> Vicinity:
    """Pre-built ANNS index over the chunks fixture."""
    return Vicinity.from_vectors_and_items(embeddings, chunks, metric=Metric.COSINE)


def test_bm25_search(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """BM25 returns results with the most relevant chunk first."""
    results = search_bm25("authenticate token", bm25, chunks, top_k=4)
    assert len(results) > 0
    assert "authenticate" in results[0].chunk.content


def test_bm25_no_results_for_garbage(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """Query with no matching tokens returns an empty list."""
    results = search_bm25("zzzznonexistentterm", bm25, chunks, top_k=3)
    assert results == []


def test_semantic_search(semantic: Vicinity, mock_model: Any) -> None:
    """Semantic search returns results with scores in [-1, 1]."""
    results = search_semantic("login", mock_model, semantic, top_k=3)
    assert len(results) > 0
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_hybrid_returns_results(chunks: list[Chunk], semantic: Vicinity, bm25: bm25s.BM25, mock_model: Any) -> None:
    """Hybrid search returns results combining semantic and BM25 signals."""
    results = search_hybrid("authenticate token", mock_model, semantic, bm25, chunks, top_k=3)
    assert len(results) > 0


def test_hybrid_keeps_both_locations_for_identical_content(mock_model: Any) -> None:
    """Identical chunk content in different files produces two distinct results."""
    shared_content = "def helper():\n    pass"
    chunk_a = _make_chunk(shared_content, "module_a.py")
    chunk_b = _make_chunk(shared_content, "module_b.py")
    all_chunks = [chunk_a, chunk_b]

    rng = np.random.default_rng(1)
    embs = rng.standard_normal((2, 256)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8

    sem_index = Vicinity.from_vectors_and_items(embs, all_chunks, metric=Metric.COSINE)
    bm25_index = bm25s.BM25()
    bm25_index.index([tokenize(c.content) for c in all_chunks], show_progress=False)

    results = search_hybrid("helper", mock_model, sem_index, bm25_index, all_chunks, top_k=5)
    result_locations = {r.chunk.file_path for r in results}
    assert "module_a.py" in result_locations
    assert "module_b.py" in result_locations


@pytest.mark.parametrize(
    ("mode", "query", "top_k"),
    [
        (SearchMode.BM25, "authenticate", 3),
        (SearchMode.SEMANTIC, "query", 4),
        (SearchMode.HYBRID, "login", 4),
    ],
)
def test_search_source_labels(
    mode: SearchMode,
    query: str,
    top_k: int,
    chunks: list[Chunk],
    semantic: Vicinity,
    bm25: bm25s.BM25,
    mock_model: Any,
) -> None:
    """Each result carries a source label matching the search mode used."""
    if mode is SearchMode.BM25:
        results = search_bm25(query, bm25, chunks, top_k)
    elif mode is SearchMode.SEMANTIC:
        results = search_semantic(query, mock_model, semantic, top_k)
    else:
        results = search_hybrid(query, mock_model, semantic, bm25, chunks, top_k)
    assert len(results) > 0
    assert all(result.source is mode for result in results)
