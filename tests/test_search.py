from typing import Any

import bm25s
import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend
from semble.search import _sort_top_k, search_bm25, search_hybrid, search_semantic
from semble.tokens import tokenize
from semble.types import Chunk, SearchMode
from tests.conftest import make_chunk


@pytest.fixture
def chunks() -> list[Chunk]:
    """Four small code chunks covering authentication, login, user service, and utils."""
    return [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        make_chunk("def login(username, password):\n    pass", "auth.py"),
        make_chunk("class UserService:\n    pass", "users.py"),
        make_chunk("def format_date(dt):\n    return str(dt)", "utils.py"),
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
def semantic(embeddings: npt.NDArray[np.float32]) -> SelectableBasicBackend:
    """Pre-built ANNS index over the chunks fixture."""
    return SelectableBasicBackend(embeddings, BasicArgs())


def test_bm25_search(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """BM25 returns results with the most relevant chunk first."""
    results = search_bm25("authenticate token", bm25, chunks, top_k=4, selector=None)
    assert len(results) > 0
    assert "authenticate" in results[0].chunk.content


def test_bm25_no_results_for_garbage(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """Query with no matching tokens returns an empty list."""
    results = search_bm25("zzzznonexistentterm", bm25, chunks, top_k=3, selector=None)
    assert results == []


def test_semantic_search(semantic: SelectableBasicBackend, chunks: list[Chunk], mock_model: Any) -> None:
    """Semantic search returns results with scores in [-1, 1]."""
    results = search_semantic("login", mock_model, semantic, chunks, top_k=3, selector=None)
    assert len(results) > 0
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_hybrid_returns_results(
    chunks: list[Chunk], semantic: SelectableBasicBackend, bm25: bm25s.BM25, mock_model: Any
) -> None:
    """Hybrid search returns results combining semantic and BM25 signals."""
    results = search_hybrid("authenticate token", mock_model, semantic, bm25, chunks, top_k=3)
    assert len(results) > 0


def test_hybrid_keeps_both_locations_for_identical_content(mock_model: Any) -> None:
    """Identical chunk content in different files produces two distinct results."""
    shared_content = "def helper():\n    pass"
    chunk_a = make_chunk(shared_content, "module_a.py")
    chunk_b = make_chunk(shared_content, "module_b.py")
    all_chunks = [chunk_a, chunk_b]

    rng = np.random.default_rng(1)
    embs = rng.standard_normal((2, 256)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8

    sem_index = SelectableBasicBackend(embs, BasicArgs())
    bm25_index = bm25s.BM25()
    bm25_index.index([tokenize(c.content) for c in all_chunks], show_progress=False)

    results = search_hybrid("helper", mock_model, sem_index, bm25_index, all_chunks, top_k=5)
    result_locations = {r.chunk.file_path for r in results}
    assert "module_a.py" in result_locations
    assert "module_b.py" in result_locations


@pytest.mark.parametrize(
    ("search_fn", "mode", "query", "top_k"),
    [
        (lambda q, m, s, b, c, k: search_bm25(q, b, c, k, selector=None), SearchMode.BM25, "authenticate", 3),
        (lambda q, m, s, b, c, k: search_semantic(q, m, s, c, k, selector=None), SearchMode.SEMANTIC, "query", 4),
        (lambda q, m, s, b, c, k: search_hybrid(q, m, s, b, c, k), SearchMode.HYBRID, "login", 4),
    ],
)
def test_search_source_labels(
    search_fn: Any,
    mode: SearchMode,
    query: str,
    top_k: int,
    chunks: list[Chunk],
    semantic: SelectableBasicBackend,
    bm25: bm25s.BM25,
    mock_model: Any,
) -> None:
    """Each result carries a source label matching the search mode used."""
    results = search_fn(query, mock_model, semantic, bm25, chunks, top_k)
    assert len(results) > 0
    assert all(result.source is mode for result in results)


def test_sort_top_k() -> None:
    """Test that the sort top k is a faster version of argsort."""
    gen = np.random.default_rng()
    x = gen.standard_normal(size=(10000,))
    top_k = 100
    indices = _sort_top_k(x, top_k)
    assert np.all(indices == np.argsort(-x)[:top_k])


def test_bm25_with_selector_high_indices(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """BM25 with a selector whose indices exceed len(selector) does not crash."""
    selector = np.array([len(chunks) - 1], dtype=np.int_)
    results = search_bm25("format", bm25, chunks, top_k=4, selector=selector)
    assert all(r.chunk is chunks[len(chunks) - 1] for r in results)


@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_bm25_empty_query_returns_empty(bm25: bm25s.BM25, chunks: list[Chunk], query: str) -> None:
    """Empty / whitespace-only queries return [] instead of crashing bm25s."""
    assert search_bm25(query, bm25, chunks, top_k=3, selector=None) == []
