from typing import Any
from unittest.mock import MagicMock, patch

import bm25s
import numpy as np
import numpy.typing as npt
import pytest
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend, embed_chunks, load_model
from semble.search import _sort_top_k, search_bm25, search_hybrid, search_semantic
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, SearchMode
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


def test_search_bm25(bm25: bm25s.BM25, chunks: list[Chunk]) -> None:
    """search_bm25: returns most relevant chunk first; selector restricts to given indices."""
    results = search_bm25("authenticate token", bm25, chunks, top_k=4, selector=None)
    assert len(results) > 0
    assert "authenticate" in results[0].chunk.content

    selector = np.array([len(chunks) - 1], dtype=np.int_)
    filtered = search_bm25("format", bm25, chunks, top_k=4, selector=selector)
    assert all(r.chunk is chunks[len(chunks) - 1] for r in filtered)


@pytest.mark.parametrize("query", ["", "   ", "\n\n", "zzzznonexistentterm"])
def test_bm25_returns_empty_for_no_match(bm25: bm25s.BM25, chunks: list[Chunk], query: str) -> None:
    """Empty / whitespace-only / token-less queries return [] instead of crashing bm25s."""
    assert search_bm25(query, bm25, chunks, top_k=3, selector=None) == []


def test_semantic_search(semantic: SelectableBasicBackend, chunks: list[Chunk], mock_model: Any) -> None:
    """Semantic search returns results with scores in [-1, 1]."""
    results = search_semantic("login", mock_model, semantic, chunks, top_k=3, selector=None)
    assert len(results) > 0
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_search_hybrid(
    chunks: list[Chunk], semantic: SelectableBasicBackend, bm25: bm25s.BM25, mock_model: Any
) -> None:
    """search_hybrid: returns combined results; identical content in different files produces separate results."""
    results = search_hybrid("authenticate token", mock_model, semantic, bm25, chunks, top_k=3)
    assert len(results) > 0

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

    deduped = search_hybrid("helper", mock_model, sem_index, bm25_index, all_chunks, top_k=5)
    result_locations = {r.chunk.file_path for r in deduped}
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
    """_sort_top_k returns the same indices as np.argsort(-x)[:top_k]."""
    gen = np.random.default_rng()
    x = gen.standard_normal(size=(10000,))
    top_k = 100
    indices = _sort_top_k(x, top_k)
    assert np.all(indices == np.argsort(-x)[:top_k])


@pytest.mark.parametrize(
    ("model_path", "expected_call_arg"),
    [
        (None, "minishlab/potion-code-16M"),  # default model
        ("some/custom/model", "some/custom/model"),  # explicit path forwarded
    ],
)
def test_load_model(model_path: str | None, expected_call_arg: str) -> None:
    """load_model calls from_pretrained with default or custom model path."""
    fake_model = MagicMock(spec=Encoder)
    with patch("semble.index.dense.StaticModel.from_pretrained", return_value=fake_model) as mock_fp:
        result = load_model(model_path)
    mock_fp.assert_called_once_with(expected_call_arg)
    assert result is fake_model


def test_embed_chunks_empty_returns_empty_array(mock_model: Any) -> None:
    """embed_chunks with an empty list returns a (0, 256) float32 array."""
    result = embed_chunks(mock_model, [])
    assert result.shape == (0, 256)
    assert result.dtype == np.float32


def test_selectable_basic_backend_rejects_k_below_one(
    semantic: SelectableBasicBackend, embeddings: npt.NDArray[np.float32]
) -> None:
    """SelectableBasicBackend.query guards against k < 1."""
    with pytest.raises(ValueError, match="k should be >= 1"):
        semantic.query(embeddings[:1], k=0)
