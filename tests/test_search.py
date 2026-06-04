from typing import Any
from unittest.mock import MagicMock, patch

import bm25s
import numpy as np
import numpy.typing as npt
import pytest
from model2vec import StaticModel
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend, embed_chunks, load_model
from semble.index.semantic_backend import StableIdSemanticBackend
from semble.search import _search_bm25, _search_semantic, _sort_top_k, search
from semble.tokens import tokenize
from semble.types import Chunk, FilterSpec, SearchResult
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
    """search_bm25: returns most relevant chunk first; FilterSpec restricts candidates."""
    results = _search_bm25("authenticate token", bm25, chunks, top_k=4)
    assert len(results) > 0
    assert "authenticate" in results[0].chunk.content

    filter_spec = FilterSpec(file_paths=frozenset({"utils.py"}))
    filtered = _search_bm25("format", bm25, chunks, top_k=4, filter_spec=filter_spec)
    assert all(r.chunk is chunks[len(chunks) - 1] for r in filtered)


@pytest.mark.parametrize("query", ["", "   ", "\n\n", "zzzznonexistentterm"])
def test_bm25_returns_empty_for_no_match(bm25: bm25s.BM25, chunks: list[Chunk], query: str) -> None:
    """Empty / whitespace-only / token-less queries return [] instead of crashing bm25s."""
    assert _search_bm25(query, bm25, chunks, top_k=3, filter_spec=None) == []


def test_semantic_search(semantic: SelectableBasicBackend, chunks: list[Chunk], mock_model: Any) -> None:
    """Semantic search returns results with scores in [-1, 1]."""
    results = _search_semantic("login", mock_model, semantic, chunks, top_k=3, filter_spec=None)
    assert len(results) > 0
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_semantic_search_applies_filter_spec(
    semantic: SelectableBasicBackend, chunks: list[Chunk], mock_model: Any
) -> None:
    """Dense filtering should be expressed as FilterSpec before backend-specific selector conversion."""
    filter_spec = FilterSpec(file_paths=frozenset({"utils.py"}))

    results = _search_semantic("login", mock_model, semantic, chunks, top_k=3, filter_spec=filter_spec)

    assert [result.chunk.file_path for result in results] == ["utils.py"]


def test_semantic_search_maps_backend_chunk_ids_to_canonical_chunks(mock_model: Any) -> None:
    """Dense backend IDs should resolve through stable chunk IDs, not list positions."""
    chunks = [
        Chunk("def old():\n    pass", "old.py", 1, 2, "python", chunk_id=100),
        Chunk("def current():\n    pass", "current.py", 10, 11, "python", chunk_id=101),
    ]

    class StableIdSemanticIndex:
        def query(
            self,
            vectors: npt.NDArray,
            k: int,
            selector: npt.NDArray[np.int_] | None = None,
        ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]]:
            return [(np.array([101]), np.array([0.25], dtype=np.float32))]

    results = _search_semantic("current", mock_model, StableIdSemanticIndex(), chunks, top_k=1)

    assert [result.chunk for result in results] == [chunks[1]]


def test_semantic_search_filter_spec_passes_stable_chunk_ids(mock_model: Any) -> None:
    """Dense filters should pass stable chunk IDs so compacted dense rows remain addressable."""
    chunks = [
        Chunk("def old():\n    pass", "old.py", 1, 2, "python", chunk_id=100),
        Chunk("def current():\n    pass", "current.py", 10, 11, "python", chunk_id=101),
    ]

    class CapturingSemanticIndex:
        seen_selector: npt.NDArray[np.int_] | None = None

        def query(
            self,
            vectors: npt.NDArray,
            k: int,
            selector: npt.NDArray[np.int_] | None = None,
        ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]]:
            self.seen_selector = selector
            return [(np.array([101]), np.array([0.25], dtype=np.float32))]

    semantic = CapturingSemanticIndex()
    results = _search_semantic(
        "current",
        mock_model,
        semantic,
        chunks,
        top_k=1,
        filter_spec=FilterSpec(file_paths=frozenset({"current.py"})),
    )

    np.testing.assert_array_equal(semantic.seen_selector, np.array([101]))
    assert [result.chunk for result in results] == [chunks[1]]


def test_semantic_search_returns_empty_for_unmatched_filter_spec(
    semantic: SelectableBasicBackend, chunks: list[Chunk], mock_model: Any
) -> None:
    """Dense filters with no matching chunks should return [] instead of querying an empty selector."""
    results = _search_semantic(
        "login",
        mock_model,
        semantic,
        chunks,
        top_k=3,
        filter_spec=FilterSpec(file_paths=frozenset({"missing.py"})),
    )

    assert results == []


def test_search_hybrid(
    chunks: list[Chunk], semantic: SelectableBasicBackend, bm25: bm25s.BM25, mock_model: Any
) -> None:
    """search_hybrid: returns combined results; identical content in different files produces separate results."""
    results = search("authenticate token", mock_model, semantic, bm25, chunks, top_k=3)
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

    deduped = search("helper", mock_model, sem_index, bm25_index, all_chunks, top_k=5)
    result_locations = {r.chunk.file_path for r in deduped}
    assert "module_a.py" in result_locations
    assert "module_b.py" in result_locations


def test_search_accepts_sparse_index_protocol(
    chunks: list[Chunk], semantic: SelectableBasicBackend, mock_model: Any
) -> None:
    """Hybrid search can use a sparse backend without relying on bm25s internals."""

    class FakeSparseIndex:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, FilterSpec | None]] = []

        def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
            self.calls.append((query, top_k, filter_spec))
            return [SearchResult(chunks[0], 42.0)]

    sparse_index = FakeSparseIndex()

    filter_spec = FilterSpec(file_paths=frozenset({"auth.py"}))
    results = search(
        "authenticate token",
        mock_model,
        semantic,
        sparse_index,
        chunks,
        top_k=1,
        alpha=0.0,
        filter_spec=filter_spec,
        rerank=False,
    )

    assert sparse_index.calls == [("authenticate token", 5, filter_spec)]
    assert results[0].chunk is chunks[0]


def test_hybrid_merge_uses_chunk_id_identity(mock_model: Any) -> None:
    """Dense/sparse hits for one stable chunk ID should merge even when Chunk objects differ."""
    canonical = Chunk("def authenticate(token):\n    return token", "auth.py", 1, 2, "python", chunk_id=7)
    stale_payload = Chunk("def authenticate(token):\n    return False", "old_auth.py", 50, 51, "python", chunk_id=7)
    chunks = [canonical]

    class FakeSparseIndex:
        def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
            return [SearchResult(stale_payload, 100.0)]

    with patch("semble.search._search_semantic", return_value=[SearchResult(canonical, 1.0)]):
        results = search(
            "authenticate token",
            mock_model,
            MagicMock(),
            FakeSparseIndex(),
            chunks,
            top_k=2,
            alpha=0.5,
            rerank=False,
        )

    assert [result.chunk for result in results] == [canonical]
    assert results[0].score > 0.01


def test_hybrid_merge_keeps_position_fallback_distinct_from_chunk_id(mock_model: Any) -> None:
    """Legacy position fallback should not collide with a stable chunk ID of the same integer value."""
    legacy = Chunk("def old_chunk():\n    pass", "legacy.py", 1, 2, "python")
    stable = Chunk("def stable_chunk():\n    pass", "stable.py", 10, 11, "python", chunk_id=0)
    chunks = [legacy, stable]

    class FakeSparseIndex:
        def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
            return [SearchResult(stable, 100.0)]

    with patch("semble.search._search_semantic", return_value=[SearchResult(legacy, 1.0)]):
        results = search(
            "chunk",
            mock_model,
            MagicMock(),
            FakeSparseIndex(),
            chunks,
            top_k=2,
            alpha=0.5,
            rerank=False,
        )

    assert {result.chunk for result in results} == {legacy, stable}


def test_stable_id_semantic_backend_returns_empty_for_unmatched_selector(mock_model: Any) -> None:
    """Stable-id dense filters should return no results when selected IDs have no dense rows."""
    chunk = Chunk("def current():\n    pass", "current.py", 10, 11, "python", chunk_id=7)
    backend = SelectableBasicBackend(np.ones((1, 256), dtype=np.float32), BasicArgs())
    semantic = StableIdSemanticBackend(backend, [7])

    class LazyChunks:
        def __len__(self) -> int:
            return 1

        def chunk_by_id(self, chunk_id: int) -> Chunk:
            assert chunk_id == 7
            return chunk

    results = _search_semantic(
        "current",
        mock_model,
        semantic,
        LazyChunks(),
        top_k=1,
        filter_spec=FilterSpec(chunk_ids=frozenset({999})),
    )

    assert results == []


def test_lazy_no_rerank_tie_order_uses_chunk_order_not_chunk_id(mock_model: Any) -> None:
    """Loaded-index no-rerank ties should match eager start-line ordering, not stable chunk_id order."""
    late = Chunk("def late_hit():\n    pass", "late.py", 100, 101, "python", chunk_id=10)
    early = Chunk("def early_hit():\n    pass", "early.py", 1, 2, "python", chunk_id=20)

    class LazyChunks:
        def __len__(self) -> int:
            return 2

        def chunk_by_id(self, chunk_id: int) -> Chunk:
            return {10: late, 20: early}[chunk_id]

        def chunks_by_id(self, chunk_ids: list[int]) -> list[Chunk]:
            return [self.chunk_by_id(chunk_id) for chunk_id in chunk_ids]

    class FakeSemanticIndex:
        def query(self, vectors: Any, k: int, selector: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.array([10]), np.array([0.0], dtype=np.float32))]

    class FakeSparseIndex:
        def search_ids(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[tuple[int, float]]:
            return [(20, 1.0)]

    results = search(
        "hit",
        mock_model,
        FakeSemanticIndex(),
        FakeSparseIndex(),
        LazyChunks(),
        top_k=2,
        alpha=0.5,
        rerank=False,
    )

    assert [result.chunk for result in results] == [early, late]


def test_lazy_rerank_boosts_candidates_without_iterating_all_chunks(mock_model: Any) -> None:
    """Hot rerank should not scan every persisted chunk to find non-candidate symbol matches."""
    candidate = Chunk("class UserSessionManager:\n    pass", "sessions.py", 1, 2, "python", chunk_id=7)

    class LazyChunks:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            raise AssertionError("lazy chunks should not be iterated during rerank")

        def __getitem__(self, index: int) -> Chunk:
            raise AssertionError("lazy chunks should not be indexed during rerank")

        def chunk_by_id(self, chunk_id: int) -> Chunk:
            assert chunk_id == 7
            return candidate

    class FakeSemanticIndex:
        def query(self, vectors: Any, k: int, selector: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.array([7]), np.array([0.0]))]

    class FakeSparseIndex:
        def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
            return [SearchResult(candidate, 100.0)]

    results = search(
        "find UserSessionManager",
        mock_model,
        FakeSemanticIndex(),
        FakeSparseIndex(),
        LazyChunks(),
        top_k=1,
        alpha=0.5,
        rerank=True,
    )

    assert [result.chunk for result in results] == [candidate]


def test_lazy_rerank_boosts_symbol_non_candidates_from_stem_lookup(mock_model: Any) -> None:
    """Lazy hot search should recover symbol-definition non-candidates without scanning every chunk."""
    candidate = Chunk("type OtherStatus int", "other_status.go", 1, 1, "go", chunk_id=7)
    non_candidate = Chunk("type PaymentStatus int", "payment_status.go", 1, 1, "go", chunk_id=99)

    class LazyChunks:
        def __len__(self) -> int:
            return 100

        def __iter__(self):
            raise AssertionError("lazy chunks should not be iterated during rerank")

        def __getitem__(self, index: int) -> Chunk:
            raise AssertionError("lazy chunks should not be indexed during rerank")

        def chunk_by_id(self, chunk_id: int) -> Chunk:
            return {7: candidate, 99: non_candidate}[chunk_id]

        def chunks_by_id(self, chunk_ids: list[int]) -> list[Chunk]:
            return [self.chunk_by_id(chunk_id) for chunk_id in chunk_ids]

        def chunk_ids_for_symbol_stem(self, symbol_name: str) -> list[int]:
            assert symbol_name == "PaymentStatus"
            return [99]

        def chunk_ids_for_embedded_symbol_stem(self, symbol_name: str, min_prefix_len: int) -> list[int]:
            return []

    class FakeSemanticIndex:
        def query(self, vectors: Any, k: int, selector: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.array([7]), np.array([0.0]))]

    class FakeSparseIndex:
        def search_ids(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[tuple[int, float]]:
            return [(7, 1.0)]

    results = search(
        "PaymentStatus",
        mock_model,
        FakeSemanticIndex(),
        FakeSparseIndex(),
        LazyChunks(),
        top_k=1,
        alpha=0.3,
        rerank=True,
    )

    assert results[0].chunk is non_candidate


def test_lazy_rerank_does_not_boost_symbol_non_candidates_outside_filter(mock_model: Any) -> None:
    """Filtered lazy hot search should not reintroduce stem matches outside the filter."""
    candidate = Chunk("type OtherStatus int", "other_status.go", 1, 1, "go", chunk_id=7)
    excluded = Chunk("type PaymentStatus int", "payment_status.go", 1, 1, "go", chunk_id=99)

    class LazyChunks:
        def __len__(self) -> int:
            return 100

        def __iter__(self):
            raise AssertionError("lazy chunks should not be iterated during rerank")

        def __getitem__(self, index: int) -> Chunk:
            raise AssertionError("lazy chunks should not be indexed during rerank")

        def chunk_by_id(self, chunk_id: int) -> Chunk:
            return {7: candidate, 99: excluded}[chunk_id]

        def chunks_by_id(self, chunk_ids: list[int]) -> list[Chunk]:
            return [self.chunk_by_id(chunk_id) for chunk_id in chunk_ids]

        def chunk_ids_for_symbol_stem(self, symbol_name: str) -> list[int]:
            raise AssertionError("filtered rerank should not consult non-candidate stem lookup")

        def chunk_ids_for_embedded_symbol_stem(self, symbol_name: str, min_prefix_len: int) -> list[int]:
            raise AssertionError("filtered rerank should not consult non-candidate stem lookup")

    class FakeSemanticIndex:
        def query(self, vectors: Any, k: int, selector: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
            assert selector.tolist() == [7]
            return [(np.array([7]), np.array([0.0]))]

    class FakeSparseIndex:
        def search_ids(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[tuple[int, float]]:
            assert filter_spec == FilterSpec(chunk_ids=frozenset({7}))
            return [(7, 1.0)]

    results = search(
        "PaymentStatus",
        mock_model,
        FakeSemanticIndex(),
        FakeSparseIndex(),
        LazyChunks(),
        top_k=1,
        alpha=0.3,
        filter_spec=FilterSpec(chunk_ids=frozenset({7})),
        rerank=True,
    )

    assert results[0].chunk is candidate


def test_lazy_rerank_orders_embedded_symbol_stem_matches_by_persisted_order(mock_model: Any) -> None:
    """Lazy embedded-symbol non-candidates should keep persisted order across multiple symbol names."""
    candidate = Chunk("def unrelated(): pass", "candidate.py", 1, 1, "python", chunk_id=7)
    first = Chunk("class BetaManager:\n    pass", "beta.py", 1, 2, "python", chunk_id=10)
    second = Chunk("class AlphaManager:\n    pass", "alpha.py", 1, 2, "python", chunk_id=20)
    chunks_by_id = {7: candidate, 10: first, 20: second}
    order = {7: 0, 10: 1, 20: 2}

    lazy_chunks = MagicMock()
    lazy_chunks.__len__.return_value = 100
    lazy_chunks.__iter__.side_effect = AssertionError("lazy chunks should not be iterated during rerank")
    lazy_chunks.__getitem__.side_effect = AssertionError("lazy chunks should not be indexed during rerank")
    lazy_chunks.chunk_by_id.side_effect = chunks_by_id.__getitem__
    lazy_chunks.chunks_by_id.side_effect = lambda chunk_ids: [chunks_by_id[chunk_id] for chunk_id in chunk_ids]
    lazy_chunks.chunk_ids_for_symbol_stem.return_value = []
    lazy_chunks.chunk_ids_for_embedded_symbol_stem.side_effect = lambda symbol_name, min_prefix_len: {
        "AlphaManager": [20],
        "BetaManager": [10],
    }.get(symbol_name, [])
    lazy_chunks.order_chunk_ids.side_effect = lambda chunk_ids: sorted(set(chunk_ids), key=order.__getitem__)

    semantic_index = MagicMock()
    semantic_index.query.return_value = [(np.array([7]), np.array([0.0]))]
    sparse_index = MagicMock()
    sparse_index.search_ids.return_value = [(7, 1.0)]

    results = search(
        "find AlphaManager and BetaManager",
        mock_model,
        semantic_index,
        sparse_index,
        lazy_chunks,
        top_k=2,
        alpha=0.3,
        rerank=True,
    )

    assert [result.chunk for result in results] == [first, second]


@pytest.mark.parametrize(
    ("search_fn", "query", "top_k"),
    [
        (lambda q, m, s, b, c, k: _search_bm25(q, b, c, k, filter_spec=None), "authenticate", 3),
        (lambda q, m, s, b, c, k: _search_semantic(q, m, s, c, k, filter_spec=None), "query", 4),
        (lambda q, m, s, b, c, k: search(q, m, s, b, c, k), "login", 4),
    ],
)
def test_search_source_labels(
    search_fn: Any,
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
    fake_model = MagicMock(spec=StaticModel)
    with patch("semble.index.dense.StaticModel.from_pretrained", return_value=fake_model) as mock_fp:
        result, _ = load_model(model_path)
    mock_fp.assert_called_once_with(expected_call_arg, force_download=False)
    assert result is fake_model


def test_embed_chunks_empty_returns_empty_array(mock_model: Any) -> None:
    """embed_chunks with an empty list returns a (0, 256) float32 array."""
    result = embed_chunks(mock_model, [])
    assert result.shape == (0, 256)
    assert result.dtype == np.float32


def test_embed_chunks_uses_model_multiprocessing(mock_model: Any) -> None:
    """Embedding keeps model2vec's built-in multiprocessing path available."""
    embed_chunks(mock_model, [make_chunk("x = 1")])
    mock_model.encode.assert_called_once_with(["x = 1"], use_multiprocessing=True)


def test_embed_chunks_can_disable_model_multiprocessing(mock_model: Any) -> None:
    """Streaming micro-batches can avoid repeated model2vec worker startup."""
    embed_chunks(mock_model, [make_chunk("x = 1")], use_multiprocessing=False)
    mock_model.encode.assert_called_once_with(["x = 1"], use_multiprocessing=False)


def test_selectable_basic_backend_rejects_k_below_one(
    semantic: SelectableBasicBackend, embeddings: npt.NDArray[np.float32]
) -> None:
    """SelectableBasicBackend.query guards against k < 1."""
    with pytest.raises(ValueError, match="k should be >= 1"):
        semantic.query(embeddings[:1], k=0)
