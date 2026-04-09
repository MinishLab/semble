from pathlib import Path
from typing import Any

import pytest

from semble import SembleIndex


@pytest.fixture
def index(mock_model: Any) -> SembleIndex:
    """SembleIndex backed by the deterministic mock encoder."""
    return SembleIndex(model=mock_model)


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project via the primary constructor."""
    return SembleIndex.from_path(tmp_project, model=mock_model)


def test_index_returns_stats(index: SembleIndex, tmp_project: Path) -> None:
    """Indexing returns stats with file and chunk counts populated."""
    stats = index.index(tmp_project)
    assert stats.indexed_files >= 2  # auth.py, utils.py
    assert stats.total_chunks > 0


def test_index_excludes_markdown_by_default(indexed_index: SembleIndex) -> None:
    """Markdown files are excluded unless include_docs=True."""
    assert ".md" not in [Path(chunk.file_path).suffix for chunk in indexed_index.chunks]


def test_index_includes_markdown_with_flag(index: SembleIndex, tmp_project: Path) -> None:
    """include_docs=True causes markdown files to be indexed."""
    index.index(tmp_project, include_docs=True)
    suffixes = {Path(c.file_path).suffix for c in index.chunks}
    assert ".md" in suffixes


def test_index_empty_returns_zero_chunks(index: SembleIndex, tmp_path: Path) -> None:
    """Indexing an empty directory yields zero files and chunks."""
    stats = index.index(tmp_path)
    assert stats.total_chunks == 0
    assert stats.indexed_files == 0


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    """Language breakdown in stats includes python with at least one chunk."""
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


def test_search_returns_empty_before_indexing() -> None:
    """Search on an empty index returns an empty list."""
    idx = SembleIndex()
    results = idx.search("anything")
    assert results == []


@pytest.mark.parametrize(
    "query, mode",
    [("authenticate token", "hybrid"), ("authenticate", "bm25"), ("authentication", "semantic")],
)
def test_search_modes(indexed_index: SembleIndex, query: str, mode: str) -> None:
    """Each search mode returns a valid list of at most top_k results."""
    results = indexed_index.search(query, top_k=3, mode=mode)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_invalid_mode(indexed_index: SembleIndex) -> None:
    """An unrecognised mode string raises ValueError."""
    with pytest.raises(ValueError):
        indexed_index.search("query", mode="invalid")


def test_search_top_k_respected(indexed_index: SembleIndex) -> None:
    """Results never exceed the requested top_k."""
    results = indexed_index.search("function", top_k=1, mode="bm25")
    assert len(results) <= 1


def test_search_no_duplicate_chunks(indexed_index: SembleIndex) -> None:
    """Each result chunk appears at most once in the result list."""
    results = indexed_index.search("authenticate", top_k=5)
    assert len(results) == len(set(r.chunk for r in results))


def test_reindex_does_not_re_embed(indexed_index: SembleIndex, tmp_project: Path, mock_model: Any) -> None:
    """Re-indexing unchanged files does not call encode again."""
    call_count_after_first = mock_model.encode.call_count

    indexed_index.index(tmp_project)

    assert mock_model.encode.call_count == call_count_after_first


def test_stats_property(indexed_index: SembleIndex) -> None:
    """Stats property reflects the most recent index call."""
    assert indexed_index.stats.indexed_files >= 2


# ---------------------------------------------------------------------------
# Disk embedding cache
# ---------------------------------------------------------------------------

_MODEL_ID = "test/model"


def test_disk_cache_required_model_id(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """Providing cache_dir without model_id raises ValueError."""
    idx = SembleIndex(model=mock_model)
    with pytest.raises(ValueError, match="model_id"):
        idx.index(tmp_project, cache_dir=tmp_path / "cache")


def test_disk_cache_writes_embeddings(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """After indexing with a cache_dir, .npy files are written to disk."""
    cache_dir = tmp_path / "cache"
    idx = SembleIndex(model=mock_model)
    idx.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)

    npy_files = list(cache_dir.rglob("*.npy"))
    assert len(npy_files) == len(idx.chunks)


def test_disk_cache_avoids_re_encode(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """A second index run with the same cache_dir never calls encode again."""
    cache_dir = tmp_path / "cache"

    first = SembleIndex(model=mock_model)
    first.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)
    encode_calls_after_first = mock_model.encode.call_count

    # Fresh instance — no in-memory cache — but disk cache is warm.
    second = SembleIndex(model=mock_model)
    second.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)

    assert mock_model.encode.call_count == encode_calls_after_first


def test_disk_cache_encodes_only_new_chunks(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """Adding a file triggers encode only for the new chunks."""
    cache_dir = tmp_path / "cache"

    first = SembleIndex(model=mock_model)
    first.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)
    calls_after_first = mock_model.encode.call_count

    (tmp_project / "new_module.py").write_text("def new_func(): pass\n")

    second = SembleIndex(model=mock_model)
    second.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)

    # encode was called exactly once more (for the new file's chunks).
    assert mock_model.encode.call_count == calls_after_first + 1


def test_disk_cache_model_namespace_isolates_models(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """Different model_ids write to separate subdirectories."""
    cache_dir = tmp_path / "cache"

    idx_a = SembleIndex(model=mock_model)
    idx_a.index(tmp_project, cache_dir=cache_dir, model_id="org/model-a")

    idx_b = SembleIndex(model=mock_model)
    idx_b.index(tmp_project, cache_dir=cache_dir, model_id="org/model-b")

    namespaces = {p.name for p in cache_dir.iterdir() if p.is_dir()}
    assert "org--model-a" in namespaces
    assert "org--model-b" in namespaces


def test_from_path_threads_cache_args(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """from_path forwards cache_dir and model_id to index()."""
    cache_dir = tmp_path / "cache"
    idx = SembleIndex.from_path(tmp_project, model=mock_model, cache_dir=cache_dir, model_id=_MODEL_ID)

    assert list(cache_dir.rglob("*.npy"))  # files were written
    assert len(idx.chunks) > 0


def test_disk_cache_tilde_expansion(mock_model: Any, tmp_project: Path, monkeypatch: Any) -> None:
    """cache_dir values containing ~ are expanded to the real home directory."""
    fake_home = tmp_project / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    idx = SembleIndex(model=mock_model)
    idx.index(tmp_project, cache_dir="~/semble-cache", model_id=_MODEL_ID)

    expected = fake_home / "semble-cache"
    assert list(expected.rglob("*.npy"))


def test_semantic_search_raises_for_non_default_model_id_without_model(
    tmp_project: Path, tmp_path: Path, mock_model: Any
) -> None:
    """semantic/hybrid search raises ValueError when model_id is not the default and no model is set.

    This prevents silent dimensionality mismatches between cached embeddings
    and the lazily-loaded default model.
    """
    cache_dir = tmp_path / "cache"

    # Warm the cache with mock_model (non-default model_id).
    first = SembleIndex(model=mock_model)
    first.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)

    # Re-index from warm cache — no model supplied, non-default model_id.
    second = SembleIndex()
    second.index(tmp_project, cache_dir=cache_dir, model_id=_MODEL_ID)

    # BM25 is fine — doesn't touch the model.
    assert second.search("authenticate", mode="bm25") != []

    # semantic and hybrid must raise rather than load an incompatible default.
    with pytest.raises(ValueError, match=_MODEL_ID):
        second.search("authenticate", mode="semantic")

    with pytest.raises(ValueError, match=_MODEL_ID):
        second.search("authenticate", mode="hybrid")
