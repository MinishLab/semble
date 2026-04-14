from pathlib import Path
from typing import Any

import pytest

from semble import SembleIndex

_MODEL_NAME = "test/mock-model"


@pytest.fixture
def index(mock_model: Any) -> SembleIndex:
    """SembleIndex backed by the deterministic mock encoder."""
    return SembleIndex(model=mock_model, enable_caching=False)


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project."""
    return SembleIndex.from_path(tmp_project, model=mock_model, enable_caching=False)


def test_index_returns_stats(index: SembleIndex, tmp_project: Path) -> None:
    """Indexing returns stats with file and chunk counts populated."""
    stats = index.index(tmp_project)
    assert stats.indexed_files >= 2  # auth.py, utils.py
    assert stats.total_chunks > 0
    assert index.stats == stats


@pytest.mark.parametrize(
    ("include_docs", "md_in_results"),
    [(False, False), (True, True)],
)
def test_index_markdown_inclusion(
    index: SembleIndex, tmp_project: Path, include_docs: bool, md_in_results: bool
) -> None:
    """Markdown files are excluded by default and included when include_docs=True."""
    index.index(tmp_project, include_docs=include_docs)
    has_md = ".md" in {Path(c.file_path).suffix for c in index.chunks}
    assert has_md is md_in_results


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


def test_disk_cache_round_trip(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """from_path writes .npy files on first run and avoids re-encoding on second."""
    cache_dir = tmp_path / "cache"

    first = SembleIndex.from_path(tmp_project, model=mock_model, cache_dir=cache_dir, model_name=_MODEL_NAME)
    assert list(cache_dir.rglob("*.npy"))
    encode_calls = mock_model.encode.call_count

    second = SembleIndex.from_path(tmp_project, model=mock_model, cache_dir=cache_dir, model_name=_MODEL_NAME)
    assert mock_model.encode.call_count == encode_calls
    assert len(second.chunks) == len(first.chunks)


def test_disk_cache_encodes_only_new_chunks(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """Adding a file triggers encode only for the new chunks."""
    cache_dir = tmp_path / "cache"

    first = SembleIndex(model=mock_model, cache_dir=cache_dir, model_name=_MODEL_NAME)
    first.index(tmp_project)
    calls_after_first = mock_model.encode.call_count

    (tmp_project / "new_module.py").write_text("def new_func(): pass\n")

    second = SembleIndex(model=mock_model, cache_dir=cache_dir, model_name=_MODEL_NAME)
    second.index(tmp_project)

    assert mock_model.encode.call_count == calls_after_first + 1


def test_disk_cache_model_name_isolates_models(mock_model: Any, tmp_project: Path, tmp_path: Path) -> None:
    """Different model_name values write to separate subdirectories."""
    cache_dir = tmp_path / "cache"

    SembleIndex(model=mock_model, cache_dir=cache_dir, model_name="org/model-a").index(tmp_project)
    SembleIndex(model=mock_model, cache_dir=cache_dir, model_name="org/model-b").index(tmp_project)

    namespaces = {p.name for p in cache_dir.iterdir() if p.is_dir()}
    assert "org--model-a" in namespaces
    assert "org--model-b" in namespaces


def test_disk_cache_tilde_expansion(mock_model: Any, tmp_project: Path, monkeypatch: Any) -> None:
    """cache_dir values containing ~ are expanded to the real home directory."""
    fake_home = tmp_project / "fakehome"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    SembleIndex(model=mock_model, cache_dir="~/semble-cache", model_name=_MODEL_NAME).index(tmp_project)

    assert list((fake_home / "semble-cache").rglob("*.npy"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_caching": False, "model_name": _MODEL_NAME},
        {"enable_caching": True},  # custom model, no model_name → silently disabled
    ],
)
def test_disk_cache_disabled(mock_model: Any, tmp_project: Path, kwargs: dict[str, Any]) -> None:
    """Disk caching is disabled both when explicitly turned off and when model_name is absent."""
    idx = SembleIndex(model=mock_model, **kwargs)
    assert idx.cache_dir is None
    assert idx.cache_namespace is None


def test_find_related_returns_similar_chunks(indexed_index: SembleIndex) -> None:
    """find_related returns semantically similar chunks for a known file location."""
    chunk = indexed_index.chunks[0]
    results = indexed_index.find_related(chunk.file_path, chunk.start_line, top_k=3)
    assert isinstance(results, list)
    assert all(r.chunk != chunk for r in results)
    assert len(results) <= 3


def test_find_related_unknown_file_returns_empty(indexed_index: SembleIndex) -> None:
    """find_related returns an empty list when the file is not in the index."""
    results = indexed_index.find_related("/does/not/exist.py", 1)
    assert results == []


def test_find_related_before_indexing_returns_empty() -> None:
    """find_related on an empty index returns an empty list."""
    idx = SembleIndex()
    assert idx.find_related("/any/file.py", 1) == []
