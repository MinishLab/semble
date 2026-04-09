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
