"""Tests for semble.index (SembleIndex)."""

from pathlib import Path
from typing import Any

import pytest

from semble import SearchMode, SembleIndex


@pytest.fixture
def index(mock_model: Any) -> SembleIndex:
    return SembleIndex(model=mock_model)


@pytest.fixture
def indexed_index(index: SembleIndex, tmp_project: Path) -> SembleIndex:
    index.index(tmp_project)
    return index


def test_index_returns_stats(index: SembleIndex, tmp_project: Path) -> None:
    stats = index.index(tmp_project)
    assert stats.total_files >= 2  # auth.py, utils.py
    assert stats.total_chunks > 0
    assert stats.embedding_time_ms > 0


def test_index_excludes_markdown_by_default(indexed_index: SembleIndex) -> None:
    assert ".md" not in [Path(chunk.file_path).suffix for chunk in indexed_index._chunks]


def test_index_includes_markdown_with_flag(index: SembleIndex, tmp_project: Path) -> None:
    index.index(tmp_project, include_docs=True)
    suffixes = {Path(c.file_path).suffix for c in index._chunks}
    assert ".md" in suffixes


def test_index_empty_returns_zero_chunks(index: SembleIndex, tmp_path: Path) -> None:
    stats = index.index(tmp_path)
    assert stats.total_chunks == 0
    assert stats.total_files == 0


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


def test_search_returns_empty_before_indexing() -> None:
    idx = SembleIndex()
    results = idx.search("anything")
    assert results == []


@pytest.mark.parametrize(
    "query, mode",
    [("authenticate token", "hybrid"), ("authenticate", "bm25"), ("authentication", "semantic")],
)
def test_search_modes(indexed_index: SembleIndex, query: str, mode: str) -> None:
    results = indexed_index.search(query, top_k=3, mode=mode)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_symbol(indexed_index: SembleIndex) -> None:
    results = indexed_index.search("authenticate", top_k=5, mode=SearchMode.SYMBOL)
    assert len(results) > 0
    assert any("authenticate" in r.chunk.content for r in results)


def test_search_invalid_mode(indexed_index: SembleIndex) -> None:
    with pytest.raises(ValueError, match="Unknown search mode"):
        indexed_index.search("query", mode="invalid")


def test_search_top_k_respected(indexed_index: SembleIndex) -> None:
    results = indexed_index.search("function", top_k=1, mode="bm25")
    assert len(results) <= 1


def test_search_no_duplicate_hashes(indexed_index: SembleIndex) -> None:
    results = indexed_index.search("authenticate", top_k=5)
    hashes = [r.chunk.content_hash for r in results]
    assert len(hashes) == len(set(hashes))


def test_reindex_does_not_re_embed(
    indexed_index: SembleIndex, tmp_project: Path, mock_model: Any
) -> None:
    call_count_after_first = mock_model.encode.call_count

    indexed_index.index(tmp_project)

    assert mock_model.encode.call_count == call_count_after_first


def test_stats_property(indexed_index: SembleIndex) -> None:
    assert indexed_index.stats.total_files >= 2
