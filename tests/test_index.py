"""Tests for semble._index (SembleIndex)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semble import SembleIndex


@pytest.fixture
def index(mock_model):
    idx = SembleIndex()
    idx._model = mock_model
    return idx


def test_index_directory_returns_stats(index, tmp_project) -> None:
    stats = index.index_directory(tmp_project)
    assert stats.total_files >= 2  # auth.py, utils.py
    assert stats.total_chunks > 0
    assert stats.index_time_ms > 0
    assert stats.embedding_time_ms > 0


def test_index_directory_excludes_markdown_by_default(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    # README.md should be excluded (not in CODE_EXTENSIONS)
    assert ".md" not in [Path(c.file_path).suffix for c in index._chunks]


def test_index_directory_includes_markdown_with_flag(index, tmp_project) -> None:
    index.index_directory(tmp_project, include_docs=True)
    suffixes = {Path(c.file_path).suffix for c in index._chunks}
    assert ".md" in suffixes


def test_index_directory_empty_returns_zero_chunks(index, tmp_path) -> None:
    stats = index.index_directory(tmp_path)
    assert stats.total_chunks == 0
    assert stats.total_files == 0


def test_index_language_counts(index, tmp_project) -> None:
    stats = index.index_directory(tmp_project)
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


def test_search_returns_empty_before_indexing() -> None:
    idx = SembleIndex()
    results = idx.search("anything")
    assert results == []


def test_search_hybrid(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate token", top_k=3, mode="hybrid")
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_bm25(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=3, mode="bm25")
    assert isinstance(results, list)


def test_search_semantic(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("authentication", top_k=3, mode="semantic")
    assert isinstance(results, list)


def test_search_symbol(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=5, mode="symbol")
    # Should find the authenticate function
    symbol_names = [r.chunk.symbol_name for r in results]
    assert "authenticate" in symbol_names


def test_search_invalid_mode(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    with pytest.raises(ValueError, match="Unknown search mode"):
        index.search("query", mode="invalid")


def test_search_top_k_respected(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("function", top_k=1, mode="bm25")
    assert len(results) <= 1


def test_search_dedup_default(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=5, dedup=True)
    # All content hashes should be unique after dedup
    hashes = [r.chunk.content_hash for r in results]
    assert len(hashes) == len(set(hashes))


def test_embedding_cache_populated(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    assert len(index._embedding_cache) > 0


def test_embedding_cache_reused_on_reindex(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    call_count_after_first = index._model.encode.call_count

    index.index_directory(tmp_project)
    call_count_after_second = index._model.encode.call_count

    # Second index should not call encode for already-cached content
    assert call_count_after_second == call_count_after_first


def test_get_context_returns_related(index, tmp_project) -> None:
    index.index_directory(tmp_project)
    if not index._chunks:
        pytest.skip("No chunks indexed")
    first = index._chunks[0]
    results = index.get_context(first.file_path, first.start_line, top_k=3)
    # Should not return the target chunk itself
    assert all(r.chunk.content_hash != first.content_hash for r in results)


def test_get_context_empty_before_index() -> None:
    idx = SembleIndex()
    results = idx.get_context("any/file.py", 1)
    assert results == []


def test_stats_property(index, tmp_project) -> None:
    stats = index.index_directory(tmp_project)
    assert index.stats is stats
