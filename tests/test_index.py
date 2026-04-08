"""Tests for semble.index (SembleIndex)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from semble import SearchMode, SembleIndex


@pytest.fixture
def index(mock_model: Any) -> SembleIndex:
    return SembleIndex(model=mock_model)


def test_index_directory_returns_stats(index: SembleIndex, tmp_project: Path) -> None:
    stats = index.index_directory(tmp_project)
    assert stats.total_files >= 2  # auth.py, utils.py
    assert stats.total_chunks > 0
    assert stats.embedding_time_ms > 0


def test_index_directory_excludes_markdown_by_default(
    index: SembleIndex, tmp_project: Path
) -> None:
    index.index_directory(tmp_project)
    assert ".md" not in [Path(c.file_path).suffix for c in index._chunks]


def test_index_directory_includes_markdown_with_flag(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project, include_docs=True)
    suffixes = {Path(c.file_path).suffix for c in index._chunks}
    assert ".md" in suffixes


def test_index_directory_empty_returns_zero_chunks(index: SembleIndex, tmp_path: Path) -> None:
    stats = index.index_directory(tmp_path)
    assert stats.total_chunks == 0
    assert stats.total_files == 0


def test_index_language_counts(index: SembleIndex, tmp_project: Path) -> None:
    stats = index.index_directory(tmp_project)
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


def test_search_returns_empty_before_indexing() -> None:
    idx = SembleIndex()
    results = idx.search("anything")
    assert results == []


def test_search_hybrid(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate token", top_k=3, mode="hybrid")
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_bm25(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=3, mode="bm25")
    assert isinstance(results, list)


def test_search_semantic(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("authentication", top_k=3, mode="semantic")
    assert isinstance(results, list)


def test_search_symbol(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=5, mode=SearchMode.SYMBOL)
    assert len(results) > 0
    assert any("authenticate" in r.chunk.content for r in results)


def test_search_invalid_mode(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    with pytest.raises(ValueError, match="Unknown search mode"):
        index.search("query", mode="invalid")


def test_search_top_k_respected(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("function", top_k=1, mode="bm25")
    assert len(results) <= 1


def test_search_no_duplicate_hashes(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    results = index.search("authenticate", top_k=5)
    hashes = [r.chunk.content_hash for r in results]
    assert len(hashes) == len(set(hashes))


def test_reindex_does_not_re_embed(index: SembleIndex, tmp_project: Path, mock_model: Any) -> None:
    index.index_directory(tmp_project)
    call_count_after_first = mock_model.encode.call_count

    index.index_directory(tmp_project)

    assert mock_model.encode.call_count == call_count_after_first


def test_get_context_returns_related(index: SembleIndex, tmp_project: Path) -> None:
    index.index_directory(tmp_project)
    if not index._chunks:
        pytest.skip("No chunks indexed")
    first = index._chunks[0]
    results = index.get_context(first.file_path, first.start_line, top_k=3)
    assert all(r.chunk.content_hash != first.content_hash for r in results)


def test_get_context_empty_before_index() -> None:
    idx = SembleIndex()
    results = idx.get_context("any/file.py", 1)
    assert results == []


def test_stats_property(index: SembleIndex, tmp_project: Path) -> None:
    stats = index.index_directory(tmp_project)
    assert index.stats is stats


def test_from_directory(index: SembleIndex, tmp_project: Path, mock_model: Any) -> None:
    built = SembleIndex.from_directory(tmp_project, model=mock_model)
    assert built.stats.total_files >= 2
    assert built.stats.total_chunks > 0


def test_constructor_accepts_encoder(tmp_project: Path, mock_model: Any) -> None:
    index = SembleIndex(model=mock_model)
    index.index_directory(tmp_project)
    assert index.model is mock_model
