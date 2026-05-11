from pathlib import Path
from typing import Any

import pytest

from semble import SembleIndex
from semble.index.create import _MAX_FILE_BYTES, create_index_from_path
from semble.types import Encoder
from tests.conftest import make_chunk


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project."""
    return SembleIndex.from_path(tmp_project, model=mock_model)


@pytest.mark.parametrize(
    ("include_text_files", "md_in_results"),
    [(False, False), (True, True)],
)
def test_index_markdown_inclusion(
    mock_model: Encoder, tmp_project: Path, include_text_files: bool, md_in_results: bool
) -> None:
    """Markdown files are excluded by default and included when include_text_files=True."""
    _, _, chunks = create_index_from_path(tmp_project, mock_model, include_text_files=include_text_files)
    has_md = ".md" in {Path(c.file_path).suffix for c in chunks}
    assert has_md is md_in_results


def test_index_empty_returns_zero_chunks(mock_model: Encoder, tmp_path: Path) -> None:
    """Indexing an empty directory yields zero files and chunks."""
    with pytest.raises(ValueError):
        create_index_from_path(tmp_path, mock_model)


def test_oversized_file_is_skipped(mock_model: Encoder, tmp_path: Path) -> None:
    """Files exceeding _MAX_FILE_BYTES are silently skipped during indexing."""
    (tmp_path / "big.py").write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError):  # no indexable content remains
        create_index_from_path(tmp_path, mock_model)


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    """Language breakdown in stats includes python with at least one chunk."""
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


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


def test_search_constraints(indexed_index: SembleIndex) -> None:
    """search: top_k is respected; no duplicate chunks are returned."""
    assert len(indexed_index.search("function", top_k=1, mode="bm25")) <= 1

    results = indexed_index.search("authenticate", top_k=5)
    assert len(results) == len(set(r.chunk for r in results))


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
def test_search_with_filter_paths_does_not_crash(indexed_index: SembleIndex, mode: str) -> None:
    """Filtered search works regardless of where the selected chunk lives in the corpus."""
    target_path = indexed_index.chunks[-1].file_path
    results = indexed_index.search("function", top_k=3, mode=mode, filter_paths=[target_path])
    assert all(r.chunk.file_path == target_path for r in results)


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_search_empty_query_returns_empty(indexed_index: SembleIndex, mode: str, query: str) -> None:
    """Empty / whitespace-only queries return [] across all modes."""
    assert indexed_index.search(query, mode=mode) == []


@pytest.mark.parametrize(
    ("disk_files", "chunk_paths", "expected"),
    [
        ({"foo.py": "hello world"}, ["foo.py", "foo.py"], {"foo.py": 11}),
        ({}, ["nonexistent.py"], {}),
    ],
    ids=["dedup-same-file", "missing-file-skipped"],
)
def test_compute_file_sizes(
    tmp_path: Path, disk_files: dict[str, str], chunk_paths: list[str], expected: dict[str, int]
) -> None:
    """_compute_file_sizes deduplicates paths and silently skips missing files."""
    for name, content in disk_files.items():
        (tmp_path / name).write_text(content)
    index = SembleIndex.__new__(SembleIndex)
    index.chunks = [make_chunk("c", p) for p in chunk_paths]
    assert index._compute_file_sizes(tmp_path) == expected


def test_find_related(indexed_index: SembleIndex) -> None:
    """find_related returns related chunks for a Chunk or SearchResult seed."""
    chunk = indexed_index.chunks[0]
    via_chunk = indexed_index.find_related(chunk, top_k=3)
    assert isinstance(via_chunk, list)
    assert len(via_chunk) <= 3
    assert all(r.chunk != chunk for r in via_chunk)

    # SearchResult form returns the same results as Chunk form.
    result = indexed_index.search("authenticate", top_k=1)[0]
    assert [r.chunk for r in indexed_index.find_related(result, top_k=3)] == [
        r.chunk for r in indexed_index.find_related(result.chunk, top_k=3)
    ]
