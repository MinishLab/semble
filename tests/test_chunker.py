from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.index.chunker import chunk_file, chunk_lines, chunk_source
from semble.index.file_walker import filter_extensions


def test_chunk_lines() -> None:
    """chunk_lines: empty input → []; real input → non-empty chunks starting at line 1."""
    assert chunk_lines("", "empty.py", "python") == []

    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_lines(content, "test.py", "python", max_lines=5, overlap_lines=1)
    assert len(chunks) >= 2
    assert all(c.content.strip() for c in chunks)
    assert chunks[0].start_line == 1


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        (None, None),  # nonexistent path
        ("empty.py", "   \n\n  "),  # whitespace-only
        ("file.xyz", "hello world\n" * 5),  # unknown extension
    ],
    ids=["nonexistent", "whitespace_only", "unknown_extension"],
)
def test_chunk_file_edge_cases_return_list(tmp_path: Path, filename: str | None, content: str | None) -> None:
    """chunk_file returns a list (usually empty) for missing / empty / unknown-type files without raising."""
    if filename is None:
        target = Path("/nonexistent/file.py")
    else:
        target = tmp_path / filename
        assert content is not None
        target.write_text(content)
    chunks = chunk_file(target)
    assert isinstance(chunks, list)


def test_chunk_file_py_produces_sorted_chunks(tmp_py_file: Path) -> None:
    """Python file with functions produces at least one chunk in ascending start-line order."""
    pytest.importorskip("tree_sitter_python")
    chunks = chunk_file(tmp_py_file)
    assert len(chunks) >= 1
    start_lines = [c.start_line for c in chunks]
    assert start_lines == sorted(start_lines)


def _whitespace_chunker() -> MagicMock:
    whitespace_chunk = MagicMock(text="   \n", start_index=0, end_index=0)
    chunker = MagicMock()
    chunker.chunk.return_value = [whitespace_chunk]
    return chunker


@pytest.mark.parametrize(
    "codechunker_patch",
    [
        {"side_effect": Exception("boom")},  # raises
        {"return_value": MagicMock(chunk=MagicMock(return_value=[]))},  # empty result
        {"return_value": _whitespace_chunker()},  # whitespace-only chunks
    ],
    ids=["raises", "empty", "whitespace_only"],
)
def test_chunk_source_falls_back_when_chonkie_unusable(codechunker_patch: dict) -> None:
    """chunk_source falls back to line-based chunking when chonkie fails or yields nothing usable."""
    source = "def foo():\n    pass\n"
    with patch("semble.index.chunker.CodeChunker", **codechunker_patch):
        chunks = chunk_source(source, "foo.py", "python")
    assert len(chunks) > 0
    assert all(c.content.strip() for c in chunks)


def test_chunk_source_empty_string() -> None:
    """chunk_source returns [] for whitespace-only input."""
    assert chunk_source("   \n\n", "foo.py", "python") == []


def test_filter_extensions_explicit() -> None:
    """filter_extensions returns the provided set unchanged when extensions is not None."""
    explicit: frozenset[str] = frozenset({".py", ".ts"})
    result = filter_extensions(explicit, include_text_files=False)
    assert result == explicit
