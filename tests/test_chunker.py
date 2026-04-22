from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.index.chunker import _chunk_with_chonkie, chunk_file, chunk_lines, chunk_source
from semble.index.file_walker import filter_extensions


def test_chunk_lines_basic() -> None:
    """Chunks are produced with non-empty content."""
    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_lines(content, "test.py", "python", max_lines=5, overlap_lines=1)
    assert len(chunks) >= 2
    assert all(c.content.strip() for c in chunks)


def test_chunk_lines_empty() -> None:
    """Empty source produces no chunks."""
    assert chunk_lines("", "empty.py", "python") == []


def test_chunk_lines_line_numbers() -> None:
    """First chunk starts at line 1."""
    chunks = chunk_lines("a\nb\nc\nd\ne\n", "t.py", "python", max_lines=3, overlap_lines=0)
    assert chunks[0].start_line == 1


def test_chunk_file_nonexistent() -> None:
    """Non-existent file returns empty list without raising."""
    chunks = chunk_file(Path("/nonexistent/file.py"))
    assert chunks == []


def test_chunk_file_empty(tmp_path: Path) -> None:
    """Whitespace-only file returns no chunks."""
    f = tmp_path / "empty.py"
    f.write_text("   \n\n  ")
    chunks = chunk_file(f)
    assert chunks == []


def test_chunk_with_chonkie_fallback() -> None:
    """Chonkie produces chunks from Python source."""
    chunks = _chunk_with_chonkie("def foo():\n    pass\n", "code.py", "python")
    assert len(chunks) > 0


def test_chunk_file_py_produces_chunks(tmp_py_file: Path) -> None:
    """Python file with functions is split into at least one chunk."""
    chunks = chunk_file(tmp_py_file)
    assert len(chunks) >= 1


def test_chunk_file_sorted_by_line(tmp_py_file: Path) -> None:
    """Chunks are returned in ascending start-line order."""
    pytest.importorskip("tree_sitter_python")
    chunks = chunk_file(tmp_py_file)
    start_lines = [c.start_line for c in chunks]
    assert start_lines == sorted(start_lines)


def test_chunk_file_unknown_extension(tmp_path: Path) -> None:
    """Unknown file extension returns a list without raising."""
    f = tmp_path / "file.xyz"
    f.write_text("hello world\n" * 5)
    chunks = chunk_file(f)
    assert isinstance(chunks, list)


def test_chunk_with_chonkie_exception_falls_back_to_lines(tmp_path: Path) -> None:
    """When CodeChunker raises, _chunk_with_chonkie falls back to line-based chunking."""
    source = "def foo():\n    pass\n"
    with patch("semble.index.chunker.CodeChunker", side_effect=Exception("boom")):
        chunks = _chunk_with_chonkie(source, "foo.py", "python")
    assert len(chunks) > 0
    assert all(c.content.strip() for c in chunks)


def test_chunk_with_chonkie_empty_result_falls_back_to_lines() -> None:
    """When CodeChunker returns an empty list, fall back to line-based chunking."""
    source = "def foo():\n    pass\n"
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = []
    with patch("semble.index.chunker.CodeChunker", return_value=mock_chunker):
        chunks = _chunk_with_chonkie(source, "foo.py", "python")
    assert len(chunks) > 0


def test_chunk_with_chonkie_whitespace_only_chunk_skipped() -> None:
    """Chonkie chunks with whitespace-only text are skipped; fallback if none remain."""
    source = "def foo():\n    pass\n"
    whitespace_chunk = MagicMock()
    whitespace_chunk.text = "   \n"
    whitespace_chunk.start_index = 0
    whitespace_chunk.end_index = 0
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [whitespace_chunk]
    with patch("semble.index.chunker.CodeChunker", return_value=mock_chunker):
        chunks = _chunk_with_chonkie(source, "foo.py", "python")
    # All whitespace chunks are skipped; fallback produces real line chunks.
    assert all(c.content.strip() for c in chunks)


def test_chunk_source_empty_string() -> None:
    """chunk_source returns [] for whitespace-only input."""
    assert chunk_source("   \n\n", "foo.py", "python") == []


def test_filter_extensions_explicit() -> None:
    """filter_extensions returns the provided set unchanged when extensions is not None."""
    explicit: frozenset[str] = frozenset({".py", ".ts"})
    result = filter_extensions(explicit, include_text_files=False)
    assert result == explicit
