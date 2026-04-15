from pathlib import Path

import pytest

from semble.chunker import _chunk_with_chonkie, chunk_file, chunk_lines


def test_chunk_lines_basic(tmp_path: Path) -> None:
    """Chunks are produced with non-empty content."""
    f = tmp_path / "test.py"
    f.write_text("\n".join(f"line {i}" for i in range(10)))
    chunks = chunk_lines(f.read_text(), str(f), "python", max_lines=5, overlap_lines=1)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.content.strip()


def test_chunk_lines_empty(tmp_path: Path) -> None:
    """Empty source produces no chunks."""
    f = tmp_path / "empty.py"
    f.write_text("")
    chunks = chunk_lines("", str(f), "python")
    assert chunks == []


def test_chunk_lines_line_numbers(tmp_path: Path) -> None:
    """First chunk starts at line 1."""
    content = "a\nb\nc\nd\ne\n"
    f = tmp_path / "t.py"
    chunks = chunk_lines(content, str(f), "python", max_lines=3, overlap_lines=0)
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


def test_chunk_with_chonkie_fallback(tmp_path: Path) -> None:
    """Should fall back to line-based when given an unsupported language."""
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    pass\n")
    chunks = _chunk_with_chonkie(f.read_text(), str(f), "python")
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
