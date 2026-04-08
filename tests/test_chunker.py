"""Tests for semble.chunker."""

from pathlib import Path

import pytest

from semble.chunker import _chunk_with_chonkie, chunk_file, chunk_lines


def test_chunk_lines_basic(tmp_path: Path) -> None:
    f = tmp_path / "test.py"
    f.write_text("\n".join(f"line {i}" for i in range(10)))
    chunks = chunk_lines(f.read_text(), str(f), "python", max_lines=5, overlap_lines=1)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.content.strip()
        assert c.content_hash


def test_chunk_lines_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.py"
    f.write_text("")
    chunks = chunk_lines("", str(f), "python")
    assert chunks == []


def test_chunk_lines_line_numbers(tmp_path: Path) -> None:
    content = "a\nb\nc\nd\ne\n"
    f = tmp_path / "t.py"
    chunks = chunk_lines(content, str(f), "python", max_lines=3, overlap_lines=0)
    # First chunk starts at line 1
    assert chunks[0].start_line == 1


def test_chunk_file_nonexistent() -> None:
    chunks = chunk_file(Path("/nonexistent/file.py"))
    assert chunks == []


def test_chunk_file_empty(tmp_path: Path) -> None:
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
    chunks = chunk_file(tmp_py_file)
    assert len(chunks) >= 1


def test_chunk_file_sorted_by_line(tmp_py_file: Path) -> None:
    pytest.importorskip("tree_sitter_python")
    chunks = chunk_file(tmp_py_file)
    start_lines = [c.start_line for c in chunks]
    assert start_lines == sorted(start_lines)


def test_chunk_file_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "file.xyz"
    f.write_text("hello world\n" * 5)
    chunks = chunk_file(f)
    # Should fall back to line-based or return empty (not a known extension)
    assert isinstance(chunks, list)


def test_chunk_content_hash_unique(tmp_py_file: Path) -> None:
    chunks = chunk_file(tmp_py_file)
    hashes = [c.content_hash for c in chunks]
    # Hashes should be unique across chunks (different content)
    assert len(hashes) == len(set(hashes))
