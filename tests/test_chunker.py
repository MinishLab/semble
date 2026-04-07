"""Tests for semble._chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from semble._chunker import chunk_by_lines, chunk_file, chunk_with_treesitter


def test_chunk_by_lines_basic(tmp_path: Path) -> None:
    f = tmp_path / "test.py"
    f.write_text("\n".join(f"line {i}" for i in range(10)))
    chunks = chunk_by_lines(f.read_text(), str(f), "python", max_lines=5, overlap_lines=1)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.content.strip()
        assert c.content_hash


def test_chunk_by_lines_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.py"
    f.write_text("")
    chunks = chunk_by_lines("", str(f), "python")
    assert chunks == []


def test_chunk_by_lines_line_numbers(tmp_path: Path) -> None:
    content = "a\nb\nc\nd\ne\n"
    f = tmp_path / "t.py"
    chunks = chunk_by_lines(content, str(f), "python", max_lines=3, overlap_lines=0)
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


def test_chunk_file_py_ast_symbol_names(tmp_py_file: Path) -> None:
    # When chonkie is unavailable, AST chunker sets symbol names
    pytest.importorskip("tree_sitter_python", reason="need tree_sitter_python for fallback test")
    from unittest.mock import patch

    with patch.dict("sys.modules", {"chonkie": None}):
        chunks = chunk_file(tmp_py_file)
    symbol_names = [c.symbol_name for c in chunks if c.symbol_name]
    assert "add" in symbol_names
    assert "subtract" in symbol_names


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


def test_chunk_with_treesitter_fallback_on_missing_grammar(tmp_path: Path) -> None:
    """Should fall back to line-based when grammar module is unavailable."""
    f = tmp_path / "code.rb"
    f.write_text("def foo\n  puts 'hi'\nend\n")
    # ruby grammar likely not installed — should fall back gracefully
    chunks = chunk_with_treesitter(f.read_text(), str(f), "ruby")
    assert len(chunks) > 0
