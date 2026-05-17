from unittest.mock import patch

from semble.chunking.chunking import Chunk, chunk_lines, chunk_source
from semble.chunking.core import ChunkBoundary, chunk


def test_chunk_lines() -> None:
    """chunk_lines: empty input → []; real input → non-empty chunks starting at line 1."""
    assert chunk_lines("", 23) == []

    content = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_lines(content, 10)
    assert len(chunks) >= 2
    assert chunks[0].start == 0


def test_chunk_source_empty_string() -> None:
    """chunk_source returns [] for whitespace-only input."""
    assert chunk_source("   \n\n", "foo.py", "python") == []


def test_chunk_source_language() -> None:
    """Check that chunking defaults to line splitting with non-existent and None languages."""
    with patch("semble.chunking.chunking.chunk_lines", wraps=chunk_lines) as chunk_line_spy:
        assert chunk_source("hello", "foo.loki", "loki") == [
            Chunk(content="hello", file_path="foo.loki", start_line=1, end_line=1, language="loki")
        ]
        chunk_line_spy.assert_called_once()
    with patch("semble.chunking.chunking.chunk_lines", wraps=chunk_lines) as chunk_line_spy:
        assert chunk_source("1+1=3", "foo.json", None) == [
            Chunk(content="1+1=3", file_path="foo.json", start_line=1, end_line=1, language=None)
        ]
        chunk_line_spy.assert_called_once()


def test_core_chunk_empty_input() -> None:
    """core.chunk returns [] for whitespace-only input."""
    assert chunk("   \n", "python", 100) == []


def test_core_chunk_lines_merges_small_lines() -> None:
    """core.chunk_lines merges adjacent small lines that fit within desired_length."""
    # Each line is 2 chars ("a\n"), desired_length=10 allows merging up to 5 lines
    content = "a\nb\nc\nd\ne\nf\n"
    chunks = chunk_lines(content, 10)
    # 6 lines x 2 chars = 12 total; first 5 merge (10 chars), last 1 alone (2 chars)
    assert len(chunks) == 2
    assert chunks[0] == ChunkBoundary(start=0, end=10)
    assert chunks[1] == ChunkBoundary(start=10, end=12)


def test_core_chunk_recursive_split_and_break() -> None:
    """core.chunk recursively splits large nodes and breaks when siblings exceed the limit."""
    code = "x = 1\ndef foo():\n    a = 1\n    b = 2\n    c = 3\ny = 2\n"
    chunks = chunk(code, "python", 10)
    assert len(chunks) >= 3
    assert chunks[0].start == 0
    # Non-overlapping and within bounds
    for i, c in enumerate(chunks):
        assert 0 <= c.start < c.end <= len(code)
        if i > 0:
            assert c.start >= chunks[i - 1].end


def test_core_chunk_leaf_node_exceeds_desired_length() -> None:
    """core.chunk handles leaf tokens (e.g. a very long identifier) that exceed desired_length."""
    from semble.chunking.core import chunk

    long_var = "x" * 100
    code = f"{long_var} = 1\n"
    chunks = chunk(code, "python", 50)
    assert len(chunks) >= 1
    assert chunks[0].start == 0
    for c in chunks:
        assert 0 <= c.start < c.end <= len(code)
