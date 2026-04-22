"""Tests for semble.types: Chunk and SearchResult."""

from semble.types import Chunk, SearchMode, SearchResult


def test_chunk_location() -> None:
    """Location property formats file path and line range."""
    chunk = Chunk(content="x = 1", file_path="src/foo.py", start_line=10, end_line=20)
    assert chunk.location == "src/foo.py:10-20"


def test_search_result_str() -> None:
    """__str__ includes location, score, and content."""
    chunk = Chunk(content="def foo(): pass", file_path="src/foo.py", start_line=1, end_line=1)
    result = SearchResult(chunk=chunk, score=0.75, source=SearchMode.HYBRID)
    text = str(result)
    assert "src/foo.py:1-1" in text
    assert "0.750" in text
    assert "def foo(): pass" in text
