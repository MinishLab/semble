from semble.types import Chunk, SearchMode, SearchResult


def test_search_result_str_includes_location_score_content() -> None:
    """SearchResult's __str__ renders location (via Chunk.location), score, and content."""
    chunk = Chunk(content="def foo(): pass", file_path="src/foo.py", start_line=1, end_line=1)
    result = SearchResult(chunk=chunk, score=0.75, source=SearchMode.HYBRID)
    text = str(result)
    assert "src/foo.py:1-1" in text
    assert "0.750" in text
    assert "def foo(): pass" in text
