"""Tests for semble.mcp: pure helpers, _IndexCache, and server tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from semble.mcp import _format_results, _IndexCache, _is_git_url, create_server
from semble.types import Encoder, SearchMode, SearchResult
from tests.conftest import make_chunk


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0][0].text


@pytest.fixture()
def cache() -> _IndexCache:
    """An _IndexCache backed by a stub model."""
    return _IndexCache(model=MagicMock(spec=Encoder))


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://github.com/org/repo", True),
        ("http://github.com/org/repo", True),
        ("git://github.com/org/repo", True),
        ("ssh://git@github.com/org/repo", True),
        ("git+ssh://git@github.com/org/repo", True),
        ("file:///tmp/repo", True),
        ("git@github.com:org/repo", True),  # scp-like
        ("/local/path/to/repo", False),
        ("./relative/path", False),
        ("repo_name", False),
    ],
)
def test_is_git_url(path: str, expected: bool) -> None:
    """Remote git URLs are detected; local paths are not."""
    assert _is_git_url(path) is expected


def test_format_results_empty() -> None:
    """Empty results list produces a header with no code blocks."""
    out = _format_results("My header", [])
    assert "My header" in out
    assert "```" not in out


def test_format_results_with_results() -> None:
    """Each result is rendered as a numbered fenced code block with score."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    result = SearchResult(chunk=chunk, score=0.85, source=SearchMode.HYBRID)
    out = _format_results("Results for: 'foo'", [result])
    assert "Results for: 'foo'" in out
    assert "0.850" in out
    assert "def foo(): pass" in out
    assert "```" in out
    assert "1." in out


def test_format_results_numbering() -> None:
    """Results are numbered sequentially starting at 1."""
    chunks = [make_chunk(f"line {i}", f"f{i}.py") for i in range(3)]
    results = [SearchResult(chunk=c, score=float(i), source=SearchMode.BM25) for i, c in enumerate(chunks)]
    out = _format_results("header", results)
    assert "## 1." in out
    assert "## 2." in out
    assert "## 3." in out


@pytest.mark.anyio
async def test_index_cache_local_path(cache: _IndexCache, tmp_path: Path) -> None:
    """_IndexCache.get() builds a local-path index and caches it."""
    fake_index = MagicMock()
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index) as mock_fp:
        index1 = await cache.get(str(tmp_path))
        index2 = await cache.get(str(tmp_path))
    assert index1 is fake_index
    assert index2 is fake_index
    mock_fp.assert_called_once()


@pytest.mark.anyio
async def test_index_cache_git_url(cache: _IndexCache) -> None:
    """_IndexCache.get() builds a git-URL index and caches it."""
    fake_index = MagicMock()
    with patch("semble.mcp.SembleIndex.from_git", return_value=fake_index) as mock_fg:
        index = await cache.get("https://github.com/org/repo")
    assert index is fake_index
    mock_fg.assert_called_once()


@pytest.mark.anyio
async def test_index_cache_evicts_on_failure(cache: _IndexCache, tmp_path: Path) -> None:
    """A failed build evicts the entry so the next call can retry."""
    call_count = 0

    def _failing_then_ok(path: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("build failed")
        return MagicMock()

    with patch("semble.mcp.SembleIndex.from_path", side_effect=_failing_then_ok):
        with pytest.raises(RuntimeError, match="build failed"):
            await cache.get(str(tmp_path))
        result = await cache.get(str(tmp_path))
    assert result is not None
    assert call_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo"}),
        ("find_related", {"file_path": "src/foo.py", "line": 10}),
    ],
)
async def test_tool_no_repo_no_default(cache: _IndexCache, tool: str, args: dict[str, object]) -> None:
    """Both tools return an error message when no repo and no default source are given."""
    server = create_server(cache, default_source=None)
    result = await server.call_tool(tool, args)
    assert "No repo specified" in _tool_text(result)


@pytest.mark.anyio
async def test_search_tool_returns_results(cache: _IndexCache) -> None:
    """Search tool formats and returns index results."""
    fake_index = MagicMock()
    chunk = make_chunk("def bar(): pass", "src/bar.py")
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)]
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source="/some/local/path")
        result = await server.call_tool("search", {"query": "bar"})
    text = _tool_text(result)
    assert "bar" in text
    assert "0.900" in text


@pytest.mark.anyio
async def test_search_tool_index_failure(cache: _IndexCache) -> None:
    """Search tool returns a friendly error when indexing fails."""
    with patch("semble.mcp.SembleIndex.from_git", side_effect=RuntimeError("clone failed")):
        server = create_server(cache)
        result = await server.call_tool("search", {"query": "foo", "repo": "https://github.com/x/y"})
    text = _tool_text(result)
    assert "Failed to index" in text
    assert "clone failed" in text


@pytest.mark.anyio
async def test_search_tool_no_results(cache: _IndexCache) -> None:
    """Search tool returns 'No results found.' when the index returns nothing."""
    fake_index = MagicMock()
    fake_index.search.return_value = []
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source="/some/path")
        result = await server.call_tool("search", {"query": "nothing"})
    assert "No results found" in _tool_text(result)


@pytest.mark.anyio
async def test_find_related_tool_returns_results(cache: _IndexCache) -> None:
    """find_related formats and returns related chunks."""
    fake_index = MagicMock()
    chunk = make_chunk("class Foo: pass", "src/foo.py")
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8, source=SearchMode.SEMANTIC)]
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source="/some/path")
        result = await server.call_tool("find_related", {"file_path": "src/foo.py", "line": 1})
    text = _tool_text(result)
    assert "src/foo.py:1" in text
    assert "0.800" in text


@pytest.mark.anyio
async def test_find_related_tool_no_results(cache: _IndexCache) -> None:
    """find_related returns a descriptive message when no related chunks are found."""
    fake_index = MagicMock()
    fake_index.find_related.return_value = []
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source="/some/path")
        result = await server.call_tool("find_related", {"file_path": "src/foo.py", "line": 99})
    assert "No related chunks found" in _tool_text(result)
