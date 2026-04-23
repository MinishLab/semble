import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semble.mcp import _format_results, _IndexCache, _is_git_url, _resolve_chunk, create_server, main, serve
from semble.types import Chunk, Encoder, SearchMode, SearchResult
from tests.conftest import make_chunk


def _tool_text(result: Any) -> str:
    """Extract the text string from a FastMCP call_tool result."""
    return result[0][0].text


async def _call_tool(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    *,
    index_method: str,
    index_return: list[SearchResult],
    index_chunks: list[Chunk] | None = None,
    default_source: str | None = "/some/path",
) -> str:
    """Patch SembleIndex.from_path with a fake index and invoke the tool, returning the text."""
    fake_index = MagicMock()
    getattr(fake_index, index_method).return_value = index_return
    if index_chunks is not None:
        fake_index.chunks = index_chunks
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        server = create_server(cache, default_source=default_source)
        result = await server.call_tool(tool, args)
    return _tool_text(result)


@pytest.fixture()
def cache() -> _IndexCache:
    """An _IndexCache backed by a stub model."""
    return _IndexCache(model=MagicMock(spec=Encoder))


def test_resolve_chunk() -> None:
    """_resolve_chunk returns the correct chunk and handles boundary and miss cases."""
    interior = make_chunk("line1\nline2\nline3", "src/a.py")  # start=1, end=3
    boundary = make_chunk("last line", "src/a.py")  # start=1, end=1 (single-line)

    # Line strictly inside a multi-line chunk hits the early-return path.
    assert _resolve_chunk([interior], "src/a.py", 2) is interior

    # Line equal to end_line of a single-line chunk hits the fallback path.
    assert _resolve_chunk([boundary], "src/a.py", 1) is boundary

    # Unknown file returns None.
    assert _resolve_chunk([interior], "src/other.py", 1) is None

    # Line out of range returns None.
    assert _resolve_chunk([interior], "src/a.py", 99) is None


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


def test_format_results() -> None:
    """_format_results: empty list → header only; with results → numbered fenced blocks with scores."""
    empty_out = _format_results("My header", [])
    assert "My header" in empty_out
    assert "```" not in empty_out

    chunks = [make_chunk(f"def fn_{i}(): pass", f"f{i}.py") for i in range(3)]
    results = [
        SearchResult(chunk=c, score=round(0.1 * (i + 1), 3), source=SearchMode.HYBRID) for i, c in enumerate(chunks)
    ]
    out = _format_results("Results for: 'foo'", results)
    assert "Results for: 'foo'" in out
    assert out.count("```") >= len(results) * 2  # opening + closing fence each
    for i, c in enumerate(chunks, start=1):
        assert f"## {i}." in out
        assert c.content in out
    assert "0.100" in out and "0.200" in out and "0.300" in out


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source", "patch_target"),
    [
        ("local_tmp_path", "from_path"),
        ("https://github.com/org/repo", "from_git"),
    ],
    ids=["local_path", "git_url"],
)
async def test_index_cache_builds_and_caches(
    cache: _IndexCache, tmp_path: Path, source: str, patch_target: str
) -> None:
    """_IndexCache.get() builds via the correct SembleIndex.* entrypoint and caches subsequent calls."""
    resolved_source = str(tmp_path) if source == "local_tmp_path" else source
    fake_index = MagicMock()
    with patch(f"semble.mcp.SembleIndex.{patch_target}", return_value=fake_index) as mock_build:
        first = await cache.get(resolved_source)
        second = await cache.get(resolved_source)
    assert first is fake_index
    assert second is fake_index
    mock_build.assert_called_once()


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
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("search", {"query": "foo", "repo": "https://github.com/x/y"}),
        ("find_related", {"file_path": "src/foo.py", "line": 1, "repo": "https://github.com/x/y"}),
    ],
)
async def test_tool_index_failure(cache: _IndexCache, tool: str, args: dict[str, object]) -> None:
    """Both tools return a friendly error message when indexing fails."""
    with patch("semble.mcp.SembleIndex.from_git", side_effect=RuntimeError("clone failed")):
        server = create_server(cache)
        result = await server.call_tool(tool, args)
    text = _tool_text(result)
    assert "Failed to index" in text
    assert "clone failed" in text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "args", "method", "results", "chunks", "expected_substrings"),
    [
        pytest.param(
            "search",
            {"query": "bar"},
            "search",
            [SearchResult(chunk=make_chunk("def bar(): pass", "src/bar.py"), score=0.9, source=SearchMode.HYBRID)],
            None,
            ["bar", "0.900"],
            id="search_with_results",
        ),
        pytest.param(
            "search",
            {"query": "nothing"},
            "search",
            [],
            None,
            ["No results found"],
            id="search_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1},
            "find_related",
            [SearchResult(chunk=make_chunk("class Foo: pass", "src/foo.py"), score=0.8, source=SearchMode.SEMANTIC)],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["src/foo.py:1", "0.800"],
            id="find_related_with_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/foo.py", "line": 1},
            "find_related",
            [],
            [make_chunk("class Foo: pass", "src/foo.py")],
            ["No related chunks found"],
            id="find_related_no_results",
        ),
        pytest.param(
            "find_related",
            {"file_path": "src/unknown.py", "line": 1},
            "find_related",
            [],
            [],
            ["No chunk found"],
            id="find_related_unknown_file",
        ),
    ],
)
async def test_tool_output(
    cache: _IndexCache,
    tool: str,
    args: dict[str, Any],
    method: str,
    results: list[SearchResult],
    chunks: list[Chunk] | None,
    expected_substrings: list[str],
) -> None:
    """Search and find_related format results (or an empty-state message) through the server."""
    text = await _call_tool(cache, tool, args, index_method=method, index_return=results, index_chunks=chunks)
    for substring in expected_substrings:
        assert substring in text


@pytest.mark.anyio
@pytest.mark.parametrize("with_path", [True, False], ids=["pre_index", "no_path"])
async def test_serve_runs_stdio(tmp_path: Path, with_path: bool) -> None:
    """serve() loads the model, runs stdio, and optionally pre-indexes when a path is given."""
    with (
        patch("semble.mcp.load_model", return_value=MagicMock(spec=Encoder)),
        patch("semble.mcp.SembleIndex.from_path", return_value=MagicMock()),
        patch("mcp.server.fastmcp.FastMCP.run_stdio_async", new_callable=AsyncMock) as mock_run,
    ):
        await (serve(str(tmp_path)) if with_path else serve())

    mock_run.assert_called_once()


@pytest.mark.parametrize(
    "argv",
    [
        ["semble", "/some/path", "--ref", "main"],
        ["semble"],
    ],
)
def test_main_calls_asyncio_run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """main() parses argv and delegates to asyncio.run(serve(...))."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("semble.mcp.asyncio.run") as mock_run:
        mock_run.side_effect = lambda coro: coro.close()
        main()
    mock_run.assert_called_once()
