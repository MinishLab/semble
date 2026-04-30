import sys
from importlib.resources import files
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semble.mcp import (
    _AGENT_FILE_PATH,
    _cli_main,
    _format_results,
    _IndexCache,
    _is_git_url,
    _resolve_chunk,
    _run_init,
    create_server,
    main,
    serve,
)
from semble.types import Chunk, Encoder, SearchMode, SearchResult
from tests.conftest import make_chunk

_AGENT_FILE = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")


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


@pytest.mark.parametrize(
    "argv, expected_in_output",
    [
        (["semble", "search", "query text", "/some/path"], ["query text", "0.9"]),
        (["semble", "search", "nothing", "/some/path", "--top-k", "3", "--mode", "bm25"], ["No results found"]),
    ],
)
def test_cli_search(
    argv: list[str],
    expected_in_output: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main search subcommand calls index.search and prints results."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    has_results = "No results" not in expected_in_output[0]
    fake_index.search.return_value = (
        [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)] if has_results else []
    )
    monkeypatch.setattr(sys, "argv", argv)
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out


def test_cli_find_related(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related subcommand calls index.find_related and prints results."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.chunks = [chunk]
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8, source=SearchMode.SEMANTIC)]
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "src/bar.py", "1", "/some/path"])
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    assert "src/bar.py" in out
    assert "0.800" in out


def test_cli_find_related_no_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints a message when the index returns no related chunks."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.chunks = [chunk]
    fake_index.find_related.return_value = []
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "src/bar.py", "1", "/some/path"])
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    assert "No related chunks found" in capsys.readouterr().out


def test_cli_find_related_unknown_chunk(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related exits with code 1 when chunk is not found."""
    fake_index = MagicMock()
    fake_index.chunks = []
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "unknown.py", "1", "/some/path"])
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
    assert exc_info.value.code == 1
    assert "No chunk found" in capsys.readouterr().err


def test_init_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """_run_init writes the agent file and prints its path."""
    monkeypatch.chdir(tmp_path)
    _run_init()
    dest = tmp_path / _AGENT_FILE_PATH
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == _AGENT_FILE
    assert str(_AGENT_FILE_PATH) in capsys.readouterr().out


def test_init_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_init exits with code 1 when the file exists and --force is not set."""
    monkeypatch.chdir(tmp_path)
    _run_init()
    with pytest.raises(SystemExit) as exc_info:
        _run_init()
    assert exc_info.value.code == 1
    assert "already exists" in capsys.readouterr().err


def test_init_overwrites_with_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_init overwrites an existing file when force=True."""
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / _AGENT_FILE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old content", encoding="utf-8")
    _run_init(force=True)
    assert dest.read_text(encoding="utf-8") == _AGENT_FILE


def test_init_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Semble init subcommand creates the agent file via _cli_main."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["semble", "init"])
    _cli_main()
    assert (tmp_path / _AGENT_FILE_PATH).exists()
    assert str(_AGENT_FILE_PATH) in capsys.readouterr().out


def test_main_dispatches_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes to _cli_main when first argument is a CLI subcommand."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", "/some/path"])
    with patch("semble.mcp.SembleIndex.from_path", return_value=fake_index):
        main()
    assert "query text" in capsys.readouterr().out
