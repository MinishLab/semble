import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import Agent, _agent_path, _cli_main, _run_index, _run_init, main
from semble.types import ContentType, SearchResult
from tests.conftest import make_chunk

_CLAUDE_FILE_PATH = _agent_path(Agent.CLAUDE)


@pytest.mark.parametrize(
    "argv",
    [
        ["semble", "/some/path", "--ref", "main"],
        ["semble"],
    ],
)
def test_main_calls_asyncio_run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """main() delegates to asyncio.run(serve(...)) when no CLI subcommand is given."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("asyncio.run") as mock_run:
        mock_run.side_effect = lambda coro: coro.close()
        main()
    mock_run.assert_called_once()


@pytest.mark.parametrize(
    "argv, expected_in_output",
    [
        (["semble", "search", "query text", "/some/path"], ["query text", "0.9"]),
        (["semble", "search", "nothing", "/some/path", "--top-k", "3"], ["No results found"]),
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
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)] if has_results else []
    monkeypatch.setattr(sys, "argv", argv)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out


@pytest.mark.parametrize(
    ("scenario", "expected_stdout", "expected_stderr", "expected_exit_code"),
    [
        ("with_results", ["src/bar.py", "0.800"], None, None),
        ("no_results", ["No related chunks found"], None, None),
        ("unknown_chunk", [], "No chunk found", 1),
    ],
)
def test_cli_find_related(
    scenario: str,
    expected_stdout: list[str],
    expected_stderr: str | None,
    expected_exit_code: int | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints results, empty states, and missing-chunk errors."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.chunks = [] if scenario == "unknown_chunk" else [chunk]
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8)] if scenario == "with_results" else []
    file_path = "unknown.py" if scenario == "unknown_chunk" else "src/bar.py"
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", file_path, "1", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        if expected_exit_code is None:
            _cli_main()
        else:
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == expected_exit_code
    captured = capsys.readouterr()
    for fragment in expected_stdout:
        assert fragment in captured.out
    if expected_stderr:
        assert expected_stderr in captured.err


@pytest.mark.parametrize("agent", list(Agent))
def test_init_creates_file(
    agent: Agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_init writes the correct agent file for every supported agent."""
    monkeypatch.chdir(tmp_path)
    _run_init(agent=agent)
    dest = tmp_path / _agent_path(agent)
    expected = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == expected
    assert str(_agent_path(agent)) in capsys.readouterr().out


def test_init_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_init exits with code 1 when the file exists and force=False."""
    monkeypatch.chdir(tmp_path)
    _run_init()
    with pytest.raises(SystemExit) as exc_info:
        _run_init()
    assert exc_info.value.code == 1
    assert "already exists" in capsys.readouterr().err


def test_init_overwrites_with_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_init overwrites an existing file when force=True."""
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / _CLAUDE_FILE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old content", encoding="utf-8")
    _run_init(force=True)
    assert dest.read_text(encoding="utf-8") == files("semble").joinpath("agents/claude.md").read_text(encoding="utf-8")


def test_init_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Semble init creates the Claude agent file via _cli_main."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["semble", "init"])
    _cli_main()
    assert (tmp_path / _CLAUDE_FILE_PATH).exists()
    assert str(_CLAUDE_FILE_PATH) in capsys.readouterr().out


def test_main_dispatches_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes to _cli_main when first argument is a CLI subcommand."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        main()
    assert "query text" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected_stdout", "expect_system_exit"),
    [
        (["semble", "--help"], "find-related", True),
        (["semble", "search", "query", "/some/path"], "query", False),
    ],
)
def test_cli_entrypoint_works_without_mcp_installed(
    argv: list[str],
    expected_stdout: str,
    expect_system_exit: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI entrypoint paths succeed even when the mcp package is not installed."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    monkeypatch.setitem(sys.modules, "semble.mcp", None)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        if expect_system_exit:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        else:
            main()
    assert expected_stdout in capsys.readouterr().out


def test_mcp_main_exits_with_message_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_mcp_main prints an actionable message and exits when mcp extras are not installed."""
    monkeypatch.setattr(sys, "argv", ["semble"])
    with patch("semble.cli.find_spec", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
    assert "pip install 'semble[mcp]'" in capsys.readouterr().err


def test_run_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_index creates the output directory and saves the index."""
    out_dir = tmp_path / "index_output"
    fake_index = MagicMock()
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _run_index(path="/some/path", include_text_files=True, out=str(out_dir))
    mock_from_path.assert_called_once_with("/some/path", include_text_files=True)
    assert out_dir.exists()
    fake_index.save.assert_called_once_with(str(out_dir))


def test_index_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli_main index subcommand calls _run_index with the correct arguments."""
    out_dir = tmp_path / "built_index"
    fake_index = MagicMock()
    monkeypatch.setattr(sys, "argv", ["semble", "index", "/some/path", "-o", str(out_dir)])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    assert out_dir.exists()
    fake_index.save.assert_called_once_with(str(out_dir))


def test_index_git_via_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_cli_main index subcommand calls _run_index with the correct arguments."""
    out_dir = tmp_path / "built_index"
    fake_index = MagicMock()
    monkeypatch.setattr(sys, "argv", ["semble", "index", "git://xyz.git", "-o", str(out_dir)])
    with patch("semble.cli.SembleIndex.from_git", return_value=fake_index):
        _cli_main()
    assert out_dir.exists()
    fake_index.save.assert_called_once_with(str(out_dir))


def test_cli_search_with_prebuilt_index(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """_cli_main search with --index loads the pre-built index from disk."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.95)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", ".", "--index", "/some/prebuilt"])
    with patch("semble.cli.SembleIndex.load_from_disk", return_value=fake_index) as mock_load:
        _cli_main()
    mock_load.assert_called_once_with("/some/prebuilt")
    out = capsys.readouterr().out
    assert "query text" in out
    assert "0.95" in out


def test_include_text_files_cli_deprecated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--include-text-files on CLI raises DeprecationWarning."""
    import warnings

    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path", "--include-text-files"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _cli_main()
    assert any(
        "include-text-files" in str(w.message).lower() for w in caught if issubclass(w.category, DeprecationWarning)
    )


@pytest.mark.parametrize(
    ("argv_content", "expected"),
    [
        (["--content", "code"], [ContentType.CODE]),
        (["--content", "code", "docs"], [ContentType.CODE, ContentType.DOCS]),
        (["--content", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        (["--content", "code", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        ([], [ContentType.CODE]),
    ],
)
def test_cli_content_argument(
    argv_content: list[str],
    expected: list[ContentType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--content parses into the right ContentType list (including the 'all' shorthand and default)."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path", *argv_content])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()
    assert list(mock_from_path.call_args.kwargs["content"]) == expected


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = files("semble").joinpath("agents/claude.md").read_text(encoding="utf-8").split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)
