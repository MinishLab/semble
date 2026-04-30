import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import _CLAUDE_FILE_PATH, _cli_main, _run_init, main
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk

_CLAUDE_AGENT_FILE = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")


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
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out


def test_cli_find_related(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints results."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.chunks = [chunk]
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8, source=SearchMode.SEMANTIC)]
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "src/bar.py", "1", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
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
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
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
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
    assert exc_info.value.code == 1
    assert "No chunk found" in capsys.readouterr().err


def test_init_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """_run_init writes the agent file and prints its path."""
    monkeypatch.chdir(tmp_path)
    _run_init()
    dest = tmp_path / _CLAUDE_FILE_PATH
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == _CLAUDE_AGENT_FILE
    assert str(_CLAUDE_FILE_PATH) in capsys.readouterr().out


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
    assert dest.read_text(encoding="utf-8") == _CLAUDE_AGENT_FILE


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
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        main()
    assert "query text" in capsys.readouterr().out


def test_cli_works_without_mcp_installed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI subcommands succeed even when the mcp package is not installed."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)]
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path"])
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    assert "query" in capsys.readouterr().out


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = _CLAUDE_AGENT_FILE.split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)
