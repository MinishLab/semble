import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import _CLAUDE_FILE_PATH, _cli_main, _run_init, main
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk

_CLAUDE_AGENT_FILE = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")

_SEARCH_CHUNK = make_chunk("def foo(): pass", "src/foo.py")
_SEARCH_RESULT = SearchResult(chunk=_SEARCH_CHUNK, score=0.9, source=SearchMode.HYBRID)


@pytest.fixture()
def fake_search_index() -> MagicMock:
    """Fake index that returns one search result."""
    fake = MagicMock()
    fake.search.return_value = [_SEARCH_RESULT]
    return fake


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


_BAR_CHUNK = make_chunk("class Bar: pass", "src/bar.py")


@pytest.mark.parametrize(
    ("find_related_return", "expected_fragments"),
    [
        pytest.param(
            [SearchResult(chunk=_BAR_CHUNK, score=0.8, source=SearchMode.SEMANTIC)],
            ["src/bar.py", "0.800"],
            id="with_results",
        ),
        pytest.param([], ["No related chunks found"], id="no_results"),
    ],
)
def test_cli_find_related(
    find_related_return: list[SearchResult],
    expected_fragments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints results or an empty-state message."""
    fake_index = MagicMock()
    fake_index.chunks = [_BAR_CHUNK]
    fake_index.find_related.return_value = find_related_return
    monkeypatch.setattr(sys, "argv", ["semble", "find-related", "src/bar.py", "1", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out


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
    fake_search_index: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes to _cli_main when first argument is a CLI subcommand."""
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query text", "/some/path"])
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_search_index):
        main()
    assert "query text" in capsys.readouterr().out


def test_cli_works_without_mcp_installed(
    fake_search_index: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI subcommands succeed even when the mcp package is not installed."""
    monkeypatch.setattr(sys, "argv", ["semble", "search", "query", "/some/path"])
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with patch("semble.cli.SembleIndex.from_path", return_value=fake_search_index):
        _cli_main()
    assert "query" in capsys.readouterr().out


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = _CLAUDE_AGENT_FILE.split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)


@pytest.mark.slow
def test_cli_search_subprocess() -> None:
    """Semble search runs end-to-end as a subprocess — the path a sub-agent takes via Bash."""
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["uv", "run", "semble", "search", "MCP server entry point", str(repo_root)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert "mcp.py" in result.stdout


@pytest.mark.slow
def test_cli_find_related_subprocess() -> None:
    """Semble find-related runs end-to-end as a subprocess — the path a sub-agent takes via Bash."""
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["uv", "run", "semble", "find-related", "src/semble/mcp.py", "30", str(repo_root)],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
