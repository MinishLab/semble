import json
import sys
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import _CLAUDE_FILE_PATH, _cli_main, _parse_stats, _run_init, _run_stats, main
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
    fake_index.find_related.return_value = (
        [SearchResult(chunk=chunk, score=0.8, source=SearchMode.SEMANTIC)] if scenario == "with_results" else []
    )
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
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)]
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


def _make_stats_record(ts: str, call: str = "search", snippet_chars: int = 100, file_chars: int = 500) -> str:
    return json.dumps({"ts": ts, "call": call, "results": 3, "snippet_chars": snippet_chars, "file_chars": file_chars})


def test_stats_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """_run_stats prints a friendly message when no stats file exists yet."""
    monkeypatch.setattr("semble.cli._STATS_FILE", tmp_path / "nonexistent.jsonl")
    _run_stats()
    assert "No stats yet" in capsys.readouterr().out


def test_stats_valid_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_stats displays period buckets and savings header."""
    stats_file = tmp_path / "stats.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    stats_file.write_text(
        _make_stats_record(now, call="search") + "\n" + _make_stats_record(now, call="find_related") + "\n"
    )
    monkeypatch.setattr("semble.cli._STATS_FILE", stats_file)
    _run_stats()
    out = capsys.readouterr().out
    assert "Savings" in out
    assert "Savings" in out
    assert "Today" in out


def test_stats_verbose_shows_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_stats --verbose adds a usage breakdown by call type."""
    stats_file = tmp_path / "stats.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    stats_file.write_text(
        _make_stats_record(now, call="search") + "\n" + _make_stats_record(now, call="find_related") + "\n"
    )
    monkeypatch.setattr("semble.cli._STATS_FILE", stats_file)
    _run_stats(verbose=True)
    out = capsys.readouterr().out
    assert "Usage breakdown" in out
    assert "search" in out
    assert "find_related" in out


@pytest.mark.parametrize(
    "bad_line",
    [
        "not valid json",
        json.dumps({"ts": "not-a-date", "call": "search", "snippet_chars": 100, "file_chars": 500}),
    ],
    ids=["malformed-json", "malformed-timestamp"],
)
def test_stats_tolerates_bad_records(
    bad_line: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_stats skips bad JSON and bad timestamps without crashing."""
    stats_file = tmp_path / "stats.jsonl"
    stats_file.write_text(bad_line + "\n")
    monkeypatch.setattr("semble.cli._STATS_FILE", stats_file)
    _run_stats()
    assert "Savings" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["semble", "savings"], "No stats yet"),
        (["semble", "savings", "--verbose"], "No stats yet"),
    ],
    ids=["default", "verbose"],
)
def test_stats_cli_dispatch(
    argv: list[str], expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Semble stats subcommand dispatches to _run_stats, with and without --verbose."""
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("semble.cli._STATS_FILE", tmp_path / "nonexistent.jsonl")
    _cli_main()
    assert expected in capsys.readouterr().out


def test_stats_buckets_exclude_old_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Records older than 7 days count in All time but not Today or Last 7 days."""
    stats_file = tmp_path / "stats.jsonl"
    old_ts = "2020-01-01T00:00:00+00:00"
    now_ts = datetime.now(timezone.utc).isoformat()
    stats_file.write_text(_make_stats_record(old_ts) + "\n" + _make_stats_record(now_ts) + "\n")
    monkeypatch.setattr("semble.cli._STATS_FILE", stats_file)
    buckets, _ = _parse_stats()
    assert buckets["All time"]["calls"] == 2
    assert buckets["Today"]["calls"] == 1
    assert buckets["Last 7 days"]["calls"] == 1


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = _CLAUDE_AGENT_FILE.split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)
