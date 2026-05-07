import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semble.cli import _cli_main, _parse_stats, _run_stats
from semble.stats import log_search_stats
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk


def _make_stats_record(ts: str, call: str = "search", snippet_chars: int = 100, file_chars: int = 500) -> str:
    return json.dumps({"ts": ts, "call": call, "results": 3, "snippet_chars": snippet_chars, "file_chars": file_chars})


@pytest.fixture
def sample_stats_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture for a stats file with two records."""
    stats_file = tmp_path / "stats.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    stats_file.write_text(
        _make_stats_record(now, call="search") + "\n" + _make_stats_record(now, call="find_related") + "\n"
    )
    monkeypatch.setattr("semble.cli._STATS_FILE", stats_file)
    return stats_file


def test_log_search_stats_deduplicates_file_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two results from the same file are counted as one file in file_chars."""
    chunk = make_chunk("hello", "src/foo.py")
    result = SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)
    stats_file = tmp_path / "stats.jsonl"
    monkeypatch.setattr("semble.stats._STATS_FILE", stats_file)
    log_search_stats([result, result], "search", {"src/foo.py": 42})
    record = json.loads(stats_file.read_text())
    assert record["file_chars"] == 42


def test_log_search_stats_silences_write_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors during stat recording are silently swallowed."""
    mock_path = MagicMock()
    mock_path.parent.mkdir.return_value = None
    mock_path.open.side_effect = OSError("no write")
    monkeypatch.setattr("semble.stats._STATS_FILE", mock_path)
    log_search_stats([], "search")  # must not raise


def test_savings_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """_run_stats prints a friendly message when no stats file exists yet."""
    monkeypatch.setattr("semble.cli._STATS_FILE", tmp_path / "nonexistent.jsonl")
    _run_stats()
    assert "No stats yet" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [
        (False, ["Savings", "Today"]),
        (True, ["Savings", "Today", "Usage Breakdown", "search", "find_related"]),
    ],
    ids=["default", "verbose"],
)
def test_savings_output(
    sample_stats_file: Path, verbose: bool, expected: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_stats displays period buckets; --verbose adds call-type breakdown."""
    _run_stats(verbose=verbose)
    out = capsys.readouterr().out
    for s in expected:
        assert s in out


@pytest.mark.parametrize(
    "bad_line",
    [
        "not valid json",
        json.dumps({"ts": "not-a-date", "call": "search", "snippet_chars": 100, "file_chars": 500}),
    ],
    ids=["malformed-json", "malformed-timestamp"],
)
def test_savings_tolerates_bad_records(
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
def test_savings_cli_dispatch(
    argv: list[str], expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Savings subcommand dispatches to _run_stats, with and without --verbose."""
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("semble.cli._STATS_FILE", tmp_path / "nonexistent.jsonl")
    _cli_main()
    assert expected in capsys.readouterr().out


def test_savings_buckets_exclude_old_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
