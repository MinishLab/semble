import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semble.cli import _cli_main
from semble.stats import build_savings_summary, format_savings_report, save_search_stats
from semble.types import CallType, SearchMode, SearchResult
from tests.conftest import make_chunk


def _make_stats_record(ts: float, call: str = "search", snippet_chars: int = 1_000, file_chars: int = 20_000) -> str:
    return json.dumps({"ts": ts, "call": call, "results": 3, "snippet_chars": snippet_chars, "file_chars": file_chars})


@pytest.fixture
def sample_stats_file(tmp_path: Path) -> Path:
    """Stats file with one search and one find_related record from today."""
    stats_file = tmp_path / "stats.jsonl"
    now = datetime.now(timezone.utc).timestamp()
    stats_file.write_text(
        _make_stats_record(now, call="search") + "\n" + _make_stats_record(now, call="find_related") + "\n"
    )
    return stats_file


def test_save_search_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_search_stats deduplicates file paths and silences write errors."""
    chunk = make_chunk("hello", "src/foo.py")
    result = SearchResult(chunk=chunk, score=0.9, source=SearchMode.HYBRID)
    stats_file = tmp_path / "stats.jsonl"
    monkeypatch.setattr("semble.stats._STATS_FILE", stats_file)
    save_search_stats([result, result], CallType.SEARCH, {"src/foo.py": 42})
    assert json.loads(stats_file.read_text())["file_chars"] == 42

    mock_path = MagicMock()
    mock_path.parent.mkdir.return_value = None
    mock_path.open.side_effect = OSError("no write")
    monkeypatch.setattr("semble.stats._STATS_FILE", mock_path)
    save_search_stats([result], CallType.SEARCH, {"src/foo.py": 42})  # must not raise


def test_savings_no_file(tmp_path: Path) -> None:
    """format_savings_report returns a friendly message when no stats file exists yet."""
    assert "No stats yet" in format_savings_report(path=tmp_path / "nonexistent.jsonl")


@pytest.mark.parametrize(
    ("verbose", "expected"),
    [
        (False, ["Savings", "Today"]),
        (True, ["Savings", "Today", "Usage Breakdown", "search", "find_related"]),
    ],
    ids=["default", "verbose"],
)
def test_savings_output(sample_stats_file: Path, verbose: bool, expected: list[str]) -> None:
    """format_savings_report displays period buckets; --verbose adds call-type breakdown."""
    result = format_savings_report(path=sample_stats_file, verbose=verbose)
    for s in expected:
        assert s in result


def test_savings_output_millions(tmp_path: Path) -> None:
    """Token counts >= 1M are formatted as M, not k."""
    stats_file = tmp_path / "stats.jsonl"
    stats_file.write_text(
        _make_stats_record(datetime.now(timezone.utc).timestamp(), snippet_chars=0, file_chars=4_000_000) + "\n"
    )
    assert "M tokens" in format_savings_report(path=stats_file)


def test_savings_do_not_subtract_unknown_baselines(tmp_path: Path) -> None:
    """Rows without a full-file baseline must not cancel out real savings."""
    stats_file = tmp_path / "stats.jsonl"
    now = datetime.now(timezone.utc).timestamp()
    stats_file.write_text(
        _make_stats_record(now, snippet_chars=100, file_chars=500)
        + "\n"
        + _make_stats_record(now, snippet_chars=1_000, file_chars=0)
        + "\n"
    )

    summary = build_savings_summary(path=stats_file)
    assert summary.buckets["All time"].saved_chars == 400
    assert "~100 tokens" in format_savings_report(path=stats_file)


def test_savings_tolerates_bad_json(tmp_path: Path) -> None:
    """Malformed JSON lines are skipped with a warning."""
    stats_file = tmp_path / "stats.jsonl"
    stats_file.write_text("not valid json\n")
    assert "Savings" in format_savings_report(path=stats_file)


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
    """Savings subcommand dispatches to format_savings_report, with and without --verbose."""
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("semble.stats._STATS_FILE", tmp_path / "nonexistent.jsonl")
    _cli_main()
    assert expected in capsys.readouterr().out


def test_savings_buckets_exclude_old_records(tmp_path: Path) -> None:
    """Records older than 7 days count in All time but not Today or Last 7 days."""
    stats_file = tmp_path / "stats.jsonl"
    old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    now_ts = datetime.now(timezone.utc).timestamp()
    stats_file.write_text(_make_stats_record(old_ts) + "\n" + _make_stats_record(now_ts) + "\n")
    summary = build_savings_summary(path=stats_file)
    assert summary.buckets["All time"].calls == 2
    assert summary.buckets["Today"].calls == 1
    assert summary.buckets["Last 7 days"].calls == 1
