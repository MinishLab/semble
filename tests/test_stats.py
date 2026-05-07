import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from semble.stats import log_search_stats
from semble.types import SearchMode, SearchResult
from tests.conftest import make_chunk


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
