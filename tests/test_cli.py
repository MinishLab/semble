"""Tests for semble.cli."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from semble.cli import main
from semble.types import Chunk, IndexStats, SearchResult


def _make_chunk(content: str = "def foo(): pass", file_path: str = "foo.py") -> Chunk:
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=1,
        language="python",
        content_hash="abc123",
    )


def _make_index_mock(
    chunks: list[SearchResult] | None = None, stats: IndexStats | None = None
) -> MagicMock:
    mock = MagicMock()
    mock.index_directory.return_value = stats or IndexStats(
        total_files=2,
        total_chunks=5,
        index_time_ms=100.0,
        embedding_time_ms=50.0,
        languages={"python": 5},
    )
    mock.search.return_value = (
        [SearchResult(chunk=_make_chunk(), score=0.9, source="hybrid")]
        if chunks is None
        else chunks
    )
    return mock


@patch("semble.SembleIndex")
def test_search_command_output(mock_cls: Any, tmp_project: Path, capsys: Any) -> None:
    mock_cls.return_value = _make_index_mock()
    import sys

    sys.argv = ["semble", "search", "authenticate", str(tmp_project)]
    main()
    out = capsys.readouterr().out
    assert "#1" in out
    assert "hybrid" in out


@patch("semble.SembleIndex")
def test_search_no_results(mock_cls: Any, tmp_project: Path, capsys: Any) -> None:
    mock = _make_index_mock(chunks=[])
    mock_cls.return_value = mock
    import sys

    sys.argv = ["semble", "search", "nothing", str(tmp_project)]
    main()
    out = capsys.readouterr().out
    assert "No results" in out


def test_no_command_exits_nonzero(capsys: Any) -> None:
    import sys

    sys.argv = ["semble"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code != 0
