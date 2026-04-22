import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from semble import SembleIndex
from semble.index.create import create_index_from_path
from semble.types import Encoder


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project."""
    return SembleIndex.from_path(tmp_project, model=mock_model)


@pytest.mark.parametrize(
    ("include_text_files", "md_in_results"),
    [(False, False), (True, True)],
)
def test_index_markdown_inclusion(
    mock_model: Encoder, tmp_project: Path, include_text_files: bool, md_in_results: bool
) -> None:
    """Markdown files are excluded by default and included when include_text_files=True."""
    _, _, chunks = create_index_from_path(tmp_project, mock_model, include_text_files=include_text_files)
    has_md = ".md" in {Path(c.file_path).suffix for c in chunks}
    assert has_md is md_in_results


def test_index_empty_returns_zero_chunks(mock_model: Encoder, tmp_path: Path) -> None:
    """Indexing an empty directory yields zero files and chunks."""
    with pytest.raises(ValueError):
        create_index_from_path(tmp_path, mock_model)


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    """Language breakdown in stats includes python with at least one chunk."""
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


@pytest.mark.parametrize(
    "query, mode",
    [("authenticate token", "hybrid"), ("authenticate", "bm25"), ("authentication", "semantic")],
)
def test_search_modes(indexed_index: SembleIndex, query: str, mode: str) -> None:
    """Each search mode returns a valid list of at most top_k results."""
    results = indexed_index.search(query, top_k=3, mode=mode)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_invalid_mode(indexed_index: SembleIndex) -> None:
    """An unrecognised mode string raises ValueError."""
    with pytest.raises(ValueError):
        indexed_index.search("query", mode="invalid")


def test_search_constraints(indexed_index: SembleIndex) -> None:
    """search: top_k is respected; no duplicate chunks are returned."""
    assert len(indexed_index.search("function", top_k=1, mode="bm25")) <= 1

    results = indexed_index.search("authenticate", top_k=5)
    assert len(results) == len(set(r.chunk for r in results))


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
def test_search_with_filter_paths_does_not_crash(indexed_index: SembleIndex, mode: str) -> None:
    """Filtered search works regardless of where the selected chunk lives in the corpus."""
    target_path = indexed_index.chunks[-1].file_path
    results = indexed_index.search("function", top_k=3, mode=mode, filter_paths=[target_path])
    assert all(r.chunk.file_path == target_path for r in results)


@pytest.mark.parametrize("mode", ["bm25", "hybrid", "semantic"])
@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_search_empty_query_returns_empty(indexed_index: SembleIndex, mode: str, query: str) -> None:
    """Empty / whitespace-only queries return [] across all modes."""
    assert indexed_index.search(query, mode=mode) == []


def test_find_related(indexed_index: SembleIndex) -> None:
    """find_related: returns similar chunks for a known location; returns [] for an unknown file."""
    chunk = indexed_index.chunks[0]
    results = indexed_index.find_related(chunk.file_path, chunk.start_line, top_k=3)
    assert isinstance(results, list)
    assert all(r.chunk != chunk for r in results)
    assert len(results) <= 3

    assert indexed_index.find_related("/does/not/exist.py", 1) == []


_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "t@t.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "t@t.com",
}


def _make_git_repo(path: Path) -> None:
    """Initialise a bare git repo at path; author identity comes from _GIT_ENV."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)


def _commit_file(repo: Path, name: str, content: str, message: str = "add file") -> None:
    """Write a file, stage it, and commit it inside repo."""
    (repo / name).write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", name], check=True, capture_output=True, env=_GIT_ENV)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True, capture_output=True, env=_GIT_ENV)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal local git repository with one Python file."""
    _make_git_repo(tmp_path)
    _commit_file(tmp_path, "main.py", "def hello():\n    return 'hello'\n")
    return tmp_path


def test_from_git_indexes_local_repo_with_relative_paths(mock_model: Any, git_repo: Path) -> None:
    """from_git clones a local repo, indexes it, and keeps chunk paths repo-relative."""
    idx = SembleIndex.from_git(str(git_repo), model=mock_model)
    assert idx.stats.indexed_files >= 1
    assert idx.stats.total_chunks > 0
    assert any("main.py" in c.file_path for c in idx.chunks)
    assert all(not Path(c.file_path).is_absolute() for c in idx.chunks)


def test_from_git_with_branch(mock_model: Any, tmp_path: Path) -> None:
    """from_git with ref= checks out the specified branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_git_repo(repo)
    _commit_file(repo, "main.py", "def on_main(): pass\n", "main")
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature"], check=True, capture_output=True)
    _commit_file(repo, "feature.py", "def on_feature(): pass\n", "feature")

    idx = SembleIndex.from_git(str(repo), ref="feature", model=mock_model)
    file_names = {Path(c.file_path).name for c in idx.chunks}
    assert "feature.py" in file_names


@pytest.mark.parametrize(
    ("kind", "expected_exc"),
    [("missing", FileNotFoundError), ("file", NotADirectoryError)],
)
def test_from_path_rejects_invalid_paths(
    mock_model: Any, tmp_path: Path, kind: str, expected_exc: type[Exception]
) -> None:
    """from_path raises FileNotFoundError for missing paths and NotADirectoryError for files."""
    if kind == "missing":
        target = tmp_path / "does_not_exist"
    else:
        target = tmp_path / "not_a_dir.py"
        target.write_text("x = 1\n")
    with pytest.raises(expected_exc):
        SembleIndex.from_path(target, model=mock_model)


def test_from_git_raises_on_failure(mock_model: Any) -> None:
    """from_git raises RuntimeError when the clone fails or git is not installed."""
    with pytest.raises(RuntimeError, match="git clone failed"):
        SembleIndex.from_git("/nonexistent/path/that/does/not/exist", model=mock_model)

    with patch("semble.index.index.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="git is not installed"):
            SembleIndex.from_git("https://github.com/x/y", model=mock_model)
