import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from semble import SembleIndex

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
