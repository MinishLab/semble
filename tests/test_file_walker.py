import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from semble.index.file_walker import language_for_path, walk_files


def test_languagef_for_path() -> None:
    """Test language_for_path returns the correct language for a given path."""
    assert language_for_path(Path("foo.py")) == "python"
    assert language_for_path(Path("bar.js")) == "javascript"
    assert language_for_path(Path("dangerous.exe")) is None


def _touch(path: Path, content: str = "x = 1\n") -> None:
    """Create path (and any missing parents) and write content to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.mark.parametrize(
    ("files", "gitignore", "expected"),
    [
        # Default-ignored dirs (.venv, node_modules, .cache) are always skipped.
        (
            ["src/a.py", ".venv/lib/b.py", "node_modules/pkg/c.py", ".cache/uv/d.py"],
            None,
            {"src/a.py"},
        ),
        # Root .gitignore excludes both directories and files.
        (
            ["src/keep.py", "local/ignored.py", "generated.py"],
            "local/\ngenerated.py\n",
            {"src/keep.py"},
        ),
        # Negation (`!`) patterns re-include previously ignored files.
        (
            ["out/a.py", "out/keep.py"],
            "out/*\n!out/keep.py\n",
            {"out/keep.py"},
        ),
    ],
)
def test_walk_files_filtering(tmp_path: Path, files: list[str], gitignore: str | None, expected: set[str]) -> None:
    """Directory defaults, gitignore patterns, and negations filter the yielded files."""
    for rel in files:
        _touch(tmp_path / rel)
    if gitignore is not None:
        (tmp_path / ".gitignore").write_text(gitignore)

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, frozenset({".py"}))}
    assert found == expected


def test_walk_files_prunes_ignored_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignored directories are pruned so os.walk never descends into them."""
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "node_modules" / "deep" / "deeper" / "b.js")

    visited: list[str] = []
    real_walk = os.walk

    def tracking_walk(top: str) -> Iterator[tuple[str, list[str], list[str]]]:
        for dirpath, dirnames, filenames in real_walk(top):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr("semble.index.file_walker.os.walk", tracking_walk)
    list(walk_files(tmp_path, frozenset({".py", ".js"})))
    assert not any("node_modules" in v for v in visited[1:]), visited
