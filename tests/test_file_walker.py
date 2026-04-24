from pathlib import Path

from semble.index.file_walker import walk_files


def _touch(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_walk_files_skips_default_ignored_dirs(tmp_path: Path) -> None:
    """Files under ``.venv``, ``node_modules``, ``.cache`` etc. are not yielded."""
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / ".venv" / "lib" / "b.py")
    _touch(tmp_path / "node_modules" / "pkg" / "c.js")
    _touch(tmp_path / ".cache" / "uv" / "d.py")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, frozenset({".py", ".js"}))}
    assert found == {"src/a.py"}


def test_walk_files_respects_root_gitignore(tmp_path: Path) -> None:
    """Patterns in a root ``.gitignore`` exclude both directories and files."""
    _touch(tmp_path / "src" / "keep.py")
    _touch(tmp_path / "local" / "ignored.py")
    _touch(tmp_path / "generated.py")
    (tmp_path / ".gitignore").write_text("local/\ngenerated.py\n")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, frozenset({".py"}))}
    assert found == {"src/keep.py"}


def test_walk_files_gitignore_negation(tmp_path: Path) -> None:
    """``!`` negation patterns re-include previously ignored files."""
    _touch(tmp_path / "build" / "out.py")
    _touch(tmp_path / "build" / "keep.py")
    (tmp_path / ".gitignore").write_text("build/*\n!build/keep.py\n")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, frozenset({".py"}))}
    assert found == {"build/keep.py"}


def test_walk_files_prunes_ignored_dirs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Ignored directories are pruned so ``os.walk`` never descends into them."""
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "node_modules" / "deep" / "deeper" / "b.js")

    visited: list[str] = []
    import os

    real_walk = os.walk

    def tracking_walk(*args, **kwargs):  # type: ignore[no-untyped-def]
        for dirpath, dirnames, filenames in real_walk(*args, **kwargs):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr("semble.index.file_walker.os.walk", tracking_walk)
    list(walk_files(tmp_path, frozenset({".py", ".js"})))
    assert not any("node_modules" in v for v in visited[1:]), visited
