from pathlib import Path

import pytest

from semble.index.file_walker import walk_files


def _touch(path: Path, content: str = "x = 1\n") -> None:
    """Create path (and any missing parents) and write content to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.mark.parametrize(
    ("files", "gitignore", "sembleignore", "expected"),
    [
        # Default-ignored dirs (.venv, node_modules, .cache) are always skipped.
        (
            ["src/a.py", ".venv/lib/b.py", "node_modules/pkg/c.py", ".cache/uv/d.py"],
            None,
            None,
            {"src/a.py"},
        ),
        # Root .gitignore excludes both directories and files.
        (
            ["src/keep.py", "local/ignored.py", "generated.py"],
            "local/\ngenerated.py\n# comment",
            None,
            {"src/keep.py"},
        ),
        # Negation (`!`) patterns re-include previously ignored files.
        (
            ["out/a.py", "out/keep.py"],
            "out/*\n!out/keep.py\n",
            None,
            {"out/keep.py"},
        ),
        # Allow-list style gitignore (`*` + `!*/` + `!*.py`) must not prune subdirs.
        (
            ["main.py", "internal/pkg/foo.py", "internal/pkg/bar.py"],
            "*\n!*/\n!*.py\n",
            None,
            {"main.py", "internal/pkg/foo.py", "internal/pkg/bar.py"},
        ),
        # Ignored-parent negation: out/* prunes out/deep/, so out/deep/keep.py must not leak.
        (
            ["out/deep/keep.py"],
            "out/*\n!out/deep/keep.py\n",
            None,
            set(),
        ),
        # Ignored-parent negation: out/* prunes out/deep/, so out/deep/keep.py must not leak.
        (
            ["out/deep/keep.py"],
            None,
            "out/*\n!out/deep/keep.py\n",
            set(),
        ),
    ],
)
def test_walk_files_filtering(
    tmp_path: Path, files: list[str], gitignore: str | None, sembleignore: str | None, expected: set[str]
) -> None:
    """Directory defaults, gitignore patterns, and negations filter the yielded files."""
    for rel in files:
        _touch(tmp_path / rel)
    if gitignore is not None:
        (tmp_path / ".gitignore").write_text(gitignore)
    if sembleignore is not None:
        (tmp_path / ".sembleignore").write_text(sembleignore)

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, [".py"])}
    assert found == expected


def test_walk_files_prunes_ignored_dirs(tmp_path: Path) -> None:
    """Ignored directories are pruned so os.walk never descends into them."""
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "node_modules" / "deep" / "deeper" / "b.js")

    visited = list(walk_files(tmp_path, [".py", ".js"]))
    assert not any("node_modules" in str(v) for v in visited), visited


def test_is_ignored_skips_spec_with_unrelated_base(tmp_path: Path) -> None:
    """An IgnoreSpec whose base is not an ancestor of the path is silently skipped.

    Files default to ignored in _is_ignored.  When the first spec has an
    unrelated base, the ValueError is caught and the spec is skipped without
    crashing.  A second spec with the correct base can still un-ignore the file.
    """
    from pathspec import GitIgnoreSpec

    from semble.index.file_walker import IgnoreSpec, _is_ignored

    # Create two unrelated directory trees
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    target_file = project_a / "keep.py"
    target_file.write_text("x = 1\n")

    # Spec rooted at project_b — unrelated to target_file
    unrelated_spec = IgnoreSpec(
        base=project_b,
        spec=GitIgnoreSpec.from_lines(["*.py"]),
    )

    # With only the unrelated spec the file stays in its default state (ignored)
    # and, crucially, no exception is raised.
    assert _is_ignored(target_file, [unrelated_spec]) is True

    # Spec rooted at project_a that un-ignores .py files
    matching_spec = IgnoreSpec(
        base=project_a,
        spec=GitIgnoreSpec.from_lines(["!*.py"]),
    )

    # The unrelated spec is safely skipped; the matching spec un-ignores the file.
    assert _is_ignored(target_file, [unrelated_spec, matching_spec]) is False


def test_walk_files_skips_symlinks(tmp_path: Path) -> None:
    """Symlinked files and directories are skipped; real paths are still walked."""
    # Real directory with a file
    real_dir = tmp_path / "real_pkg" / "src"
    _touch(real_dir / "mod.py")

    # A symlink to that directory from another location
    link_parent = tmp_path / "wrapper" / "src"
    link_parent.mkdir(parents=True)
    (link_parent / "linked").symlink_to(real_dir)

    # A symlink to a single file
    _touch(tmp_path / "original.py")
    (tmp_path / "link_to_original.py").symlink_to(tmp_path / "original.py")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, [".py"])}

    # Real paths are present
    assert "real_pkg/src/mod.py" in found
    assert "original.py" in found

    # Symlink-based paths are absent
    assert "wrapper/src/linked/mod.py" not in found
    assert "link_to_original.py" not in found
