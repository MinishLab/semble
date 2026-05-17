from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec


@dataclass(frozen=True)
class IgnoreSpec:
    base: Path
    spec: GitIgnoreSpec


_DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git/",
        ".hg/",
        ".svn/",
        "__pycache__/",
        "node_modules/",
        ".venv/",
        "venv/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".cache/",
        ".semble/",
        ".next/",
        "dist/",
        "build/",
        ".eggs/",
    }
)


def _load_ignore_for_dir(directory: Path) -> GitIgnoreSpec | None:
    """Loads a gitignore and sembleignore for a dir."""
    gitignore = directory / ".gitignore"
    sembleignore = directory / ".sembleignore"

    lines = []
    if gitignore.is_file():
        lines.extend(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if sembleignore.is_file():
        lines.extend(sembleignore.read_text(encoding="utf-8", errors="ignore").splitlines())
    if lines:
        return GitIgnoreSpec.from_lines(lines)
    return None


def walk_files(root: Path, extensions: Sequence[str], ignore: Sequence[str] | None = None) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored paths.

    Directories matching DEFAULT_IGNORED_DIRS plus any names in ignore are always
    skipped. If the root contains a .gitignore, its patterns are also honoured.

    :param root: Root directory to walk.
    :param extensions: List of file extensions to match.
    :param ignore: Additional patterns to ignore.
    :yield: Path to each file under root matching the criteria.
    :ytype: Path
    """
    # This should be a list. Traversal is done in order, so the order matters.
    ignored = []
    extensions_as_patterns = [f"!*{ext}" for ext in extensions]
    ignored.extend(extensions_as_patterns)
    ignored.extend(_DEFAULT_IGNORED_DIRS)
    # Always give user patterns preference
    ignored.extend(ignore or [])
    base_spec = GitIgnoreSpec.from_lines(ignored, backend="simple")
    s = IgnoreSpec(base=root, spec=base_spec)
    yield from _walk(root, [s])


def _is_ignored(path: Path, specs: list[IgnoreSpec]) -> bool:
    """Check if a path is ignored by any of the provided ignore specs."""
    is_dir = path.is_dir()
    # Everything starts off as unignored
    ignored = not is_dir

    for ignore_spec in specs:
        try:
            # If there is no relative path, this is invalid.
            relative = path.relative_to(ignore_spec.base)
        except ValueError:
            continue

        relative_str = relative.as_posix()
        # We need to add a trailing slash. Gitignore
        # matches dirs as trailing '/'.
        if is_dir:
            relative_str += "/"

        # Loop over all the patterns
        for pattern in ignore_spec.spec.patterns:
            # This pattern doesn't do anything.
            if pattern.include is None:
                continue

            if pattern.match_file(relative_str) is not None:
                ignored = pattern.include

    return ignored


def _walk(
    directory: Path,
    inherited_specs: list[IgnoreSpec],
) -> Iterator[Path]:
    """Recursive function for walking files under a directory."""
    active_specs = inherited_specs

    spec = _load_ignore_for_dir(directory)
    if spec is not None:
        active_specs = [
            *inherited_specs,
            IgnoreSpec(base=directory, spec=spec),
        ]

    for item in directory.iterdir():
        # Don't follow symlinks
        if item.is_symlink():
            continue
        if _is_ignored(item, active_specs):
            continue

        if item.is_dir():
            yield from _walk(item, active_specs)
        elif item.is_file():
            yield item
