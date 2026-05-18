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
    extensions_set = frozenset(extensions)
    dir_patterns = list(sorted(_DEFAULT_IGNORED_DIRS)) + list(ignore or [])
    base_spec = GitIgnoreSpec.from_lines(dir_patterns, backend="simple")
    s = IgnoreSpec(base=root, spec=base_spec)
    yield from _walk(root, [s], extensions_set)


def _is_ignored(path: Path, specs: list[IgnoreSpec]) -> tuple[bool, bool]:
    """Check if a path is ignored by any of the provided ignore specs."""
    is_dir = path.is_dir()
    ignored = False
    found = False
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
                # Bypass extension filter only for negation patterns with a file
                # extension suffix (e.g. !special.kjs, !*.py). Patterns without
                # a suffix (e.g. !vendor/, !.github/*) target directories or
                # broad globs and should not bypass extension filtering.
                pat = pattern.pattern or ""
                found = not ignored and bool(Path(pat.rstrip("/")).suffix)

    return ignored, found


def _walk(
    directory: Path,
    inherited_specs: list[IgnoreSpec],
    extensions: frozenset[str],
) -> Iterator[Path]:
    """Recursive function for walking files under a directory."""
    spec = _load_ignore_for_dir(directory)
    if spec is not None:
        inherited_specs = [
            *inherited_specs,
            IgnoreSpec(base=directory, spec=spec),
        ]

    for item in sorted(directory.iterdir()):
        # Don't follow symlinks
        if item.is_symlink():
            continue
        is_ignored, found = _is_ignored(item, inherited_specs)
        if is_ignored:
            continue

        if item.is_dir():
            yield from _walk(item, inherited_specs, extensions)
        elif item.is_file() and (found or item.suffix.lower() in extensions):
            yield item
