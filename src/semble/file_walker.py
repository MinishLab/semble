import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileType:
    """Language and indexing policy for a file extension."""

    language: str
    index_by_default: bool = True


FILE_TYPES: dict[str, FileType] = {
    ".py": FileType("python"),
    ".js": FileType("javascript"),
    ".jsx": FileType("javascript"),
    ".ts": FileType("typescript"),
    ".tsx": FileType("typescript"),
    ".go": FileType("go"),
    ".rs": FileType("rust"),
    ".java": FileType("java"),
    ".kt": FileType("kotlin"),
    ".kts": FileType("kotlin"),
    ".rb": FileType("ruby"),
    ".php": FileType("php"),
    ".c": FileType("c"),
    ".h": FileType("c"),
    ".cpp": FileType("cpp"),
    ".hpp": FileType("cpp"),
    ".cs": FileType("csharp"),
    ".swift": FileType("swift"),
    ".scala": FileType("scala"),
    ".sbt": FileType("scala"),
    ".dart": FileType("dart"),
    ".lua": FileType("lua"),
    ".sql": FileType("sql"),
    ".sh": FileType("bash"),
    ".md": FileType("markdown", index_by_default=False),
    ".yaml": FileType("yaml", index_by_default=False),
    ".yml": FileType("yaml", index_by_default=False),
    ".toml": FileType("toml", index_by_default=False),
    ".json": FileType("json", index_by_default=False),
}

DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        ".tox",
        "dist",
        "build",
        ".eggs",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".semble",
    }
)


def language_for_path(path: Path) -> str | None:
    """Return the language for a file path, or None for unknown extensions."""
    spec = FILE_TYPES.get(path.suffix.lower())
    return None if spec is None else spec.language


def resolve_extensions(extensions: frozenset[str] | None, *, include_docs: bool) -> frozenset[str]:
    """Return the set of file extensions to index."""
    if extensions is not None:
        return extensions
    return frozenset(ext for ext, spec in FILE_TYPES.items() if include_docs or spec.index_by_default)


def walk_files(root: Path, extensions: frozenset[str], ignore: frozenset[str] | None = None) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored directories."""
    ignore = DEFAULT_IGNORED_DIRS if ignore is None else ignore
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames[:] = sorted(d for d in dirnames if d not in ignore)
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in extensions:
                yield file_path
