import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileTypeType(str, Enum):
    CODE = "CODE"
    DOCUMENT = "DOCUMENT"


@dataclass(frozen=True)
class FileType:
    """Language and indexing policy for a file extension."""

    language: str
    type: FileTypeType


FILE_TYPES: dict[str, FileType] = {
    ".py": FileType("python", FileTypeType.CODE),
    ".js": FileType("javascript", FileTypeType.CODE),
    ".jsx": FileType("javascript", FileTypeType.CODE),
    ".ts": FileType("typescript", FileTypeType.CODE),
    ".tsx": FileType("typescript", FileTypeType.CODE),
    ".go": FileType("go", FileTypeType.CODE),
    ".rs": FileType("rust", FileTypeType.CODE),
    ".java": FileType("java", FileTypeType.CODE),
    ".kt": FileType("kotlin", FileTypeType.CODE),
    ".kts": FileType("kotlin", FileTypeType.CODE),
    ".rb": FileType("ruby", FileTypeType.CODE),
    ".php": FileType("php", FileTypeType.CODE),
    ".c": FileType("c", FileTypeType.CODE),
    ".h": FileType("c", FileTypeType.CODE),
    ".cpp": FileType("cpp", FileTypeType.CODE),
    ".hpp": FileType("cpp", FileTypeType.CODE),
    ".cs": FileType("csharp", FileTypeType.CODE),
    ".swift": FileType("swift", FileTypeType.CODE),
    ".scala": FileType("scala", FileTypeType.CODE),
    ".sbt": FileType("scala", FileTypeType.CODE),
    ".ex": FileType("elixir", FileTypeType.CODE),
    ".exs": FileType("elixir", FileTypeType.CODE),
    ".dart": FileType("dart", FileTypeType.CODE),
    ".lua": FileType("lua", FileTypeType.CODE),
    ".sql": FileType("sql", FileTypeType.CODE),
    ".sh": FileType("bash", FileTypeType.CODE),
    ".bash": FileType("bash", FileTypeType.CODE),
    ".zig": FileType("zig", FileTypeType.CODE),
    ".hs": FileType("haskell", FileTypeType.CODE),
    ".md": FileType("markdown", FileTypeType.DOCUMENT),
    ".yaml": FileType("yaml", FileTypeType.DOCUMENT),
    ".yml": FileType("yaml", FileTypeType.DOCUMENT),
    ".toml": FileType("toml", FileTypeType.DOCUMENT),
    ".json": FileType("json", FileTypeType.DOCUMENT),
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
    if spec := FILE_TYPES.get(path.suffix.lower()):
        return spec.language
    return None


def filter_extensions(extensions: frozenset[str] | None, *, include_docs: bool) -> frozenset[str]:
    """Return the set of file extensions to index."""
    if extensions is not None:
        return extensions
    # Always index code files
    types_to_include = {FileTypeType.CODE}
    if include_docs:
        types_to_include.add(FileTypeType.DOCUMENT)
    # Return a default set of extensions
    return frozenset(ext for ext, spec in FILE_TYPES.items() if spec.type in types_to_include)


def walk_files(root: Path, extensions: frozenset[str], ignore: frozenset[str] | None = None) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored directories."""
    # Always skip the defaults.
    ignore = (ignore or frozenset()) | DEFAULT_IGNORED_DIRS
    for dirpath, _, filenames in os.walk(root):
        dirpath_as_path = Path(dirpath)
        if set(dirpath_as_path.parts) & ignore:
            continue
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in extensions:
                yield file_path
