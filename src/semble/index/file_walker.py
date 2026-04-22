import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileCategory(str, Enum):
    CODE = "CODE"
    DOCUMENT = "DOCUMENT"


@dataclass(frozen=True)
class FileType:
    """Language and indexing policy for a file extension."""

    language: str
    category: FileCategory


FILE_TYPES: dict[str, FileType] = {
    ".py": FileType("python", FileCategory.CODE),
    ".js": FileType("javascript", FileCategory.CODE),
    ".jsx": FileType("javascript", FileCategory.CODE),
    ".ts": FileType("typescript", FileCategory.CODE),
    ".tsx": FileType("typescript", FileCategory.CODE),
    ".go": FileType("go", FileCategory.CODE),
    ".rs": FileType("rust", FileCategory.CODE),
    ".java": FileType("java", FileCategory.CODE),
    ".kt": FileType("kotlin", FileCategory.CODE),
    ".kts": FileType("kotlin", FileCategory.CODE),
    ".rb": FileType("ruby", FileCategory.CODE),
    ".php": FileType("php", FileCategory.CODE),
    ".c": FileType("c", FileCategory.CODE),
    ".h": FileType("c", FileCategory.CODE),
    ".cpp": FileType("cpp", FileCategory.CODE),
    ".hpp": FileType("cpp", FileCategory.CODE),
    ".cs": FileType("csharp", FileCategory.CODE),
    ".swift": FileType("swift", FileCategory.CODE),
    ".scala": FileType("scala", FileCategory.CODE),
    ".sbt": FileType("scala", FileCategory.CODE),
    ".ex": FileType("elixir", FileCategory.CODE),
    ".exs": FileType("elixir", FileCategory.CODE),
    ".dart": FileType("dart", FileCategory.CODE),
    ".lua": FileType("lua", FileCategory.CODE),
    ".sql": FileType("sql", FileCategory.CODE),
    ".sh": FileType("bash", FileCategory.CODE),
    ".bash": FileType("bash", FileCategory.CODE),
    ".zig": FileType("zig", FileCategory.CODE),
    ".hs": FileType("haskell", FileCategory.CODE),
    ".md": FileType("markdown", FileCategory.DOCUMENT),
    ".yaml": FileType("yaml", FileCategory.DOCUMENT),
    ".yml": FileType("yaml", FileCategory.DOCUMENT),
    ".toml": FileType("toml", FileCategory.DOCUMENT),
    ".json": FileType("json", FileCategory.DOCUMENT),
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


def filter_extensions(extensions: frozenset[str] | None, *, include_text_files: bool) -> frozenset[str]:
    """Return the set of file extensions to index."""
    if extensions is not None:
        return extensions
    # Always index code files
    categories_to_include = {FileCategory.CODE}
    if include_text_files:
        categories_to_include.add(FileCategory.DOCUMENT)
    # Return a default set of extensions
    return frozenset(ext for ext, spec in FILE_TYPES.items() if spec.category in categories_to_include)


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
