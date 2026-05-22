from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

from semble.types import Chunk, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")


def is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def find_index_from_cache_folder(path: Path) -> Path:
    """Finds an index from a cache folder and a project path."""
    normalized = path.expanduser().resolve()
    data = str(normalized).encode("utf-8")
    subdir_path = hashlib.new("sha256", data).hexdigest()
    cache_dir = resolve_cache_folder() / subdir_path
    return cache_dir / "index"


def resolve_cache_folder() -> Path:
    """Resolves a cache folder, respects XDG_CACHE_HOME."""
    name = "semble"
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base is None:
            base = Path.home() / "AppData" / "Local"
        else:
            base = Path(base)
        cache_dir = base / name / "Cache"
    elif sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches" / name
    else:
        base = os.getenv("XDG_CACHE_HOME")
        if base:
            cache_dir = Path(base) / name
        else:
            cache_dir = Path.home() / ".cache" / name

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_chunk(chunks: list[Chunk], file_path: str, line: int) -> Chunk | None:
    """Return the chunk containing *line* in *file_path*, or None.

    Reconstructs a Chunk from its JSON-primitive MCP tool arguments (file_path + line)
    before calling into the library.
    """
    fallback = None
    for chunk in chunks:
        if chunk.file_path == file_path and chunk.start_line <= line <= chunk.end_line:
            if line < chunk.end_line:
                return chunk
            if fallback is None:  # line == end_line: boundary; keep as fallback for end-of-file chunks
                fallback = chunk
    return fallback


def format_results(header: str, results: list[SearchResult]) -> str:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        lines.append("```")
        lines.append(r.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
