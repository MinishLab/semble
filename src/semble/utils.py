from __future__ import annotations

import os
import re
from typing import Any

from semble.types import Chunk, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")
DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"


def is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


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


def format_results(query: str, results: list[SearchResult]) -> dict[str, Any]:
    """Render SearchResult objects as a JSONable object."""
    return {"query": query, "results": [r.to_dict() for r in results]}


def format_results_snippet(query: str, results: list[SearchResult], snippet_lines: int | None) -> dict[str, Any]:
    """Render results, optionally truncating chunk content to the first snippet_lines lines.

    snippet_lines=None → full content (same as format_results).
    snippet_lines=0    → no content, only file path and line range.
    snippet_lines=N>0  → first N lines (function signature without body).
    """
    if snippet_lines is None:
        return format_results(query, results)
    formatted = []
    for r in results:
        entry: dict[str, Any] = {
            "file_path": r.chunk.file_path,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "score": r.score,
        }
        if snippet_lines > 0:
            lines = r.chunk.content.splitlines()
            entry["snippet"] = "\n".join(lines[:snippet_lines])
        formatted.append(entry)
    return {"query": query, "results": formatted}


def resolve_model_name() -> str:
    """Resolve a model name to a configurable."""
    return os.environ.get("SEMBLE_MODEL_NAME", DEFAULT_MODEL_NAME)
