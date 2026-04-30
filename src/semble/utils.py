from __future__ import annotations

import re

from semble.types import Chunk, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def _resolve_chunk(chunks: list[Chunk], file_path: str, line: int) -> Chunk | None:
    """Return the chunk that contains *line* in *file_path*, or None.

    MCP tool arguments are JSON primitives (strings and ints), so the agent
    passes file_path + line rather than a Chunk object. This function
    reconstructs the Chunk at the MCP boundary before calling into the library.

    :param chunks: All indexed chunks to search.
    :param file_path: File path as stored in the index.
    :param line: 1-indexed line number to resolve.
    :return: The best-matching Chunk, or None if not found.
    """
    fallback = None
    for chunk in chunks:
        if chunk.file_path == file_path and chunk.start_line <= line <= chunk.end_line:
            if line < chunk.end_line:
                return chunk
            if fallback is None:  # line == end_line: boundary; keep as fallback for end-of-file chunks
                fallback = chunk
    return fallback


def _format_results(header: str, results: list[SearchResult]) -> str:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        lines.append("```")
        lines.append(r.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
