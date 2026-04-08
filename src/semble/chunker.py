from __future__ import annotations

import hashlib
from pathlib import Path

from chonkie.chunker import CodeChunker

from semble.types import Chunk

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def chunk_by_lines(
    source: str,
    file_path: str,
    language: str | None = None,
    max_lines: int = 50,
    overlap_lines: int = 5,
) -> list[Chunk]:
    """Fallback chunker: split by line count with overlap.

    :param source: Source text to chunk.
    :param file_path: Path of the source file (for metadata).
    :param language: Language identifier (optional).
    :param max_lines: Maximum lines per chunk.
    :param overlap_lines: Lines of overlap between consecutive chunks.
    :returns: List of line-based chunks.
    """
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        content = "".join(lines[start:end])
        if content.strip():
            chunks.append(
                Chunk(
                    content=content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    language=language,
                    content_hash=_content_hash(content),
                )
            )
        start = end - overlap_lines if end < len(lines) else end

    return chunks


def chunk_source(source: str, file_path: str, language: str | None) -> list[Chunk]:
    """Chunk pre-read source text using Chonkie with line-based fallback.

    :param source: Source text to chunk.
    :param file_path: Path of the source file (for metadata).
    :param language: Language identifier, or None to use line-based chunking.
    :returns: List of chunks.
    """
    if not source.strip():
        return []
    if language:
        return _chunk_with_chonkie(source, file_path, language)
    return chunk_by_lines(source, file_path, language)


def chunk_file(file_path: Path) -> list[Chunk]:
    """Chunk a single file, reading it from disk.

    :param file_path: Path to the source file.
    :returns: List of chunks extracted from the file.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    language = EXTENSION_MAP.get(file_path.suffix.lower())
    return chunk_source(source, str(file_path), language)


def _chunk_with_chonkie(source: str, file_path: str, language: str) -> list[Chunk]:
    try:
        cc = CodeChunker(language=language, chunk_size=1500)
        raw = cc.chunk(source)
    except Exception:
        return chunk_by_lines(source, file_path, language)

    if not raw:
        return chunk_by_lines(source, file_path, language)

    # Convert character offsets to 1-based line numbers via cumulative positions
    cum: list[int] = [0]
    for line in source.splitlines(keepends=True):
        cum.append(cum[-1] + len(line))

    def _char_to_line(offset: int) -> int:
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cum[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    chunks: list[Chunk] = []
    for c in raw:
        text = c.text
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=_char_to_line(c.start_index),
                end_line=_char_to_line(max(c.end_index - 1, c.start_index)),
                language=language,
                content_hash=_content_hash(text),
            )
        )
    return chunks if chunks else chunk_by_lines(source, file_path, language)
