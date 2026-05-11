import logging

from semble.chunking.core import chunk, chunk_lines, is_supported_language
from semble.types import Chunk

logger = logging.getLogger(__name__)

# The desired length of chunks in chars.
# TODO: makes this configurable
_DESIRED_CHUNK_LENGTH_CHARS = 1500


def chunk_source(source: str, file_path: str, language: str | None) -> list[Chunk]:
    """Chunk pre-read source text."""
    if not source.strip():
        return []
    if language is not None and is_supported_language(language):
        chunk_boundaries = chunk(source, language, _DESIRED_CHUNK_LENGTH_CHARS)
    else:
        chunk_boundaries = chunk_lines(source, _DESIRED_CHUNK_LENGTH_CHARS)

    chunks: list[Chunk] = []
    for boundary in chunk_boundaries:
        # Clamp to start_index so zero-length chunks don't produce an off-by-one.
        end_index = max(boundary.end - 1, boundary.start)
        text = source[boundary.start : end_index + 1]
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=source[: boundary.start].count("\n") + 1,
                end_line=source[:end_index].count("\n") + 1,
                language=language,
            )
        )
    return chunks
