import logging
import re
from bisect import bisect_left

from semble.chunking.core import chunk, chunk_lines
from semble.types import Chunk

logger = logging.getLogger(__name__)

# The desired length of chunks in chars.
# TODO: make this configurable
_DESIRED_CHUNK_LENGTH_CHARS = 750


def chunk_source(source: str, file_path: str, language: str | None) -> list[Chunk]:
    """Chunk pre-read source text."""
    if not source.strip():
        return []
    chunk_boundaries = None
    if language is not None:
        chunk_boundaries = chunk(source, language, _DESIRED_CHUNK_LENGTH_CHARS)
    # This is an if because the error state of the parser above
    # is a None.
    if chunk_boundaries is None:
        chunk_boundaries = chunk_lines(source, _DESIRED_CHUNK_LENGTH_CHARS)

    # Line numbers come from a binary search over newline offsets, so large files aren't rescanned per chunk.
    newline_offsets = [match.start() for match in re.finditer("\n", source)]
    chunks: list[Chunk] = []
    for boundary in chunk_boundaries:
        # Clamp to start_index so zero-length chunks don't produce an off-by-one.
        end_index = max(boundary.end - 1, boundary.start)
        text = source[boundary.start : end_index + 1]
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=bisect_left(newline_offsets, boundary.start) + 1,
                end_line=bisect_left(newline_offsets, end_index) + 1,
                language=language,
            )
        )
    return chunks
