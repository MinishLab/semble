import logging
from functools import cache
from typing import Literal

from magika import Magika

from semble.chunking.chunk_machine import chunk, chunk_lines, is_supported_language
from semble.types import Chunk

logger = logging.getLogger(__name__)

_DESIRED_LENGTH = 1500


@cache
def _get_magika_instance() -> Magika:
    return Magika()


def predict_language(source: bytes) -> str:
    """Predict the language of a document byte source."""
    magika = _get_magika_instance()
    result = magika.identify_bytes(source)
    return result.output.label


def chunk_source(source: str, file_path: str, language: str | None | Literal["auto"]) -> list[Chunk]:
    """Chunk pre-read source text."""
    if not source.strip():
        return []
    if language == "auto":
        language = predict_language(source.encode())
    if is_supported_language(language):
        # None is not a supported language.
        assert language is not None
        try:
            chunk_boundaries = chunk(source, language, _DESIRED_LENGTH)
        except Exception:
            logger.error("Chunking failed for language %r, falling back to line chunking", language, exc_info=True)
            chunk_boundaries = chunk_lines(source, _DESIRED_LENGTH)
    else:
        chunk_boundaries = chunk_lines(source, _DESIRED_LENGTH)

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
