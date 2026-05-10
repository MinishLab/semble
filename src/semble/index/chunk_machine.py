from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from logging import getLogger

from tree_sitter import Parser
from tree_sitter_language_pack import SupportedLanguage, get_parser

logger = getLogger(__name__)


@dataclass
class ChunkBoundary:
    start: int
    end: int


@cache
def _cached_get_parser(language: SupportedLanguage) -> Parser:
    """Gets a parser from tree_sitter."""
    return get_parser(language)


class Chunker:
    def __init__(
        self,
        language: str,
        desired_size: int,
    ) -> None:
        """Initialize the chunker."""
        self.language = language
        self.desired_size = desired_size

    def chunk(self, text: str) -> list[ChunkBoundary]:
        """Chunk source code."""
        if not text.strip():
            return []

        as_bytes = text.encode("utf-8")
        parser = _cached_get_parser(self.language)
        root = parser.parse(as_bytes).root_node

        chunks = []
        for chunk in root.children:
            start = chunk.start_byte
            end = chunk.end_byte
            start_char = len(as_bytes[:start].decode("utf-8"))
            end_char = len(as_bytes[:end].decode("utf-8"))
            chunks.append(ChunkBoundary(start=start_char, end=end_char))

        return chunks
