"""Core data types for semble."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SymbolKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    CHUNK = "chunk"  # fallback: raw text chunk, not AST-extracted


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    symbol_kind: SymbolKind = SymbolKind.CHUNK
    language: str | None = None
    content_hash: str = ""

    @property
    def location(self) -> str:
        name = self.symbol_name or "chunk"
        return f"{self.file_path}:{self.start_line}-{self.end_line} ({name})"


@dataclass
class SearchResult:
    """A single search result with score and source."""

    chunk: Chunk
    score: float
    source: str  # "semantic", "bm25", "symbol", "hybrid"


@dataclass
class IndexStats:
    """Statistics about the current index state."""

    total_files: int = 0
    total_chunks: int = 0
    total_symbols: int = 0
    index_time_ms: float = 0
    embedding_time_ms: float = 0
    languages: dict[str, int] = field(default_factory=dict)
