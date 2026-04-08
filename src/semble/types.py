from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class SearchMode(str, Enum):
    """Search mode for SembleIndex.search()."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"
    SYMBOL = "symbol"


class Encoder(Protocol):
    """Protocol for embedding models. Any object with a compatible encode() works."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> npt.NDArray[np.float32]:
        """Encode a sequence of texts into embeddings."""
        ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None
    content_hash: str = ""

    @property
    def location(self) -> str:
        """Return the file path and line range for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


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
    index_time_ms: float = 0
    embedding_time_ms: float = 0
    languages: dict[str, int] = field(default_factory=dict)
