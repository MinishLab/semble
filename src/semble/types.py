from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

EmbeddingMatrix = npt.NDArray[np.float32]


class SearchMode(str, Enum):
    """Search mode for SembleIndex.search()."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"


class Encoder(Protocol):
    """Protocol for embedding models."""

    def encode(self, texts: Sequence[str], /) -> npt.NDArray[Any]:
        """Encode texts into embeddings."""
        ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single indexable unit of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None

    @property
    def location(self) -> str:
        """File path and line range as a string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with score and source."""

    chunk: Chunk
    score: float
    source: SearchMode

    def __str__(self) -> str:
        """Return a human-readable summary of the result."""
        header = f"{self.chunk.location}  score={self.score:.3f}"
        separator = "-" * len(header)
        return f"{header}\n{separator}\n{self.chunk.content.strip()}\n"


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics about the current index state."""

    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
