from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class CallType(str, Enum):
    """Call type for token-savings tracking."""

    SEARCH = "search"
    FIND_RELATED = "find_related"


class Encoder(Protocol):
    """Protocol for embedding models."""

    @property
    def dim(self) -> int:
        """The dimensionality of the embedding."""
        ...

    def encode(self, texts: Sequence[str], /, **kwargs: Any) -> EmbeddingMatrix:
        """Encode texts into embeddings as a 2D float32 array."""
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


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Statistics about the current index state."""

    indexed_files: int = 0
    total_chunks: int = 0
    languages: dict[str, int] = field(default_factory=dict)
