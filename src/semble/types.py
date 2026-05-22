from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class CallType(str, Enum):
    """Call type for token-savings tracking."""

    SEARCH = "search"
    FIND_RELATED = "find_related"


class ContentType(str, Enum):
    """Content type for indexing and search pipeline selection."""

    CODE = "code"
    DOCS = "docs"
    CONFIG = "config"


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

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[Chunk], data: dict[str, Any]) -> Chunk:
        """Create a Chunk from a dict."""
        return cls(**data)


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
