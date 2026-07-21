from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semble.index.bm25 import BM25
from semble.types import Chunk, EmbeddingMatrix

CACHE_FORMAT_VERSION = 1  # Bump when the persisted index schema changes.


def make_chunk_id(indexed_path: str, slot: int) -> str:
    """Return the stable document ID for a file chunk."""
    return f"{indexed_path}:{slot}"


@dataclass
class PersistencePath:
    """Simple model so that the save/load roundtrip is typed."""

    chunks: Path
    bm25_index: Path
    semantic_index: Path
    metadata: Path

    def non_existing(self) -> list[Path]:
        """Return all resolved that do not exist."""
        return [
            path for path in [self.chunks, self.bm25_index, self.semantic_index, self.metadata] if not path.exists()
        ]

    @classmethod
    def from_path(cls: type[PersistencePath], path: Path) -> PersistencePath:
        """Create a PersistencePath from a base path."""
        return PersistencePath(
            chunks=path / "chunks.json",
            bm25_index=path / "bm25_index",
            semantic_index=path / "semantic_index",
            metadata=path / "metadata.json",
        )


@dataclass
class FileManifestEntry:
    """Record a file's modification time and chunk range within the global chunk list."""

    mtime_ns: int
    start: int
    count: int

    @property
    def end(self) -> int:
        """Return the exclusive end of the chunk range."""
        return self.start + self.count


@dataclass
class PreviousIndex:
    """A previously built index, loaded for reuse during incremental reindexing."""

    chunks: list[Chunk]
    vectors: EmbeddingMatrix
    manifest: dict[str, FileManifestEntry]
    bm25_index: BM25
