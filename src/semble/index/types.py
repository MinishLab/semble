from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersistencePath:
    """Simple model so that the save/load roundtrip is typed."""

    chunks: Path
    chunk_store: Path
    bm25_index: Path
    semantic_index: Path
    metadata: Path
    chunk_cache: Path

    def non_existing(self) -> list[Path]:
        """Return required paths that do not exist."""
        missing = [path for path in [self.bm25_index, self.semantic_index, self.metadata] if not path.exists()]
        if not self.chunks.exists() and not self.chunk_store.exists():
            missing.extend([self.chunks, self.chunk_store])
        return missing

    @classmethod
    def from_path(cls: type[PersistencePath], path: Path) -> PersistencePath:
        """Create a PersistencePath from a base path."""
        return PersistencePath(
            chunks=path / "chunks.json",
            chunk_store=path / "chunks.lmdb",
            bm25_index=path / "bm25_index",
            semantic_index=path / "semantic_index",
            metadata=path / "metadata.json",
            chunk_cache=path / "chunk_cache.json",
        )
