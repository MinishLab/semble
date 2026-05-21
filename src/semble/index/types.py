from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersistencePath:
    """Simple model so that the save/load roundtrip is typed."""

    chunks: Path
    bm25_index: Path
    semantic_index: Path
    metadata: Path

    @classmethod
    def from_path(cls: type[PersistencePath], path: Path) -> PersistencePath:
        """Create a PersistencePath from a base path."""
        return PersistencePath(
            chunks=path / "chunks.json",
            bm25_index=path / "bm25_index",
            semantic_index=path / "semantic_index",
            metadata=path / "metadata.json",
        )
