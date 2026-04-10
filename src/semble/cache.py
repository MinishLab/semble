import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np

from semble.types import EmbeddingMatrix


class EmbeddingCache:
    """Embedding cache combining an in-memory dict with optional disk storage."""

    def __init__(
        self,
        memory: dict[str, EmbeddingMatrix],
        cache_dir: Path | None,
        cache_namespace: str | None,
    ) -> None:
        """Initialize the cache."""
        self._memory = memory
        safe = cache_namespace.replace("/", "--").replace("..", "__") if cache_namespace else None
        self._root = cache_dir / safe if cache_dir and safe else None

    def get(self, key: str) -> EmbeddingMatrix | None:
        """Return the embedding for key, promoting a disk hit to memory. None on miss."""
        if key in self._memory:
            return self._memory[key]
        if self._root is None:
            return None
        try:
            embedding = np.load(self._root / key[:2] / f"{key}.npy", allow_pickle=False)
        except (FileNotFoundError, ValueError, OSError):
            return None
        self._memory[key] = embedding
        return embedding

    def put(self, key: str, embedding: EmbeddingMatrix) -> None:
        """Store embedding in memory and atomically write to disk if caching is enabled."""
        self._memory[key] = embedding
        if self._root is None:
            return
        path = self._root / key[:2] / f"{key}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, embedding, allow_pickle=False)
            os.replace(tmp, path)
        finally:
            # No-op on success (tmp was renamed); cleans up on any failure.
            with contextlib.suppress(OSError):
                os.unlink(tmp)
