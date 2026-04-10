import contextlib
import os
import tempfile
from pathlib import Path

import numpy as np

from semble.types import EmbeddingMatrix


class _EmbeddingCache:
    """Embedding cache combining an in-memory dict with optional disk storage."""

    def __init__(
        self,
        memory: dict[str, EmbeddingMatrix],
        cache_dir: Path | None,
        cache_namespace: str | None,
    ) -> None:
        self._memory = memory
        self._root = cache_dir / cache_namespace.replace("/", "--") if cache_dir and cache_namespace else None

    def _path_for(self, key: str) -> Path:
        assert self._root is not None
        return self._root / key[:2] / f"{key}.npy"

    def get(self, key: str) -> EmbeddingMatrix | None:
        if key in self._memory:
            return self._memory[key]
        if self._root is None:
            return None
        try:
            embedding = np.load(self._path_for(key), allow_pickle=False)
        except (FileNotFoundError, ValueError, OSError):
            return None
        self._memory[key] = embedding
        return embedding

    def put(self, key: str, embedding: EmbeddingMatrix) -> None:
        self._memory[key] = embedding
        if self._root is None:
            return
        path = self._path_for(key)
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
