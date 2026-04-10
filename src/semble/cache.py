import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semble.types import EmbeddingMatrix


@dataclass(frozen=True, slots=True)
class _CacheSpec:
    """Disk-cache location and model namespace derived from a root path and model ID."""

    root: Path
    model_id: str

    @property
    def namespace(self) -> str:
        """Return a filesystem-safe directory name for the model ID.

        :return: Model ID with / replaced by --.
        """
        return self.model_id.replace("/", "--")

    def path_for(self, content_hash: str) -> Path:
        """Return the per-embedding file path for content_hash.

        :param content_hash: Hash of the chunk content used as the cache key.
        :return: Absolute path to the .npy file for this embedding.
        """
        return self.root / self.namespace / content_hash[:2] / f"{content_hash}.npy"


class _EmbeddingCache:
    """Embedding cache combining an in-memory dict with optional disk storage."""

    def __init__(
        self,
        memory: dict[str, EmbeddingMatrix],
        spec: _CacheSpec | None,
    ) -> None:
        """Initialise the cache.

        :param memory: Shared in-memory dict.
        :param spec: Disk-cache specification, or None to disable disk persistence.
        """
        self._memory = memory
        self._spec = spec

    def get(self, key: str) -> EmbeddingMatrix | None:
        """Return the embedding for a key, or None on a full miss.

        A disk hit is promoted to memory before returning.

        :param key: Content hash of the chunk.
        :return: Cached embedding array, or None if not found.
        """
        if key in self._memory:
            return self._memory[key]
        if self._spec is None:
            return None
        try:
            embedding = np.load(self._spec.path_for(key), allow_pickle=False)
        except (FileNotFoundError, ValueError, OSError):
            return None
        self._memory[key] = embedding
        return embedding

    def put(self, key: str, embedding: EmbeddingMatrix) -> None:
        """Store embedding under key in both memory and (optionally) disk.

        Disk writes use an atomic temp-file rename so concurrent processes
        never read a partial file.

        :param key: Content hash of the chunk.
        :param embedding: Embedding array to store.
        """
        self._memory[key] = embedding
        if self._spec is None:
            return
        path = self._spec.path_for(key)
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


def make_embedding_cache(
    memory: dict[str, EmbeddingMatrix],
    cache_dir: Path | None,
    model_id: str | None,
) -> _EmbeddingCache:
    """Build an _EmbeddingCache with the given shared memory and optional disk cache spec.

    :param memory: Shared in-memory embedding dict.
    :param cache_dir: Resolved (already expanded) root path for disk storage, or None.
    :param model_id: Model identifier used as the cache namespace, or None.
    :return: A configured _EmbeddingCache instance.
    """
    spec = _CacheSpec(cache_dir, model_id) if cache_dir is not None and model_id is not None else None
    return _EmbeddingCache(memory, spec)
