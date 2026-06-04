from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from semble.index.dense import SelectableBasicBackend


class SemanticBackend(Protocol):
    """Semantic index interface used by search and persistence paths."""

    @property
    def vectors(self) -> npt.NDArray[np.float32]:
        """Expose dense vectors for cache reuse paths."""
        ...

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]]:
        """Return nearest semantic IDs and distances for each query vector."""
        ...

    def save(self, path: Path) -> None:
        """Persist the semantic backend."""
        ...


class StableIdSemanticBackend:
    """Map persisted dense row positions back to stable chunk IDs."""

    def __init__(self, backend: SelectableBasicBackend, chunk_ids: Sequence[int]) -> None:
        """Wrap a row-position backend with stable chunk-id outputs."""
        normalized_chunk_ids = tuple(int(chunk_id) for chunk_id in chunk_ids)
        if len(normalized_chunk_ids) != len(backend.vectors):
            raise ValueError("Chunk IDs must match dense backend row count")
        if len(set(normalized_chunk_ids)) != len(normalized_chunk_ids):
            raise ValueError("Chunk IDs must be unique")

        self._backend = backend
        self._chunk_ids = np.array(normalized_chunk_ids, dtype=np.int_)
        self._rows_by_id = {chunk_id: row for row, chunk_id in enumerate(normalized_chunk_ids)}

    @property
    def vectors(self) -> npt.NDArray[np.float32]:
        """Expose underlying vectors for cache reuse paths."""
        return self._backend.vectors

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]]:
        """Run dense query with stable chunk-id selectors and results."""
        row_selector = None
        if selector is not None:
            row_selector = np.array(
                [self._rows_by_id[int(chunk_id)] for chunk_id in selector if int(chunk_id) in self._rows_by_id],
                dtype=np.int_,
            )
            if len(row_selector) == 0:
                return []
        results = self._backend.query(vectors, k, selector=row_selector)
        return [(self._chunk_ids[indices], distances) for indices, distances in results]

    def save(self, path: Path) -> None:
        """Persist the underlying dense backend."""
        self._backend.save(path)
