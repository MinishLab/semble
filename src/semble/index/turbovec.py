from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson
from turbovec import IdMapIndex
from vicinity.backends.basic import BasicArgs
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize, normalize_or_copy

from semble.index.dense import SelectableBasicBackend

_DENSE_SEGMENTS_DIR = "dense_segments"
_SEGMENT_ID = 0
_SEGMENT_MANIFEST = "manifest.json"
_BUILD_ADD_BATCH_SIZE = 8192


def _segment_file(segment_id: int) -> str:
    return f"segment-{segment_id:04d}.tvim"


def _segment_meta(segment_id: int) -> str:
    return f"segment-{segment_id:04d}.meta.json"


def _segment_vectors_file(segment_id: int) -> str:
    return f"segment-{segment_id:04d}.vectors.npy"


def _copy_existing_file(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return True


SegmentVectors = Sequence[npt.NDArray[np.float32]] | npt.NDArray[np.float32]


class _MatrixRows(Sequence[npt.NDArray[np.float32]]):
    def __init__(self, matrix: npt.NDArray[np.float32], rows: npt.NDArray[np.int_]) -> None:
        self._matrix = matrix
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int | slice) -> npt.NDArray[np.float32] | list[npt.NDArray[np.float32]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return self._matrix[int(self._rows[index])]


@dataclass(frozen=True, slots=True)
class _DenseSegment:
    segment_id: int
    index: IdMapIndex
    ids: npt.NDArray[np.uint64]
    vectors: SegmentVectors | None = None


def _segment_vectors_array(vectors: SegmentVectors) -> npt.NDArray[np.float32]:
    if isinstance(vectors, np.ndarray):
        return np.asarray(vectors, dtype=np.float32)
    return np.asarray([vectors[index] for index in range(len(vectors))], dtype=np.float32)


@dataclass(slots=True)
class TurboVecBuildWriter:
    arguments: BasicArgs
    ids: list[int] = field(default_factory=list)
    index: IdMapIndex | None = None
    dim: int | None = None
    pending_rows: list[npt.NDArray[np.float32]] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)
    add_batch_size: int = _BUILD_ADD_BATCH_SIZE

    def add_rows(self, vectors: Sequence[npt.NDArray[np.float32]], vector_ids: Sequence[int]) -> None:
        """Append dense rows to the build-time native segment writer."""
        if len(vectors) != len(vector_ids):
            raise ValueError("vectors and vector_ids must have the same length")
        for vector, vector_id in zip(vectors, vector_ids):
            row = np.asarray(vector, dtype=np.float32).reshape(-1)
            if self.dim is None:
                self.dim = row.shape[0]
                self.index = IdMapIndex(dim=self.dim, bit_width=4)
            elif row.shape[0] != self.dim:
                raise ValueError("segment vectors must match existing vector dimension")
            self.pending_rows.append(row)
            self.pending_ids.append(int(vector_id))
            self.ids.append(int(vector_id))

    def finish(self) -> "TurboVecBasicBackend":
        """Return a searchable TurboVec backend from streamed rows."""
        if self.dim is None or self.index is None:
            return TurboVecBasicBackend(
                np.empty((0, 0), dtype=np.float32),
                self.arguments,
                np.array([], dtype=np.uint64),
            )
        rows = np.asarray(self.pending_rows, dtype=np.float32)
        ids = np.asarray(self.pending_ids, dtype=np.uint64)
        normalized_rows = normalize_or_copy(rows)
        self.index.add_with_ids(normalized_rows, ids)
        backend = TurboVecBasicBackend(np.empty((0, self.dim), dtype=np.float32), self.arguments, ids)
        backend._segments = [_DenseSegment(_SEGMENT_ID, self.index, ids.copy(), normalized_rows)]
        self.pending_rows.clear()
        self.pending_ids.clear()
        return backend


class TurboVecBasicBackend(SelectableBasicBackend):
    def __init__(
        self,
        vectors: npt.NDArray[np.float32],
        arguments: BasicArgs,
        vector_ids: npt.NDArray[np.uint64] | None = None,
        tombstones: set[int] | None = None,
        vectors_path: Path | None = None,
    ) -> None:
        """Create a TurboVec backend with stable vector IDs."""
        super().__init__(vectors, arguments)
        self._vectors_path = vectors_path
        self.vector_ids = (
            np.arange(len(self._vectors), dtype=np.uint64)
            if vector_ids is None
            else np.asarray(vector_ids, dtype=np.uint64)
        )
        self.tombstones = set() if tombstones is None else set(tombstones)
        self._segments: list[_DenseSegment] = []
        self._rebuild_segments()

    @property
    def vectors(self) -> npt.NDArray[np.float32]:
        """Return raw vectors, loading vectors.npy only when needed."""
        self._load_vectors_if_needed()
        return self._vectors

    @classmethod
    def from_embedding_rows(
        cls,
        vectors: Sequence[npt.NDArray[np.float32]],
        arguments: BasicArgs,
        vector_ids: npt.NDArray[np.uint64] | None = None,
    ) -> "TurboVecBasicBackend":
        """Build native dense segment rows without retaining one raw matrix."""
        row_count = len(vectors)
        ids = np.arange(row_count, dtype=np.uint64) if vector_ids is None else np.asarray(vector_ids, dtype=np.uint64)
        if len(ids) != row_count:
            raise ValueError("vectors and vector_ids must have the same length")
        if row_count == 0:
            return cls(np.empty((0, 0), dtype=np.float32), arguments, ids)

        rows = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
        dim = rows[0].shape[0]
        if any(row.shape[0] != dim for row in rows):
            raise ValueError("segment vectors must match existing vector dimension")

        segment_vectors = np.asarray(rows, dtype=np.float32)
        normalized_vectors = normalize_or_copy(segment_vectors)
        segment_index = IdMapIndex(dim=dim, bit_width=4)
        segment_index.add_with_ids(normalized_vectors, ids)

        backend = cls(np.empty((0, dim), dtype=np.float32), arguments, ids)
        backend._segments = [_DenseSegment(_SEGMENT_ID, segment_index, ids.copy(), normalized_vectors)]
        return backend

    def _segment_rows_by_id(self, requested_ids: set[int]) -> dict[int, npt.NDArray[np.float32]]:
        rows_by_id: dict[int, npt.NDArray[np.float32]] = {}
        for segment in self._segments:
            segment_ids = [int(vector_id) for vector_id in segment.ids]
            if segment.vectors is not None:
                rows_by_id.update(
                    {
                        vector_id: segment.vectors[row]
                        for row, vector_id in enumerate(segment_ids)
                        if vector_id in requested_ids
                    }
                )
                continue
            if self._vectors_path is None:
                continue
            vectors_file = self._vectors_path / _DENSE_SEGMENTS_DIR / _segment_vectors_file(segment.segment_id)
            if not vectors_file.exists():
                continue
            missing_rows = [
                (row, vector_id)
                for row, vector_id in enumerate(segment_ids)
                if vector_id in requested_ids and vector_id not in rows_by_id
            ]
            if missing_rows:
                segment_vectors = np.load(vectors_file, mmap_mode="r")
                rows_by_id.update(
                    {vector_id: np.asarray(segment_vectors[row], dtype=np.float32) for row, vector_id in missing_rows}
                )
        return rows_by_id

    def _matrix_rows_by_id(self, chunk_ids: Sequence[int]) -> Sequence[npt.NDArray[np.float32]] | None:
        vectors = self.vectors
        if len(self.vector_ids) != len(vectors):
            return None
        rows_by_id = {int(vector_id): vectors[row] for row, vector_id in enumerate(self.vector_ids)}
        if any(chunk_id not in rows_by_id for chunk_id in chunk_ids):
            return None
        return [rows_by_id[chunk_id] for chunk_id in chunk_ids]

    def vector_rows_for_ids(self, chunk_ids: Sequence[int]) -> Sequence[npt.NDArray[np.float32]] | None:
        """Return dense rows by stable chunk ID without loading raw matrix when possible."""
        rows_by_id = self._segment_rows_by_id(set(chunk_ids))
        if all(chunk_id in rows_by_id for chunk_id in chunk_ids):
            return [rows_by_id[chunk_id] for chunk_id in chunk_ids]
        return self._matrix_rows_by_id(chunk_ids)

    @vectors.setter
    def vectors(self, value: npt.NDArray[np.float32]) -> None:
        matrix = np.asarray(value, dtype=np.float32)
        if np.ndim(matrix) != 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(matrix)}")
        self._vectors = normalize_or_copy(matrix)
        self._vectors_path = None
        self._update_precomputed_data()
        self._rebuild_segments()

    def _load_vectors_if_needed(self) -> None:
        if self._vectors_path is None:
            return
        vectors_path = self._vectors_path
        loaded = SelectableBasicBackend.load(vectors_path)
        vectors = loaded.vectors
        missing_count = len(self.vector_ids) - len(vectors)
        if missing_count > 0:
            pending_vectors = []
            covered_ids = {int(vector_id) for vector_id in self.vector_ids[: len(vectors)]}
            segment_vectors_path = vectors_path / _DENSE_SEGMENTS_DIR
            for segment in self._segments:
                if all(int(vector_id) in covered_ids for vector_id in segment.ids):
                    continue
                if segment.vectors is not None:
                    pending_vectors.append(_segment_vectors_array(segment.vectors))
                    continue
                vectors_file = segment_vectors_path / _segment_vectors_file(segment.segment_id)
                if vectors_file.exists():
                    pending_vectors.append(np.load(vectors_file).astype(np.float32, copy=False))
            if pending_vectors:
                vectors = np.vstack([vectors, *pending_vectors])[: len(self.vector_ids)]
        if len(vectors) != len(self.vector_ids):
            raise FileNotFoundError("Index is missing raw dense vectors for appended segments")
        self._vectors = vectors
        self.arguments = loaded.arguments
        self._vectors_path = None

    def _live_mask(self) -> npt.NDArray[np.bool_]:
        return np.array([int(vector_id) not in self.tombstones for vector_id in self.vector_ids], dtype=bool)

    def _rebuild_segments(self) -> None:
        self._segments = []
        if not self._vectors.size:
            return
        live_mask = self._live_mask()
        if not live_mask.any():
            return
        live_rows = np.flatnonzero(live_mask)
        live_vectors = self._vectors[live_rows]
        live_ids = self.vector_ids[live_mask].astype(np.uint64, copy=False)
        index = IdMapIndex(dim=self._vectors.shape[1], bit_width=4)
        index.add_with_ids(live_vectors, live_ids)
        self._segments = [_DenseSegment(_SEGMENT_ID, index, live_ids.copy(), _MatrixRows(self._vectors, live_rows))]

    def add_segment(self, vectors: npt.NDArray[np.float32], vector_ids: npt.NDArray[np.uint64]) -> None:
        """Append a native dense segment by stable chunk ID."""
        segment_vectors = np.asarray(vectors, dtype=np.float32)
        if np.ndim(segment_vectors) != 2:
            raise ValueError(f"Your array does not have 2 dimensions: {np.ndim(segment_vectors)}")
        segment_vectors = normalize_or_copy(segment_vectors)
        segment_ids = np.asarray(vector_ids, dtype=np.uint64)
        if len(segment_vectors) != len(segment_ids):
            raise ValueError("vectors and vector_ids must have the same length")
        if not len(segment_ids):
            return
        if self._vectors.shape[1] and segment_vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("segment vectors must match existing vector dimension")

        segment_id = max((segment.segment_id for segment in self._segments), default=-1) + 1
        index = IdMapIndex(dim=segment_vectors.shape[1], bit_width=4)
        index.add_with_ids(segment_vectors, segment_ids)
        self._segments.append(_DenseSegment(segment_id, index, segment_ids.copy(), segment_vectors.copy()))
        self.vector_ids = np.concatenate([self.vector_ids, segment_ids])
        self.tombstones.difference_update(int(vector_id) for vector_id in segment_ids)
        if self._vectors_path is None:
            self._vectors = np.vstack([self._vectors, segment_vectors]).astype(np.float32, copy=False)
            self._update_precomputed_data()

    def _empty_result(self, vectors: npt.NDArray) -> QueryResult:
        return [(np.array([], dtype=np.int_), np.array([], dtype=np.float32)) for _ in range(len(vectors))]

    def _selector_rows(self, selector: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        live_rows_by_id = {
            int(vector_id): row_index
            for row_index, vector_id in enumerate(self.vector_ids)
            if int(vector_id) not in self.tombstones
        }
        selected_rows = [live_rows_by_id[int(vector_id)] for vector_id in selector if int(vector_id) in live_rows_by_id]
        return np.array(selected_rows, dtype=np.int_)

    def _query_segments(
        self,
        vectors: npt.NDArray,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> QueryResult:
        if not self._segments:
            return self._empty_result(vectors)

        if selector is not None:
            allowed_ids = {int(vector_id) for vector_id in selector} - self.tombstones
        elif self.tombstones:
            allowed_ids = {
                int(vector_id)
                for segment in self._segments
                for vector_id in segment.ids
                if int(vector_id) not in self.tombstones
            }
        else:
            allowed_ids = None
        if allowed_ids is not None and not allowed_ids:
            return self._empty_result(vectors)

        query_vectors = normalize(vectors.astype(np.float32, copy=False))
        merged: list[list[tuple[int, float]]] = [[] for _ in range(len(query_vectors))]
        for segment in self._segments:
            effective_k = len(segment.ids) if allowed_ids is not None else min(k, len(segment.ids))
            scores, indices = segment.index.search(query_vectors, effective_k)
            distances = 1 - np.clip(scores, -1.0, 1.0)
            for query_index, (query_indices, query_distances) in enumerate(zip(indices, distances)):
                for index, distance in zip(query_indices, query_distances):
                    chunk_id = int(index)
                    if allowed_ids is None or chunk_id in allowed_ids:
                        merged[query_index].append((chunk_id, float(distance)))

        results: QueryResult = []
        for items in merged:
            nearest = sorted(items, key=lambda item: item[1])[:k]
            results.append(
                (
                    np.array([index for index, _ in nearest], dtype=np.int_),
                    np.array([distance for _, distance in nearest], dtype=np.float32),
                )
            )
        return results

    def query(self, vectors: npt.NDArray, k: int, selector: npt.NDArray[np.int_] | None = None) -> QueryResult:
        """Query live dense vectors by stable IDs."""
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")
        if selector is not None and self._segments:
            return self._query_segments(vectors, k, selector)
        if selector is not None:
            selector_rows = self._selector_rows(selector)
            if len(selector_rows) == 0:
                return self._empty_result(vectors)
            return [
                (self.vector_ids[indices].astype(np.int_), distances)
                for indices, distances in super().query(vectors, k, selector_rows)
            ]
        return self._query_segments(vectors, k)

    def delete_ids(self, chunk_ids: set[int]) -> None:
        """Tombstone dense IDs so queries skip deleted vectors."""
        self.tombstones.update(chunk_ids)
        if self._vectors_path is None:
            self._rebuild_segments()

    def compact(self) -> None:
        """Remove tombstoned vectors from the dense segment store."""
        self._load_vectors_if_needed()
        live_mask = self._live_mask()
        self._vectors = self._vectors[live_mask]
        self.vector_ids = self.vector_ids[live_mask].copy()
        self.tombstones.clear()
        self._rebuild_segments()

    def _save_arguments(self, path: Path) -> None:
        arguments = {
            key: value.value if hasattr(value, "value") else value for key, value in self.arguments.dict().items()
        }
        (path / "arguments.json").write_bytes(orjson.dumps(arguments))

    def _save_raw_vectors(self, path: Path, source_path: Path | None) -> None:
        if source_path is None:
            super().save(path)
            return
        self._save_arguments(path)
        _copy_existing_file(source_path / "vectors.npy", path / "vectors.npy")

    def _save_dense_segment(
        self,
        segment: _DenseSegment,
        segments_path: Path,
        source_segments_path: Path | None,
        *,
        created_generation: int | None,
        vector_dim: int,
        model_path: str | None,
        source_snapshot_id: str | None,
    ) -> set[str]:
        segment_file = _segment_file(segment.segment_id)
        segment_meta = _segment_meta(segment.segment_id)
        segment_vectors = _segment_vectors_file(segment.segment_id)
        target_file = segments_path / segment_file
        source_file = None if source_segments_path is None else source_segments_path / segment_file
        if segment.vectors is None and source_file is not None and _copy_existing_file(source_file, target_file):
            pass
        else:
            segment.index.write(str(target_file))

        expected_files = {segment_file, segment_meta}
        target_vectors = segments_path / segment_vectors
        source_vectors = None if source_segments_path is None else source_segments_path / segment_vectors
        if segment.vectors is not None:
            np.save(target_vectors, _segment_vectors_array(segment.vectors))
            expected_files.add(segment_vectors)
        elif source_vectors is not None and _copy_existing_file(source_vectors, target_vectors):
            expected_files.add(segment_vectors)

        live_count = sum(1 for vector_id in segment.ids if int(vector_id) not in self.tombstones)
        metadata = {
            "segment_id": segment.segment_id,
            "chunk_ids": [int(vector_id) for vector_id in segment.ids],
            "created_generation": created_generation,
            "live_count": live_count,
            "vector_dim": vector_dim,
            "model_path": model_path,
        }
        if source_snapshot_id is not None:
            metadata["source_snapshot_id"] = source_snapshot_id
        (segments_path / segment_meta).write_bytes(orjson.dumps(metadata))
        return expected_files

    def _delete_stale_segment_files(self, segments_path: Path, expected_files: set[str]) -> None:
        for item in segments_path.iterdir():
            if item.name not in expected_files:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def write_to_path(
        self,
        path: Path,
        *,
        created_generation: int | None = None,
        model_path: str | None = None,
        source_snapshot_id: str | None = None,
    ) -> None:
        """Write dense vectors and native segment metadata to path."""
        path.mkdir(parents=True, exist_ok=True)
        source_path = self._vectors_path
        self._save_raw_vectors(path, source_path)

        segments_path = path / _DENSE_SEGMENTS_DIR
        source_segments_path = None if source_path is None else source_path / _DENSE_SEGMENTS_DIR
        segments_path.mkdir(parents=True, exist_ok=True)

        vector_dim = int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0
        manifest = {
            "vector_ids": [int(vector_id) for vector_id in self.vector_ids],
            "tombstones": sorted(self.tombstones),
            "vector_dim": vector_dim,
        }
        (segments_path / _SEGMENT_MANIFEST).write_bytes(orjson.dumps(manifest))

        expected_files = {_SEGMENT_MANIFEST}
        for segment in self._segments:
            expected_files.update(
                self._save_dense_segment(
                    segment,
                    segments_path,
                    source_segments_path,
                    created_generation=created_generation,
                    vector_dim=vector_dim,
                    model_path=model_path,
                    source_snapshot_id=source_snapshot_id,
                )
            )
        self._delete_stale_segment_files(segments_path, expected_files)

    def save(
        self,
        path: Path,
        *,
        created_generation: int | None = None,
        model_path: str | None = None,
        source_snapshot_id: str | None = None,
    ) -> None:
        """Save the selectable basic backend and TurboVec native segment metadata."""
        self.write_to_path(
            path,
            created_generation=created_generation,
            model_path=model_path,
            source_snapshot_id=source_snapshot_id,
        )

    @classmethod
    def load(cls, path: Path) -> "TurboVecBasicBackend":
        """Load a TurboVec backend with persisted segment metadata."""
        segments_path = path / _DENSE_SEGMENTS_DIR
        manifest_path = segments_path / _SEGMENT_MANIFEST
        if manifest_path.exists():
            manifest = orjson.loads(manifest_path.read_bytes())
            vector_ids = np.array(manifest.get("vector_ids", []), dtype=np.uint64)
            tombstones = set(int(chunk_id) for chunk_id in manifest.get("tombstones", []))
            vector_dim = int(manifest.get("vector_dim", 0))
            backend = cls(
                np.empty((0, vector_dim), dtype=np.float32),
                BasicArgs.load(path / "arguments.json"),
                vector_ids,
                tombstones,
                vectors_path=path,
            )
            backend._segments = []
            for segment_meta in sorted(segments_path.glob("segment-*.meta.json")):
                metadata = orjson.loads(segment_meta.read_bytes())
                segment_id = int(metadata["segment_id"])
                segment_file = segments_path / _segment_file(segment_id)
                if not segment_file.exists():
                    continue
                segment_ids = np.array(metadata["chunk_ids"], dtype=np.uint64)
                backend._segments.append(_DenseSegment(segment_id, IdMapIndex.load(str(segment_file)), segment_ids))
            return backend

        loaded = super().load(path)
        return TurboVecBasicBackend(loaded.vectors, loaded.arguments)
