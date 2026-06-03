from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson
from bm25s import BM25
from model2vec.model import StaticModel
from vicinity.backends.basic import BasicArgs

from semble.cache import (
    GIT_CACHE_ROOTS_VERSION,
    build_git_cache_save_metadata,
    get_rebuild_cache,
    get_validated_cache,
    refresh_git_cache_metadata,
)
from semble.chunking import chunk_source
from semble.index.chunk_store import LmdbChunkStore
from semble.index.create import create_index_build_from_path
from semble.index.dense import SelectableBasicBackend, embed_chunks, load_model
from semble.index.file_walker import walk_files
from semble.index.files import FileStatus, detect_language, get_extensions, get_file_status, read_file_text
from semble.index.source_inventory import GitWalkPlan, build_git_walk_plan
from semble.index.sparse import SparseIndex, TantivySparseIndex
from semble.index.types import PersistencePath
from semble.search import _search_semantic, search
from semble.stats import save_search_stats
from semble.types import CallType, Chunk, ContentType, FilterSpec, IndexStats, SearchResult

_GIT_CLONE_TIMEOUT = int(os.environ.get("SEMBLE_CLONE_TIMEOUT", 60))
_DEFAULT_CONTENT: tuple[ContentType, ...] = (ContentType.CODE,)
_ALL_CONTENT: tuple[ContentType, ...] = (ContentType.CODE, ContentType.DOCS, ContentType.CONFIG)
_INCLUDE_TEXT_FILES_DEPRECATION_MSG = (
    "include_text_files is deprecated and will be removed in a future version. "
    "Use content=(ContentType.CODE, ContentType.DOCS, ContentType.CONFIG) instead."
)
_REBUILD_CACHE_VERSION = 1


def _apply_include_text_files(
    content: ContentType | Sequence[ContentType], include_text_files: bool | None
) -> tuple[ContentType, ...]:
    """Apply the deprecated include_text_files override, emitting a DeprecationWarning."""
    if include_text_files is None:
        return (content,) if isinstance(content, ContentType) else tuple(content)
    warnings.warn(
        _INCLUDE_TEXT_FILES_DEPRECATION_MSG,
        DeprecationWarning,
        stacklevel=3,
    )
    return _ALL_CONTENT if include_text_files else _DEFAULT_CONTENT


def _cache_is_loadable_for_git(cache_path: Path) -> bool:
    """Reject hybrid-generation caches that this non-streaming path cannot prove active."""
    metadata_path = PersistencePath.from_path(cache_path).metadata
    try:
        metadata = orjson.loads(metadata_path.read_bytes())
    except OSError:
        return True
    except ValueError:
        return False
    return metadata.get("active_generation") is None


class _StableIdSemanticBackend:
    """Map persisted dense row positions back to stable chunk IDs."""

    def __init__(self, backend: SelectableBasicBackend, chunk_ids: Sequence[int]) -> None:
        """Wrap a row-position backend with stable chunk-id outputs."""
        self._backend = backend
        self._chunk_ids = np.array(chunk_ids, dtype=np.int_)
        self._rows_by_id = {int(chunk_id): row for row, chunk_id in enumerate(chunk_ids)}

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


class LazyChunkList(Sequence[Chunk]):
    """Sequence that loads persisted chunks by stable ID only when requested."""

    def __init__(
        self,
        chunk_ids: Sequence[int],
        store_path: Path,
        file_paths: Sequence[str] = (),
        languages: Sequence[str | None] = (),
    ) -> None:
        """Create a lazy chunk sequence backed by LMDB payloads."""
        self._chunk_ids = list(chunk_ids)
        self._store_path = store_path
        self.chunk_file_paths = list(file_paths)
        self.chunk_languages = list(languages)
        self._cache: dict[int, Chunk] = {}
        self._file_mapping = self._build_mapping(self.chunk_file_paths)
        self._language_mapping = self._build_language_mapping(self.chunk_languages)

    def __len__(self) -> int:
        """Return total persisted chunk count."""
        return len(self._chunk_ids)

    def __iter__(self) -> Iterator[Chunk]:
        """Yield chunks in persisted order."""
        for chunk_id in self._chunk_ids:
            yield self.chunk_by_id(chunk_id)

    def __getitem__(self, index: int) -> Chunk:
        """Return one chunk by persisted position."""
        return self.chunk_by_id(self._chunk_ids[index])

    def __eq__(self, other: object) -> bool:
        """Compare with another chunk sequence by loaded chunk values."""
        if not isinstance(other, Sequence):
            return False
        return list(self) == list(other)

    def chunk_by_id(self, chunk_id: int) -> Chunk:
        """Return one chunk by stable ID, loading it from LMDB on first access."""
        chunk = self._cache.get(chunk_id)
        if chunk is not None:
            return chunk
        store = LmdbChunkStore.open(self._store_path, readonly=True)
        try:
            loaded = store.get_chunk(chunk_id)
        finally:
            store.close()
        if loaded is None:
            raise FileNotFoundError(f"Index chunk store is missing chunk payload for id {chunk_id}")
        self._cache[chunk_id] = loaded
        return loaded

    def chunks_by_id(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Return chunks for stable IDs, preserving requested order."""
        missing_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in self._cache]
        if missing_ids:
            store = LmdbChunkStore.open(self._store_path, readonly=True)
            try:
                loaded_chunks = store.get_chunks(missing_ids)
            finally:
                store.close()
            if len(loaded_chunks) != len(missing_ids):
                raise FileNotFoundError("Index chunk store is missing chunk payloads")
            self._cache.update(zip(missing_ids, loaded_chunks))
        return [self._cache[chunk_id] for chunk_id in chunk_ids]

    def chunk_ids_for_paths(self, file_paths: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks under selected file paths."""
        return [chunk_id for path in file_paths for chunk_id in self._file_mapping.get(path, [])]

    def chunk_ids_for_languages(self, languages: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks in selected languages."""
        return [chunk_id for language in languages for chunk_id in self._language_mapping.get(language, [])]

    def _build_mapping(self, file_paths: Sequence[str]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, file_path in zip(self._chunk_ids, file_paths):
            mapping[file_path].append(chunk_id)
        return dict(mapping)

    def file_mapping(self) -> dict[str, list[int]]:
        """Return persisted file-to-chunk-ID mapping."""
        return self._file_mapping

    def language_mapping(self) -> dict[str, list[int]]:
        """Return persisted language-to-chunk-ID mapping."""
        return self._language_mapping

    def _build_language_mapping(self, languages: Sequence[str | None]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, language in zip(self._chunk_ids, languages):
            if language:
                mapping[language].append(chunk_id)
        return dict(mapping)

    def chunk_ids(self) -> list[int]:
        """Return stable chunk IDs in sequence order."""
        return list(self._chunk_ids)


class MergedLazyChunkList(Sequence[Chunk]):
    """Lazy chunk sequence that overlays changed chunks on a persisted chunk store."""

    def __init__(
        self,
        chunk_ids: Sequence[int],
        store_path: Path,
        file_paths: Sequence[str],
        languages: Sequence[str | None],
        changed_chunks_by_id: dict[int, Chunk],
        deleted_chunk_ids: set[int],
    ) -> None:
        """Create a sequence for incremental rebuilds without loading unchanged payloads."""
        self._chunk_ids = list(chunk_ids)
        self._store_path = store_path
        self.chunk_file_paths = list(file_paths)
        self.chunk_languages = list(languages)
        self._changed_chunks_by_id = changed_chunks_by_id
        self._deleted_chunk_ids = set(deleted_chunk_ids)
        self._cache: dict[int, Chunk] = dict(changed_chunks_by_id)
        self._file_mapping = self._build_mapping(self.chunk_file_paths)
        self._language_mapping = self._build_language_mapping(self.chunk_languages)

    def __len__(self) -> int:
        """Return total chunk count."""
        return len(self._chunk_ids)

    def __iter__(self) -> Iterator[Chunk]:
        """Yield chunks in sequence order."""
        for chunk_id in self._chunk_ids:
            yield self.chunk_by_id(chunk_id)

    def __getitem__(self, index: int) -> Chunk:
        """Return one chunk by sequence position."""
        return self.chunk_by_id(self._chunk_ids[index])

    def chunk_by_id(self, chunk_id: int) -> Chunk:
        """Return one chunk by stable ID."""
        cached = self._cache.get(chunk_id)
        if cached is not None:
            return cached
        store = LmdbChunkStore.open(self._store_path, readonly=True)
        try:
            chunk = store.get_chunk(chunk_id)
        finally:
            store.close()
        if chunk is None:
            raise FileNotFoundError(f"Index chunk store is missing chunk payload for id {chunk_id}")
        self._cache[chunk_id] = chunk
        return chunk

    def chunks_by_id(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Return chunks for stable IDs, preserving requested order."""
        missing_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in self._cache]
        if missing_ids:
            store = LmdbChunkStore.open(self._store_path, readonly=True)
            try:
                loaded_chunks = store.get_chunks(missing_ids)
            finally:
                store.close()
            if len(loaded_chunks) != len(missing_ids):
                raise FileNotFoundError("Index chunk store is missing chunk payloads")
            self._cache.update(zip(missing_ids, loaded_chunks))
        return [self._cache[chunk_id] for chunk_id in chunk_ids]

    def chunk_ids_for_paths(self, file_paths: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks under selected file paths."""
        return [chunk_id for path in file_paths for chunk_id in self._file_mapping.get(path, [])]

    def chunk_ids_for_languages(self, languages: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks in selected languages."""
        return [chunk_id for language in languages for chunk_id in self._language_mapping.get(language, [])]

    def file_mapping(self) -> dict[str, list[int]]:
        """Return file-to-chunk-ID mapping."""
        return self._file_mapping

    def language_mapping(self) -> dict[str, list[int]]:
        """Return language-to-chunk-ID mapping."""
        return self._language_mapping

    def chunk_ids(self) -> list[int]:
        """Return stable chunk IDs in sequence order."""
        return list(self._chunk_ids)

    def write_chunks_to_store(self, store: LmdbChunkStore, chunk_ids: list[int]) -> None:
        """Copy unchanged payloads and write changed payloads without decoding unchanged chunks."""
        changed_ids = set(self._changed_chunks_by_id)
        old_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in changed_ids]
        if old_ids:
            store.copy_chunks_from(self._store_path, old_ids)
        changed_ids_in_order = [chunk_id for chunk_id in chunk_ids if chunk_id in changed_ids]
        if changed_ids_in_order:
            store.write_chunks_with_ids(
                [self._changed_chunks_by_id[chunk_id] for chunk_id in changed_ids_in_order],
                changed_ids_in_order,
            )

    def copy_chunk_store_to(self, target_path: Path, chunk_ids: list[int]) -> None:
        """Copy the old LMDB store wholesale, then patch changed/deleted chunk payloads."""
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(self._store_path, target_path)
        store = LmdbChunkStore.open(target_path)
        try:
            store.delete_chunks(sorted(self._deleted_chunk_ids))
            changed_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in self._changed_chunks_by_id]
            if changed_ids:
                store.write_chunks_with_ids(
                    [self._changed_chunks_by_id[chunk_id] for chunk_id in changed_ids],
                    changed_ids,
                )
        finally:
            store.close()

    def _build_mapping(self, file_paths: Sequence[str]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, file_path in zip(self._chunk_ids, file_paths):
            mapping[file_path].append(chunk_id)
        return dict(mapping)

    def _build_language_mapping(self, languages: Sequence[str | None]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, language in zip(self._chunk_ids, languages):
            if language:
                mapping[language].append(chunk_id)
        return dict(mapping)


def _file_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()


@dataclass(slots=True)
class _RebuildState:
    chunks: list[Chunk]
    vectors: list[npt.NDArray[np.float32]]
    file_sizes: dict[str, int]
    file_hashes: dict[str, str]
    changed_chunks: list[Chunk]
    changed_positions: list[int]
    deleted_chunk_ids: set[int]
    next_chunk_id: int


@dataclass(slots=True)
class _LazyRebuildState:
    chunk_ids: list[int]
    chunk_file_paths: list[str]
    chunk_languages: list[str | None]
    changed_chunks_by_id: dict[int, Chunk]
    vectors: list[npt.NDArray[np.float32]]
    file_sizes: dict[str, int]
    file_hashes: dict[str, str]
    changed_chunks: list[Chunk]
    changed_positions: list[int]
    deleted_chunk_ids: set[int]
    next_chunk_id: int


@dataclass(frozen=True, slots=True)
class _SeedData:
    file_hashes: dict[str, str]
    chunks_by_file: dict[str, list[Chunk]]
    vectors_by_file: dict[str, list[npt.NDArray[np.float32]]]
    chunk_ids_by_file: dict[str, list[int]]
    file_sizes: dict[str, int]
    file_paths: list[str]
    tracked_paths: list[str] | None
    git_roots: dict[str, str] | None
    write_time: float | None
    next_chunk_id: int


@dataclass(frozen=True, slots=True)
class _SeedMetadata:
    file_hashes: dict[str, str]
    vectors: npt.NDArray[np.float32]
    chunk_ids: list[int]
    chunk_file_paths: list[str]
    chunk_languages: list[str | None]
    chunk_ids_by_file: dict[str, list[int]]
    vector_rows_by_file: dict[str, list[npt.NDArray[np.float32]]]
    languages_by_file: dict[str, list[str | None]]
    file_sizes: dict[str, int]
    file_paths: list[str]
    tracked_paths: list[str] | None
    git_roots: dict[str, str] | None
    write_time: float | None
    next_chunk_id: int
    chunk_store_path: Path


@dataclass(frozen=True, slots=True)
class _RebuiltFile:
    path: str
    chunks: list[Chunk]
    reused: bool
    size: int | None
    file_hash: str | None


def _empty_rebuild_state(next_chunk_id: int = 0) -> _RebuildState:
    return _RebuildState([], [], {}, {}, [], [], set(), next_chunk_id)


def _seed_file_hashes(metadata: dict) -> dict[str, str]:
    if metadata.get("rebuild_cache_version") != _REBUILD_CACHE_VERSION:
        return {}
    file_hashes = metadata.get("file_hashes", {})
    if not isinstance(file_hashes, dict):
        return {}
    return {str(path): str(file_hash) for path, file_hash in file_hashes.items()}


def _seed_chunk_ids(metadata: dict) -> list[int] | None:
    chunk_ids = metadata.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        return None
    return [int(chunk_id) for chunk_id in chunk_ids]


def _load_seed_chunks(persistence_paths: PersistencePath, chunk_ids: list[int]) -> list[Chunk] | None:
    if not persistence_paths.chunk_store.exists():
        return None
    store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
    try:
        chunks = store.get_chunks(chunk_ids)
    finally:
        store.close()
    if len(chunks) != len(chunk_ids):
        return None
    return chunks


def _seed_git_roots(metadata: dict) -> dict[str, str] | None:
    if metadata.get("git_roots_version") != GIT_CACHE_ROOTS_VERSION:
        return None
    git_roots = metadata.get("git_roots")
    if not isinstance(git_roots, list):
        return None
    return {str(item.get("path", "")): str(item.get("head", "")) for item in git_roots if isinstance(item, dict)}


def _seed_tracked_paths(metadata: dict) -> list[str] | None:
    tracked_paths = metadata.get("tracked_paths")
    if not isinstance(tracked_paths, list):
        return None
    return [str(path) for path in tracked_paths]


def _seed_file_sizes(metadata: dict) -> dict[str, int]:
    return {str(key): int(value) for key, value in metadata.get("file_sizes", {}).items()}


def _seed_file_paths(metadata: dict) -> list[str]:
    return [str(path) for path in metadata.get("file_paths", [])]


def _seed_data_from_chunks(
    metadata: dict,
    file_hashes: dict[str, str],
    chunk_ids: list[int],
    chunks: list[Chunk],
    vectors: npt.NDArray[np.float32],
) -> _SeedData:
    chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
    vectors_by_file: dict[str, list[npt.NDArray[np.float32]]] = defaultdict(list)
    chunk_ids_by_file: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        chunks_by_file[chunk.file_path].append(chunk)
        vectors_by_file[chunk.file_path].append(vectors[index])
        chunk_ids_by_file[chunk.file_path].append(chunk_ids[index])
    return _SeedData(
        file_hashes,
        chunks_by_file,
        vectors_by_file,
        chunk_ids_by_file,
        _seed_file_sizes(metadata),
        _seed_file_paths(metadata),
        _seed_tracked_paths(metadata),
        _seed_git_roots(metadata),
        float(metadata["time"]) if "time" in metadata else None,
        max(chunk_ids, default=-1) + 1,
    )


def _metadata_chunk_file_paths(metadata: dict) -> list[str] | None:
    file_paths = metadata.get("chunk_file_paths")
    if not isinstance(file_paths, list):
        return None
    return [str(file_path) for file_path in file_paths]


def _metadata_chunk_languages(metadata: dict) -> list[str | None] | None:
    languages = metadata.get("chunk_languages")
    if not isinstance(languages, list):
        return None
    return [None if language is None else str(language) for language in languages]


def _seed_metadata_from_vectors(
    metadata: dict,
    file_hashes: dict[str, str],
    chunk_ids: list[int],
    chunk_file_paths: list[str],
    chunk_languages: list[str | None],
    vectors: npt.NDArray[np.float32],
    chunk_store_path: Path,
) -> _SeedMetadata:
    chunk_ids_by_file: dict[str, list[int]] = defaultdict(list)
    vector_rows_by_file: dict[str, list[npt.NDArray[np.float32]]] = defaultdict(list)
    languages_by_file: dict[str, list[str | None]] = defaultdict(list)
    for index, (chunk_id, file_path, language) in enumerate(zip(chunk_ids, chunk_file_paths, chunk_languages)):
        chunk_ids_by_file[file_path].append(chunk_id)
        vector_rows_by_file[file_path].append(vectors[index])
        languages_by_file[file_path].append(language)
    return _SeedMetadata(
        file_hashes,
        vectors,
        chunk_ids,
        chunk_file_paths,
        chunk_languages,
        chunk_ids_by_file,
        vector_rows_by_file,
        languages_by_file,
        _seed_file_sizes(metadata),
        _seed_file_paths(metadata),
        _seed_tracked_paths(metadata),
        _seed_git_roots(metadata),
        float(metadata["time"]) if "time" in metadata else None,
        max(chunk_ids, default=-1) + 1,
        chunk_store_path,
    )


def _load_rebuild_seed(seed_path: Path) -> _SeedData | None:
    persistence_paths = PersistencePath.from_path(seed_path)
    try:
        metadata = orjson.loads(persistence_paths.metadata.read_bytes())
        vectors = SelectableBasicBackend.load(persistence_paths.semantic_index).vectors
    except (OSError, ValueError, FileNotFoundError):
        return None
    file_hashes = _seed_file_hashes(metadata)
    chunk_ids = _seed_chunk_ids(metadata)
    if not file_hashes or chunk_ids is None:
        return None
    chunks = _load_seed_chunks(persistence_paths, chunk_ids)
    if chunks is None or len(vectors) != len(chunks):
        return None
    return _seed_data_from_chunks(metadata, file_hashes, chunk_ids, chunks, vectors)


def _load_rebuild_seed_metadata(seed_path: Path) -> _SeedMetadata | None:
    persistence_paths = PersistencePath.from_path(seed_path)
    if not persistence_paths.chunk_store.exists():
        return None
    try:
        metadata = orjson.loads(persistence_paths.metadata.read_bytes())
        vectors = SelectableBasicBackend.load(persistence_paths.semantic_index).vectors
    except (OSError, ValueError, FileNotFoundError):
        return None
    file_hashes = _seed_file_hashes(metadata)
    chunk_ids = _seed_chunk_ids(metadata)
    chunk_file_paths = _metadata_chunk_file_paths(metadata)
    chunk_languages = _metadata_chunk_languages(metadata)
    if not file_hashes or chunk_ids is None or chunk_file_paths is None or chunk_languages is None:
        return None
    if (
        len(vectors) != len(chunk_ids)
        or len(chunk_file_paths) != len(chunk_ids)
        or len(chunk_languages) != len(chunk_ids)
    ):
        return None
    return _seed_metadata_from_vectors(
        metadata,
        file_hashes,
        chunk_ids,
        chunk_file_paths,
        chunk_languages,
        vectors,
        persistence_paths.chunk_store,
    )


def _rebuild_file(
    root: Path,
    file_path: Path,
    seed: _SeedData,
) -> _RebuiltFile | None:
    language = detect_language(file_path)
    try:
        relative_path = file_path.relative_to(root).as_posix()
        if get_file_status(file_path, None) != FileStatus.VALID:
            return _RebuiltFile(relative_path, [], False, None, None)
        source = read_file_text(file_path)
    except OSError:
        return None
    file_hash = _file_hash(source)
    if seed.file_hashes.get(relative_path) == file_hash:
        old_chunks = seed.chunks_by_file.get(relative_path)
        if old_chunks is None:
            return None
        return _RebuiltFile(relative_path, old_chunks, True, len(source) if old_chunks else None, file_hash)
    chunks = chunk_source(source, relative_path, language)
    return _RebuiltFile(relative_path, chunks, False, len(source) if chunks else None, file_hash if chunks else None)


def _chunk_with_id(chunk: Chunk, chunk_id: int) -> Chunk:
    return replace(chunk, chunk_id=chunk_id)


def _append_rebuilt_file(state: _RebuildState, rebuilt: _RebuiltFile, seed: _SeedData, model_dim: int) -> bool:
    if rebuilt.size is not None and rebuilt.chunks:
        state.file_sizes[rebuilt.path] = rebuilt.size
    if rebuilt.file_hash is not None and rebuilt.chunks:
        state.file_hashes[rebuilt.path] = rebuilt.file_hash
    if rebuilt.reused:
        reused_vectors = seed.vectors_by_file.get(rebuilt.path, [])
        reused_ids = seed.chunk_ids_by_file.get(rebuilt.path, [])
        if len(reused_vectors) != len(rebuilt.chunks) or len(reused_ids) != len(rebuilt.chunks):
            return False
        state.chunks.extend(_chunk_with_id(chunk, chunk_id) for chunk, chunk_id in zip(rebuilt.chunks, reused_ids))
        state.vectors.extend(reused_vectors)
        return True
    state.deleted_chunk_ids.update(seed.chunk_ids_by_file.get(rebuilt.path, ()))
    for chunk in rebuilt.chunks:
        chunk_id = state.next_chunk_id
        state.next_chunk_id += 1
        state.chunks.append(_chunk_with_id(chunk, chunk_id))
        state.changed_positions.append(len(state.vectors))
        state.vectors.append(np.empty(model_dim, dtype=np.float32))
        state.changed_chunks.append(state.chunks[-1])
    return True


def _collect_rebuild_state(
    path: Path,
    content: tuple[ContentType, ...],
    seed: _SeedData,
    model_dim: int,
) -> _RebuildState | None:
    state = _empty_rebuild_state(seed.next_chunk_id)
    for file_path in walk_files(path, get_extensions(content)):
        rebuilt = _rebuild_file(path, file_path, seed)
        if rebuilt is None or not _append_rebuilt_file(state, rebuilt, seed, model_dim):
            return None
    return state if state.chunks else None


def _seed_root_has_files(file_paths: Sequence[str], root_path: str) -> bool:
    if root_path == "":
        return bool(file_paths)
    prefix = f"{root_path}/"
    return any(file_path.startswith(prefix) for file_path in file_paths)


def _plan_heads_match(plan: GitWalkPlan, seed: _SeedData) -> bool:
    if seed.git_roots is None or plan.git_cache_metadata is None:
        return False
    current_heads = {str(item.get("path", "")): str(item.get("head", "")) for item in plan.git_cache_metadata}
    for path, head in seed.git_roots.items():
        current_head = current_heads.get(path)
        if current_head is None:
            if _seed_root_has_files(seed.file_paths, path):
                return False
            continue
        if current_head != head:
            return False
    return True


def _git_rebuild_plan(path: Path, content: tuple[ContentType, ...], seed: _SeedData) -> GitWalkPlan | None:
    if seed.git_roots is None:
        return None
    plan = build_git_walk_plan(
        path,
        get_extensions(content),
        seed.file_paths,
        previous_git_heads=seed.git_roots,
        previous_write_time=seed.write_time,
        previous_tracked_paths=seed.tracked_paths,
    )
    if plan is None or plan.stale_roots or not _plan_heads_match(plan, seed):
        return None
    return plan


def _append_reused_file(state: _RebuildState, file_path: str, seed: _SeedData) -> bool:
    chunks = seed.chunks_by_file.get(file_path)
    vectors = seed.vectors_by_file.get(file_path)
    chunk_ids = seed.chunk_ids_by_file.get(file_path)
    if chunks is None or vectors is None or chunk_ids is None:
        return False
    if len(chunks) != len(vectors) or len(chunks) != len(chunk_ids):
        return False
    state.chunks.extend(_chunk_with_id(chunk, chunk_id) for chunk, chunk_id in zip(chunks, chunk_ids))
    state.vectors.extend(vectors)
    if file_path in seed.file_sizes:
        state.file_sizes[file_path] = seed.file_sizes[file_path]
    if file_path in seed.file_hashes:
        state.file_hashes[file_path] = seed.file_hashes[file_path]
    return True


def _walk_order_key(file_path: str) -> tuple[str, ...]:
    return Path(file_path).parts


def _collect_rebuild_state_from_plan(
    path: Path,
    seed: _SeedData,
    plan: GitWalkPlan,
    model_dim: int,
) -> _RebuildState | None:
    state = _empty_rebuild_state(seed.next_chunk_id)
    for deleted_path in plan.deleted_paths:
        state.deleted_chunk_ids.update(seed.chunk_ids_by_file.get(deleted_path, ()))
    changed_paths = set(plan.changed_paths)
    for relative_path in sorted(plan.current_paths, key=_walk_order_key):
        if relative_path not in changed_paths:
            if not _append_reused_file(state, relative_path, seed):
                return None
            continue
        rebuilt = _rebuild_file(path, path / relative_path, seed)
        if rebuilt is None or not _append_rebuilt_file(state, rebuilt, seed, model_dim):
            return None
    return state if state.chunks else None


def _lazy_seed_for_plan(seed: _SeedMetadata) -> _SeedData:
    """Adapt metadata-only seed to Git plan validation without loading chunk payloads."""
    return _SeedData(
        seed.file_hashes,
        {},
        {},
        seed.chunk_ids_by_file,
        seed.file_sizes,
        seed.file_paths,
        seed.tracked_paths,
        seed.git_roots,
        seed.write_time,
        seed.next_chunk_id,
    )


def _rebuild_changed_file(path: Path, relative_path: str) -> _RebuiltFile | None:
    language = detect_language(path / relative_path)
    try:
        file_path = path / relative_path
        if get_file_status(file_path, None) != FileStatus.VALID:
            return _RebuiltFile(relative_path, [], False, None, None)
        source = read_file_text(file_path)
    except OSError:
        return None
    file_hash = _file_hash(source)
    chunks = chunk_source(source, relative_path, language)
    return _RebuiltFile(relative_path, chunks, False, len(source) if chunks else None, file_hash if chunks else None)


def _empty_lazy_rebuild_state(next_chunk_id: int) -> _LazyRebuildState:
    return _LazyRebuildState([], [], [], {}, [], {}, {}, [], [], set(), next_chunk_id)


def _append_lazy_reused_file(state: _LazyRebuildState, relative_path: str, seed: _SeedMetadata) -> bool:
    reused_ids = seed.chunk_ids_by_file.get(relative_path)
    reused_vectors = seed.vector_rows_by_file.get(relative_path)
    reused_languages = seed.languages_by_file.get(relative_path)
    if reused_ids is None or reused_vectors is None or reused_languages is None:
        return False
    if len(reused_ids) != len(reused_vectors) or len(reused_ids) != len(reused_languages):
        return False
    state.chunk_ids.extend(reused_ids)
    state.chunk_file_paths.extend([relative_path] * len(reused_ids))
    state.chunk_languages.extend(reused_languages)
    state.vectors.extend(reused_vectors)
    if relative_path in seed.file_sizes:
        state.file_sizes[relative_path] = seed.file_sizes[relative_path]
    if relative_path in seed.file_hashes:
        state.file_hashes[relative_path] = seed.file_hashes[relative_path]
    return True


def _append_lazy_changed_file(
    state: _LazyRebuildState,
    path: Path,
    relative_path: str,
    seed: _SeedMetadata,
    model_dim: int,
) -> bool:
    rebuilt = _rebuild_changed_file(path, relative_path)
    if rebuilt is None:
        return False
    state.deleted_chunk_ids.update(seed.chunk_ids_by_file.get(relative_path, ()))
    if rebuilt.size is not None and rebuilt.chunks:
        state.file_sizes[rebuilt.path] = rebuilt.size
    if rebuilt.file_hash is not None and rebuilt.chunks:
        state.file_hashes[rebuilt.path] = rebuilt.file_hash
    for chunk in rebuilt.chunks:
        chunk_id = state.next_chunk_id
        state.next_chunk_id += 1
        changed_chunk = _chunk_with_id(chunk, chunk_id)
        state.chunk_ids.append(chunk_id)
        state.chunk_file_paths.append(changed_chunk.file_path)
        state.chunk_languages.append(changed_chunk.language)
        state.changed_chunks_by_id[chunk_id] = changed_chunk
        state.changed_positions.append(len(state.vectors))
        state.vectors.append(np.empty(model_dim, dtype=np.float32))
        state.changed_chunks.append(changed_chunk)
    return True


def _collect_lazy_rebuild_state_from_plan(
    path: Path,
    seed: _SeedMetadata,
    plan: GitWalkPlan,
    model_dim: int,
) -> _LazyRebuildState | None:
    state = _empty_lazy_rebuild_state(seed.next_chunk_id)
    changed_paths = set(plan.changed_paths)
    for deleted_path in plan.deleted_paths:
        state.deleted_chunk_ids.update(seed.chunk_ids_by_file.get(deleted_path, ()))
    for relative_path in sorted(plan.current_paths, key=_walk_order_key):
        if relative_path not in changed_paths:
            if not _append_lazy_reused_file(state, relative_path, seed):
                return None
            continue
        if not _append_lazy_changed_file(state, path, relative_path, seed, model_dim):
            return None
    return state if state.chunk_ids else None


def _embed_changed_chunks(model: StaticModel, state: _RebuildState | _LazyRebuildState) -> None:
    if not state.changed_chunks:
        return
    new_vectors = embed_chunks(model, state.changed_chunks)
    for position, vector in zip(state.changed_positions, new_vectors):
        state.vectors[position] = vector


def _merged_git_cache_metadata(plan: GitWalkPlan, seed: _SeedData) -> tuple[dict[str, str], ...] | None:
    if seed.git_roots is None or plan.git_cache_metadata is None:
        return plan.git_cache_metadata
    metadata = {str(item.get("path", "")): dict(item) for item in plan.git_cache_metadata}
    for path, head in seed.git_roots.items():
        if path not in metadata and not _seed_root_has_files(seed.file_paths, path):
            metadata[path] = {"path": path, "head": head}
    return tuple(metadata[path] for path in sorted(metadata, key=lambda item: (len(item), item)))


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: StaticModel,
        bm25_index: SparseIndex | BM25,
        semantic_index: SelectableBasicBackend,
        chunks: Sequence[Chunk],
        model_path: str,
        root: Path | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        loaded_from_disk: bool = False,
    ) -> None:
        """Initialize a SembleIndex. Should be created with from_path or from_git.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        :param model_path: Path to the model file.
        :param root: Root directory used to read file sizes for token-savings stats.
        :param content: Content type used when indexing; controls the search pipeline.
        :param loaded_from_disk: Whether the index was loaded from disk (cache hit); controls CLI messaging.
        """
        self.model = model
        self.chunks: Sequence[Chunk] = chunks
        self._bm25_index: SparseIndex | BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._model_path: str = model_path
        self._root: Path | None = root
        self._content: tuple[ContentType, ...] = (content,) if isinstance(content, ContentType) else tuple(content)
        self._file_sizes: dict[str, int] = {}
        self._file_hashes: dict[str, str] = {}
        self._git_cache_metadata: tuple[dict[str, str], ...] | None = None
        self._git_cache_metadata_fresh = False
        self._tracked_paths: tuple[str, ...] | None = None
        self._file_mapping, self._language_mapping = self._populate_mapping()
        self.loaded_from_disk: bool = loaded_from_disk

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order."""
        if hasattr(self.chunks, "file_mapping") and hasattr(self.chunks, "language_mapping"):
            return self.chunks.file_mapping(), self.chunks.language_mapping()
        language_to_id = defaultdict(list)
        file_to_id = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            language = chunk.language
            if language:
                language_to_id[language].append(i)
            file_to_id[chunk.file_path].append(i)

        return dict(file_to_id), dict(language_to_id)

    def _compute_file_sizes(self, root: Path) -> dict[str, int]:
        """Return a mapping of repo-relative file path to total character count."""
        sizes: dict[str, int] = {}
        for chunk in self.chunks:
            if chunk.file_path in sizes:
                continue
            try:
                sizes[chunk.file_path] = len(read_file_text(root / chunk.file_path))
            except OSError:
                pass
        return sizes

    @property
    def stats(self) -> IndexStats:
        """Stats of an index."""
        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks),
            languages={language: len(chunk_ids) for language, chunk_ids in self._language_mapping.items()},
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
        model_path: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param content: Content types to index, e.g. ContentType.CODE or [ContentType.CODE, ContentType.DOCS].
        :param include_text_files: Deprecated. Pass a content sequence directly instead.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        normalized = _apply_include_text_files(content, include_text_files)
        cache_path = get_validated_cache(str(path), model_path, normalized)
        if cache_path:
            return cls.load_from_disk(cache_path)
        model, model_path = load_model(model_path)

        path = path.resolve()
        seed_path = get_rebuild_cache(str(path), model_path, normalized)
        if seed_path is not None:
            seeded = cls._try_rebuild_from_cache(path, seed_path, model, model_path, normalized)
            if seeded is not None:
                return seeded
        build = create_index_build_from_path(
            path,
            model=model,
            content=normalized,
            display_root=path,
        )

        index = SembleIndex(
            model,
            build.sparse_index,
            build.semantic_index,
            build.chunks,
            model_path,
            root=path,
            content=normalized,
        )
        index._file_sizes = build.file_sizes
        index._file_hashes = build.file_hashes
        index._git_cache_metadata = build.git_cache_metadata
        index._git_cache_metadata_fresh = build.git_cache_metadata is not None
        index._tracked_paths = build.tracked_paths
        return index

    @classmethod
    def _try_rebuild_from_cache(
        cls,
        path: Path,
        seed_path: Path,
        model: StaticModel,
        model_path: str,
        content: tuple[ContentType, ...],
    ) -> SembleIndex | None:
        """Rebuild changed worktrees by reusing unchanged chunks and dense rows."""
        seed_metadata = _load_rebuild_seed_metadata(seed_path)
        if seed_metadata is None:
            lazy_plan = None
        else:
            lazy_plan = _git_rebuild_plan(path, content, _lazy_seed_for_plan(seed_metadata))
        if seed_metadata is not None and lazy_plan is not None:
            state = _collect_lazy_rebuild_state_from_plan(path, seed_metadata, lazy_plan, model.dim)
            if state is None:
                return None
            _embed_changed_chunks(model, state)
            embeddings = np.asarray(state.vectors, dtype=np.float32)
            semantic_backend = SelectableBasicBackend(embeddings, BasicArgs())
            semantic_index = _StableIdSemanticBackend(semantic_backend, state.chunk_ids)
            chunks = MergedLazyChunkList(
                state.chunk_ids,
                seed_metadata.chunk_store_path,
                state.chunk_file_paths,
                state.chunk_languages,
                state.changed_chunks_by_id,
                state.deleted_chunk_ids,
            )
            persistence_paths = PersistencePath.from_path(seed_path)
            sparse_index = TantivySparseIndex.load_copy_from_store(
                persistence_paths.bm25_index,
                seed_metadata.chunk_store_path,
            )
            sparse_index.update_chunks(chunks, state.deleted_chunk_ids, state.changed_chunks)
            index = SembleIndex(model, sparse_index, semantic_index, chunks, model_path, root=path, content=content)
            index._file_sizes = state.file_sizes
            index._file_hashes = state.file_hashes
            index._git_cache_metadata = _merged_git_cache_metadata(lazy_plan, _lazy_seed_for_plan(seed_metadata))
            index._git_cache_metadata_fresh = index._git_cache_metadata is not None
            index._tracked_paths = tuple(sorted(lazy_plan.tracked_paths))
            return index

        seed = _load_rebuild_seed(seed_path)
        if seed is None:
            return None
        plan = _git_rebuild_plan(path, content, seed)
        if plan is None:
            state = _collect_rebuild_state(path, content, seed, model.dim)
        else:
            state = _collect_rebuild_state_from_plan(path, seed, plan, model.dim)
        if state is None:
            return None
        _embed_changed_chunks(model, state)
        embeddings = np.asarray(state.vectors, dtype=np.float32)
        semantic_backend = SelectableBasicBackend(embeddings, BasicArgs())
        chunk_ids = [
            chunk.chunk_id if chunk.chunk_id is not None else index for index, chunk in enumerate(state.chunks)
        ]
        semantic_index = _StableIdSemanticBackend(semantic_backend, chunk_ids)
        if plan is None:
            sparse_index = TantivySparseIndex.build_temporary(state.chunks)
        else:
            persistence_paths = PersistencePath.from_path(seed_path)
            sparse_index = TantivySparseIndex.load_copy(persistence_paths.bm25_index, state.chunks)
            sparse_index.update_chunks(state.chunks, state.deleted_chunk_ids, state.changed_chunks)
        index = SembleIndex(model, sparse_index, semantic_index, state.chunks, model_path, root=path, content=content)
        index._file_sizes = state.file_sizes
        index._file_hashes = state.file_hashes
        if plan is not None:
            index._git_cache_metadata = _merged_git_cache_metadata(plan, seed)
            index._git_cache_metadata_fresh = index._git_cache_metadata is not None
            index._tracked_paths = tuple(sorted(plan.tracked_paths))
        return index

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model_path: str | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
    ) -> SembleIndex:
        """Clone a git repository and index it.

        The repository is cloned into a temporary directory that is removed once
        indexing finishes. Chunk content is preserved in-memory, but
        chunk.file_path will not point to a readable file after this call
        returns — it is a repo-relative label, not a filesystem path.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :param content: Content types to index, e.g. (ContentType.CODE,) or (ContentType.CODE, ContentType.DOCS).
        :param include_text_files: Deprecated. Pass content=(ContentType.CODE, ContentType.DOCS, ...) instead.
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH, the clone fails, or times out.
        """
        normalized = _apply_include_text_files(content, include_text_files)
        cache_key = f"{url}@{ref}" if ref else url
        cache_path = get_validated_cache(cache_key, model_path, normalized)
        if cache_path and _cache_is_loadable_for_git(cache_path):
            return cls.load_from_disk(cache_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=_GIT_CLONE_TIMEOUT
                )
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"git clone timed out for {url!r} (limit: {_GIT_CLONE_TIMEOUT} s)") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")

            model, model_path = load_model(model_path)
            resolved_path = Path(tmp_dir).resolve()
            build = create_index_build_from_path(
                resolved_path,
                model=model,
                content=normalized,
                display_root=resolved_path,
            )

            index = SembleIndex(
                model,
                build.sparse_index,
                build.semantic_index,
                build.chunks,
                model_path,
                root=resolved_path,
                content=normalized,
            )
            index._file_sizes = build.file_sizes
            index._file_hashes = build.file_hashes
            index._git_cache_metadata = build.git_cache_metadata
            index._git_cache_metadata_fresh = build.git_cache_metadata is not None
            index._tracked_paths = build.tracked_paths
            return index

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        filter_spec = FilterSpec(languages=frozenset([target.language])) if target.language else None
        results = _search_semantic(
            target.content, self.model, self._semantic_index, self.chunks, top_k + 1, filter_spec
        )
        results = [r for r in results if r.chunk != target][:top_k]
        save_search_stats(results, CallType.FIND_RELATED, self._file_sizes)
        return results

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to."""
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        return np.unique(selector) if selector else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        rerank: bool | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. None auto-detects from query type.
        :param filter_languages: Optional list of language codes; if set, only chunks in
            these languages are returned.
        :param filter_paths: Optional list of repo-relative file paths; if set, only
            chunks from these files are returned.
        :param rerank: Apply code-tuned reranking (file boost, identifier boost, path penalties).
            Defaults to True when ContentType.CODE was indexed.
        :return: Ranked list of SearchResult objects, best match first.
        """
        if not self.chunks or not query.strip():
            return []

        resolved_rerank = (ContentType.CODE in self._content) if rerank is None else rerank

        filter_spec = None
        if filter_languages or filter_paths:
            filter_spec = FilterSpec(
                file_paths=frozenset(filter_paths) if filter_paths else None,
                languages=frozenset(filter_languages) if filter_languages else None,
            )
        results = search(
            query,
            self.model,
            self._semantic_index,
            self._bm25_index,
            self.chunks,
            top_k,
            alpha=alpha,
            filter_spec=filter_spec,
            rerank=resolved_rerank,
        )
        save_search_stats(results, CallType.SEARCH, self._file_sizes)
        return results

    @classmethod
    def load_from_disk(cls: type[SembleIndex], path: Path | str) -> SembleIndex:
        """Load the index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        persistence_paths = PersistencePath.from_path(path)
        non_existent = persistence_paths.non_existing()
        if non_existent:
            missing = ", ".join(str(p) for p in non_existent)
            raise FileNotFoundError(f"Index not found at {path}. Missing: {missing}")

        semantic_index = SelectableBasicBackend.load(persistence_paths.semantic_index)
        with open(persistence_paths.metadata, "rb") as f:
            metadata = orjson.loads(f.read())

        chunk_ids = metadata.get("chunk_ids")
        if chunk_ids is not None:
            ids = [int(chunk_id) for chunk_id in chunk_ids]
            semantic_index = _StableIdSemanticBackend(semantic_index, ids)
            if metadata.get("sparse_backend") == "tantivy":
                chunks = LazyChunkList(
                    ids,
                    persistence_paths.chunk_store,
                    metadata.get("chunk_file_paths", []),
                    metadata.get("chunk_languages", []),
                )
                bm_25_index: SparseIndex | BM25 = TantivySparseIndex.load_from_store(
                    persistence_paths.bm25_index,
                    persistence_paths.chunk_store,
                )
            else:
                bm_25_index = BM25.load(persistence_paths.bm25_index)
                store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
                try:
                    chunks = store.get_chunks(ids)
                finally:
                    store.close()
                if len(chunks) != len(ids):
                    raise FileNotFoundError("Index chunk store is missing chunk payloads")
        else:
            bm_25_index = BM25.load(persistence_paths.bm25_index)
            with open(persistence_paths.chunks, "r") as f:
                chunk_data = orjson.loads(f.read())
            chunks = [Chunk.from_dict(chunk_item) for chunk_item in chunk_data]
        root_path = metadata["root_path"]
        model_path = metadata["model_path"]
        content = tuple(ContentType(s) for s in metadata.get("content_type", ["code"]))
        if root_path:
            root_path = Path(root_path)

        model, model_path = load_model(model_path)

        index = cls(
            model,
            bm_25_index,
            semantic_index,
            chunks,
            model_path,
            root=root_path,
            content=content,
            loaded_from_disk=True,
        )
        index._file_sizes = {str(key): int(value) for key, value in metadata.get("file_sizes", {}).items()}
        index._file_hashes = _seed_file_hashes(metadata)
        tracked_paths = _seed_tracked_paths(metadata)
        index._tracked_paths = None if tracked_paths is None else tuple(tracked_paths)
        return index

    def save(self, path: Path | str) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = path.parent / f"{path.name}.tmp-{uuid.uuid4().hex}"
        backup_path = path.parent / f"{path.name}.old-{uuid.uuid4().hex}"

        try:
            self._write_staged_index(staging_path)
            self._replace_saved_index(path, staging_path, backup_path)
        except Exception:
            self._restore_failed_save(path, staging_path, backup_path)
            raise
        if backup_path.exists():
            shutil.rmtree(backup_path)

    def _write_staged_index(self, staging_path: Path) -> None:
        """Write a complete index into a staging directory."""
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(parents=True)
        persistence_paths = PersistencePath.from_path(staging_path)

        if self._root is not None and not self._file_sizes:
            self._file_sizes = self._compute_file_sizes(self._root)
        self._bm25_index.save(persistence_paths.bm25_index)
        self._semantic_index.save(persistence_paths.semantic_index)
        chunk_ids = self._chunk_ids_for_save()
        chunk_store_id = self._write_chunk_store(persistence_paths, chunk_ids)
        metadata = self._metadata_for_save(chunk_ids, chunk_store_id)
        with open(persistence_paths.metadata, "wb") as f:
            data = orjson.dumps(metadata)
            f.write(data)

    def _chunk_ids_for_save(self) -> list[int]:
        """Return stable chunk IDs without forcing lazy chunk payload loads."""
        chunk_ids = getattr(self.chunks, "chunk_ids", None)
        if chunk_ids is not None:
            return list(chunk_ids())
        return [chunk.chunk_id if chunk.chunk_id is not None else index for index, chunk in enumerate(self.chunks)]

    def _write_chunk_store(self, persistence_paths: PersistencePath, chunk_ids: list[int]) -> str:
        """Write LMDB chunks and return the store identifier bound to metadata."""
        chunk_store_id = uuid.uuid4().hex
        copy_chunk_store_to = getattr(self.chunks, "copy_chunk_store_to", None)
        if copy_chunk_store_to is not None:
            copy_chunk_store_to(persistence_paths.chunk_store, chunk_ids)
            store = LmdbChunkStore.open(persistence_paths.chunk_store)
            try:
                store.write_store_id(chunk_store_id)
            finally:
                store.close()
            return chunk_store_id

        store = LmdbChunkStore.open(persistence_paths.chunk_store)
        try:
            store.write_chunks_with_ids(self.chunks, chunk_ids)
            store.write_store_id(chunk_store_id)
        finally:
            store.close()
        return chunk_store_id

    def _chunk_file_paths_for_save(self) -> list[str]:
        chunk_file_paths = getattr(self.chunks, "chunk_file_paths", None)
        if chunk_file_paths is not None:
            return list(chunk_file_paths)
        return [chunk.file_path for chunk in self.chunks]

    def _chunk_languages_for_save(self) -> list[str | None]:
        chunk_languages = getattr(self.chunks, "chunk_languages", None)
        if chunk_languages is not None:
            return list(chunk_languages)
        return [chunk.language for chunk in self.chunks]

    def _metadata_for_save(self, chunk_ids: list[int], chunk_store_id: str) -> dict[str, object]:
        """Build metadata for a staged full-save index."""
        root_str = None if self._root is None else str(self._root)
        file_paths = sorted(self._file_mapping)
        metadata: dict[str, object] = {
            "root_path": root_str,
            "time": datetime.now().timestamp(),
            "model_path": self._model_path,
            "content_type": list(x.value for x in self._content),
            "file_paths": file_paths,
            "chunk_ids": chunk_ids,
            "chunk_file_paths": self._chunk_file_paths_for_save(),
            "chunk_languages": self._chunk_languages_for_save(),
            "chunk_store_id": chunk_store_id,
            "file_sizes": self._file_sizes,
            "file_hashes": self._file_hashes,
            "rebuild_cache_version": _REBUILD_CACHE_VERSION,
            "sparse_backend": "tantivy" if isinstance(self._bm25_index, TantivySparseIndex) else "bm25s",
        }
        if self._root is not None:
            save_metadata = None
            git_roots = self._git_cache_metadata
            if git_roots is None:
                save_metadata = build_git_cache_save_metadata(self._root, file_paths)
            elif not self._git_cache_metadata_fresh:
                refreshed_git_roots = refresh_git_cache_metadata(self._root, git_roots)
                if refreshed_git_roots is not None:
                    git_roots = tuple(refreshed_git_roots)
            tracked_paths = self._tracked_paths
            if save_metadata is not None:
                git_roots, tracked_paths = save_metadata
            if git_roots is not None:
                metadata["git_roots"] = list(git_roots)
                metadata["git_roots_version"] = GIT_CACHE_ROOTS_VERSION
            if tracked_paths is not None:
                metadata["tracked_paths"] = sorted(tracked_paths)
        return metadata

    def _replace_saved_index(self, path: Path, staging_path: Path, backup_path: Path) -> None:
        """Replace an existing index with a complete staged index."""
        if path.exists():
            path.rename(backup_path)
        staging_path.rename(path)

    def _restore_failed_save(self, path: Path, staging_path: Path, backup_path: Path) -> None:
        """Remove incomplete staged data and restore the previous index if needed."""
        if staging_path.exists():
            shutil.rmtree(staging_path)
        if backup_path.exists() and not path.exists():
            backup_path.rename(path)
