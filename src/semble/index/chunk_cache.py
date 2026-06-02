from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import orjson

from semble.index.chunk_store import LmdbChunkStore
from semble.index.turbovec import TurboVecBasicBackend
from semble.index.types import PersistencePath
from semble.types import Chunk

_CHUNK_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChunkCacheEntry:
    file_path: str
    file_hash: str
    file_size: int
    file_mtime_ns: int
    embedding_key: str
    bm25_key: str
    bm25_document: list[str] | None
    language: str | None = None
    source_root: str = ""
    git_path: str | None = None
    tracked: bool = True
    chunk_id: int | None = None
    reuses_cache: bool = False
    cache_store_path: Path | None = None
    reuses_embedding: bool = False
    external_embedding_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the cache entry to a serializable dict."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "file_mtime_ns": self.file_mtime_ns,
            "embedding_key": self.embedding_key,
            "bm25_key": self.bm25_key,
            "bm25_document": self.bm25_document,
            "language": self.language,
            "source_root": self.source_root,
            "git_path": self.git_path,
            "tracked": self.tracked,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkCacheEntry":
        """Create a cache entry from persisted data."""
        git_path = data.get("git_path")
        bm25_document = data.get("bm25_document")
        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            file_size=data.get("file_size", -1),
            file_mtime_ns=data.get("file_mtime_ns", -1),
            embedding_key=data["embedding_key"],
            bm25_key=data["bm25_key"],
            bm25_document=None if bm25_document is None else list(bm25_document),
            language=None if data.get("language") is None else str(data["language"]),
            source_root=str(data.get("source_root", "")),
            git_path=None if git_path is None else str(git_path),
            tracked=bool(data.get("tracked", True)),
            chunk_id=None if data.get("chunk_id") is None else int(data["chunk_id"]),
        )


@dataclass(frozen=True, slots=True)
class CachedChunk:
    chunk: Chunk
    file_hash: str
    file_size: int
    file_mtime_ns: int
    embedding: npt.NDArray[np.float32]
    bm25_document: list[str] | None


@dataclass(frozen=True, slots=True)
class LoadedChunkCache:
    chunks_by_file: dict[tuple[str, str], list[CachedChunk]]
    chunks_by_file_metadata: dict[tuple[str, int, int], list[CachedChunk]]
    embeddings_by_key: dict[str, npt.NDArray[np.float32]]
    bm25_documents_by_key: dict[str, list[str]]
    next_chunk_id: int


EMPTY_CHUNK_CACHE = LoadedChunkCache({}, {}, {}, {}, 0)


def _load_chunks(persistence_paths: PersistencePath, metadata: dict[str, Any]) -> list[Chunk]:
    chunk_ids = metadata.get("chunk_ids")
    if chunk_ids is not None:
        if not persistence_paths.chunk_store.exists():
            return []
        ids = [int(chunk_id) for chunk_id in chunk_ids]
        store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
        try:
            chunks = store.get_chunks(ids)
        finally:
            store.close()
        return chunks if len(chunks) == len(ids) else []

    if not persistence_paths.chunks.exists():
        return []
    chunk_data = orjson.loads(persistence_paths.chunks.read_bytes())
    return [Chunk.from_dict(item) for item in chunk_data]


def _load_metadata(persistence_paths: PersistencePath) -> dict[str, Any]:
    if not persistence_paths.metadata.exists():
        return {}
    return orjson.loads(persistence_paths.metadata.read_bytes())


def _has_chunk_storage(persistence_paths: PersistencePath) -> bool:
    return persistence_paths.chunks.exists() or persistence_paths.chunk_store.exists()


def load_index_chunks(index_path: Path | None) -> list[Chunk]:
    """Load persisted chunk payloads from an index cache."""
    if index_path is None:
        return []
    persistence_paths = PersistencePath.from_path(index_path)
    if not _has_chunk_storage(persistence_paths):
        return []
    return _load_chunks(persistence_paths, _load_metadata(persistence_paths))


def _load_lmdb_chunk_cache(persistence_paths: PersistencePath, chunks: list[Chunk]) -> LoadedChunkCache | None:
    if not persistence_paths.chunk_store.exists():
        return None

    store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
    try:
        manifests = {manifest.file_path: manifest for manifest in store.iter_file_manifests()}
        if not manifests:
            return None

        chunks_by_file: dict[tuple[str, str], list[CachedChunk]] = {}
        chunks_by_file_metadata: dict[tuple[str, int, int], list[CachedChunk]] = {}
        embeddings_by_key: dict[str, npt.NDArray[np.float32]] = {}
        bm25_documents_by_key: dict[str, list[str]] = {}

        for chunk in chunks:
            manifest = manifests.get(chunk.file_path)
            embedding_key_value = embedding_cache_key(chunk)
            bm25_key_value = bm25_cache_key(chunk)
            embedding = store.get_embedding(embedding_key_value)
            bm25_document = store.get_bm25_document(bm25_key_value)
            if manifest is None or embedding is None:
                return None

            cached_chunk = CachedChunk(
                chunk,
                manifest.file_hash,
                manifest.file_size,
                manifest.file_mtime_ns,
                embedding,
                bm25_document,
            )
            embeddings_by_key.setdefault(embedding_key_value, embedding)
            if bm25_document is not None:
                bm25_documents_by_key.setdefault(bm25_key_value, bm25_document)
            chunks_by_file.setdefault((manifest.file_path, manifest.file_hash), []).append(cached_chunk)
            chunks_by_file_metadata.setdefault(
                (manifest.file_path, manifest.file_size, manifest.file_mtime_ns),
                [],
            ).append(cached_chunk)

        return LoadedChunkCache(
            chunks_by_file,
            chunks_by_file_metadata,
            embeddings_by_key,
            bm25_documents_by_key,
            store.next_chunk_id(),
        )
    finally:
        store.close()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def file_cache_key(source: str) -> str:
    """Return the content hash used to identify unchanged files."""
    return _hash_text(source)


def embedding_cache_key(chunk: Chunk) -> str:
    """Return the cache key for a chunk embedding."""
    return _hash_text(f"embedding:v1:{chunk.language or ''}\0{chunk.content}")


def bm25_cache_key(chunk: Chunk) -> str:
    """Return the cache key for a tokenized BM25 document."""
    return _hash_text(f"bm25:v1:{chunk.file_path}\0{chunk.language or ''}\0{chunk.content}")


def make_chunk_cache_entry(
    chunk: Chunk,
    file_hash: str,
    file_size: int,
    file_mtime_ns: int,
    bm25_document: list[str],
    source_root: str = "",
    git_path: str | None = None,
    tracked: bool = True,
    reuses_cache: bool = False,
    cache_store_path: Path | None = None,
) -> ChunkCacheEntry:
    """Create one cache entry for a built chunk."""
    return ChunkCacheEntry(
        file_path=chunk.file_path,
        file_hash=file_hash,
        file_size=file_size,
        file_mtime_ns=file_mtime_ns,
        embedding_key=embedding_cache_key(chunk),
        bm25_key=bm25_cache_key(chunk),
        bm25_document=bm25_document,
        language=chunk.language,
        source_root=source_root,
        git_path=git_path,
        tracked=tracked,
        chunk_id=chunk.chunk_id,
        reuses_cache=reuses_cache,
        cache_store_path=cache_store_path,
    )


def load_chunk_cache_entries(path: Path) -> list[ChunkCacheEntry]:
    """Load persisted chunk cache entries, ignoring incompatible versions."""
    if not path.exists():
        return []
    data = orjson.loads(path.read_bytes())
    if data.get("version") != _CHUNK_CACHE_VERSION:
        return []
    return [ChunkCacheEntry.from_dict(item) for item in data.get("entries", [])]


def load_lmdb_chunk_cache_entries(persistence_paths: PersistencePath, chunks: list[Chunk]) -> list[ChunkCacheEntry]:
    """Load per-chunk rebuild cache metadata from LMDB."""
    if not persistence_paths.chunk_store.exists():
        return []

    store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
    try:
        entries: list[ChunkCacheEntry] = []
        for chunk in chunks:
            manifest = store.get_file_manifest(chunk.file_path)
            bm25_key_value = bm25_cache_key(chunk)
            bm25_document = store.get_bm25_document(bm25_key_value)
            if manifest is None:
                return []
            entries.append(
                ChunkCacheEntry(
                    file_path=chunk.file_path,
                    file_hash=manifest.file_hash,
                    file_size=manifest.file_size,
                    file_mtime_ns=manifest.file_mtime_ns,
                    embedding_key=embedding_cache_key(chunk),
                    bm25_key=bm25_key_value,
                    bm25_document=bm25_document,
                    language=chunk.language,
                    source_root=manifest.source_root,
                    git_path=manifest.git_path,
                    tracked=manifest.tracked,
                    chunk_id=chunk.chunk_id,
                )
            )
        return entries
    finally:
        store.close()


def save_chunk_cache_entries(path: Path, entries: list[ChunkCacheEntry]) -> None:
    """Persist chunk cache entries beside an index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _CHUNK_CACHE_VERSION,
        "entries": [entry.to_dict() for entry in entries],
    }
    path.write_bytes(orjson.dumps(data))


def _load_json_chunk_cache(persistence_paths: PersistencePath, chunks: list[Chunk]) -> LoadedChunkCache:
    if not persistence_paths.semantic_index.exists():
        return EMPTY_CHUNK_CACHE
    semantic_index = TurboVecBasicBackend.load(persistence_paths.semantic_index)
    vectors = semantic_index.vectors
    if len(chunks) != len(vectors):
        return EMPTY_CHUNK_CACHE

    entries = load_chunk_cache_entries(persistence_paths.chunk_cache)
    if len(entries) != len(chunks):
        entries = []
    next_chunk_id = max((chunk.chunk_id for chunk in chunks if chunk.chunk_id is not None), default=-1) + 1

    chunks_by_file: dict[tuple[str, str], list[CachedChunk]] = {}
    chunks_by_file_metadata: dict[tuple[str, int, int], list[CachedChunk]] = {}
    embeddings_by_key: dict[str, npt.NDArray[np.float32]] = {}
    bm25_documents_by_key: dict[str, list[str]] = {}

    for index, chunk in enumerate(chunks):
        vector = vectors[index]
        entry = entries[index] if entries else None
        embeddings_by_key.setdefault(entry.embedding_key if entry else embedding_cache_key(chunk), vector)
        if entry is None:
            continue
        cached_chunk = CachedChunk(
            chunk,
            entry.file_hash,
            entry.file_size,
            entry.file_mtime_ns,
            vector,
            entry.bm25_document,
        )
        if entry.bm25_document is not None:
            bm25_documents_by_key.setdefault(entry.bm25_key, entry.bm25_document)
        chunks_by_file.setdefault((entry.file_path, entry.file_hash), []).append(cached_chunk)
        if entry.file_size >= 0 and entry.file_mtime_ns >= 0:
            chunks_by_file_metadata.setdefault((entry.file_path, entry.file_size, entry.file_mtime_ns), []).append(
                cached_chunk
            )

    return LoadedChunkCache(
        chunks_by_file,
        chunks_by_file_metadata,
        embeddings_by_key,
        bm25_documents_by_key,
        next_chunk_id,
    )


def load_chunk_cache(index_path: Path | None) -> LoadedChunkCache:
    """Load reusable chunk work from a persisted sparse index."""
    if index_path is None:
        return EMPTY_CHUNK_CACHE

    persistence_paths = PersistencePath.from_path(index_path)
    if not _has_chunk_storage(persistence_paths):
        return EMPTY_CHUNK_CACHE

    metadata = _load_metadata(persistence_paths)
    chunks = _load_chunks(persistence_paths, metadata)
    if not chunks:
        return EMPTY_CHUNK_CACHE

    lmdb_cache = _load_lmdb_chunk_cache(persistence_paths, chunks)
    if lmdb_cache is not None:
        return lmdb_cache

    return _load_json_chunk_cache(persistence_paths, chunks)
