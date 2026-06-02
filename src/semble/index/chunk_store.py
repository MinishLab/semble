from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lmdb
import msgspec
import numpy as np
import numpy.typing as npt
import orjson

from semble.types import Chunk

_CHUNKS_DB = b"chunks"
_FILES_DB = b"files"
_EMBEDDINGS_DB = b"embeddings"
_BM25_DOCS_DB = b"bm25_docs"
_QUERY_EMBEDDINGS_DB = b"query_embeddings"
_FILE_HASHES_DB = b"file_hashes"
_META_DB = b"meta"
_GENERATIONS_DB = b"generations"
_NEXT_CHUNK_ID = b"next_chunk_id"
_ACTIVE_GENERATION = b"active_generation"
_PENDING_GENERATION = b"pending_generation"
_ACTIVE_GENERATION_SNAPSHOT_ID = b"active_generation_snapshot_id"
_DEFAULT_MAP_SIZE = 4 * 1024 * 1024 * 1024
_CHUNK_PAYLOAD_V1 = b"\x01"
_ChunkPayload = tuple[str, str, int, int, str | None, int | None]


def _chunk_payload(chunk: Chunk) -> _ChunkPayload:
    return (
        chunk.content,
        chunk.file_path,
        chunk.start_line,
        chunk.end_line,
        chunk.language,
        chunk.chunk_id,
    )


def _serialize_chunk(chunk: Chunk) -> bytes:
    return _CHUNK_PAYLOAD_V1 + msgspec.msgpack.encode(_chunk_payload(chunk))


def _deserialize_chunk(data: bytes) -> Chunk:
    if data.startswith(_CHUNK_PAYLOAD_V1):
        content, file_path, start_line, end_line, language, chunk_id = msgspec.msgpack.decode(
            data[len(_CHUNK_PAYLOAD_V1) :]
        )
        return Chunk(
            str(content),
            str(file_path),
            int(start_line),
            int(end_line),
            None if language is None else str(language),
            None if chunk_id is None else int(chunk_id),
        )
    return Chunk.from_dict(orjson.loads(data))


@dataclass(frozen=True, slots=True)
class FileManifest:
    """Persisted metadata for one indexed file."""

    file_path: str
    file_hash: str
    file_size: int
    file_mtime_ns: int
    chunk_ids: list[int]
    language: str | None = None
    generation: int = 0
    source_root: str = ""
    git_path: str | None = None
    tracked: bool = True
    chunk_cache_keys: list[dict[str, str | bool]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest to a serializable dict."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "file_mtime_ns": self.file_mtime_ns,
            "chunk_ids": self.chunk_ids,
            "language": self.language,
            "generation": self.generation,
            "source_root": self.source_root,
            "git_path": self.git_path,
            "tracked": self.tracked,
            "chunk_cache_keys": self.chunk_cache_keys,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileManifest":
        """Create a manifest from persisted data."""
        file_path = str(data["file_path"])
        git_path = data.get("git_path", file_path)
        chunk_cache_keys = data.get("chunk_cache_keys", [])
        if not isinstance(chunk_cache_keys, list):
            chunk_cache_keys = []
        return cls(
            file_path=file_path,
            file_hash=str(data["file_hash"]),
            file_size=int(data["file_size"]),
            file_mtime_ns=int(data["file_mtime_ns"]),
            chunk_ids=[int(chunk_id) for chunk_id in data["chunk_ids"]],
            language=None if data.get("language") is None else str(data["language"]),
            generation=int(data.get("generation", 0)),
            source_root=str(data.get("source_root", "")),
            git_path=None if git_path is None else str(git_path),
            tracked=bool(data.get("tracked", True)),
            chunk_cache_keys=[
                {
                    key: value
                    for key, value in {
                        "embedding_key": str(item["embedding_key"]),
                        "bm25_key": str(item["bm25_key"]),
                        "reuses_embedding": True if item.get("reuses_embedding", False) else None,
                        "external_embedding_key": (
                            None if item.get("external_embedding_key") is None else str(item["external_embedding_key"])
                        ),
                    }.items()
                    if value is not None
                }
                for item in chunk_cache_keys
                if isinstance(item, dict) and "embedding_key" in item and "bm25_key" in item
            ],
        )


@dataclass(slots=True)
class LmdbChunkStore:
    """LMDB-backed store for chunk payloads and file manifests."""

    env: lmdb.Environment
    path: Path
    chunks_db: lmdb._Database
    files_db: lmdb._Database
    embeddings_db: lmdb._Database | None
    bm25_docs_db: lmdb._Database | None
    query_embeddings_db: lmdb._Database | None
    file_hashes_db: lmdb._Database | None
    meta_db: lmdb._Database
    generations_db: lmdb._Database | None

    @classmethod
    def open(cls, path: Path, *, readonly: bool = False, map_size: int = _DEFAULT_MAP_SIZE) -> "LmdbChunkStore":
        """Open a chunk store rooted at path."""
        if not readonly:
            path.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            str(path),
            map_size=map_size,
            max_dbs=8,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            create=not readonly,
        )
        return cls(
            env,
            path,
            env.open_db(_CHUNKS_DB),
            env.open_db(_FILES_DB),
            _open_db(env, _EMBEDDINGS_DB, readonly),
            _open_db(env, _BM25_DOCS_DB, readonly),
            _open_db(env, _QUERY_EMBEDDINGS_DB, readonly),
            _open_db(env, _FILE_HASHES_DB, readonly),
            env.open_db(_META_DB),
            _open_db(env, _GENERATIONS_DB, readonly),
        )

    def close(self) -> None:
        """Close the underlying LMDB environment."""
        self.env.close()

    def write_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks keyed by stable chunk_id."""
        self._write_keyed_chunks((chunk, chunk.chunk_id) for chunk in chunks)

    def write_chunks_with_ids(self, chunks: Sequence[Chunk], chunk_ids: Sequence[int]) -> None:
        """Persist chunks keyed by supplied IDs without changing chunk payloads."""
        self._write_keyed_chunks(zip(chunks, chunk_ids))

    def _write_keyed_chunks(self, keyed_chunks: Iterator[tuple[Chunk, int | None]]) -> None:
        with self.env.begin(write=True) as txn:
            next_chunk_id = self.next_chunk_id()
            for chunk, chunk_id in keyed_chunks:
                if chunk_id is None:
                    continue
                txn.put(_int_key(chunk_id), _serialize_chunk(chunk), db=self.chunks_db)
                next_chunk_id = max(next_chunk_id, chunk_id + 1)
            txn.put(_NEXT_CHUNK_ID, str(next_chunk_id).encode(), db=self.meta_db)

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        """Return one chunk by stable chunk_id."""
        with self.env.begin(buffers=True) as txn:
            data = txn.get(_int_key(chunk_id), db=self.chunks_db)
            if data is None:
                return None
            return _deserialize_chunk(bytes(data))

    def get_chunks(self, chunk_ids: list[int]) -> list[Chunk]:
        """Return chunks by stable chunk_id, preserving requested order."""
        chunks = []
        with self.env.begin(buffers=True) as txn:
            for chunk_id in chunk_ids:
                data = txn.get(_int_key(chunk_id), db=self.chunks_db)
                if data is not None:
                    chunks.append(_deserialize_chunk(bytes(data)))
        return chunks

    def copy_chunks_from(self, source_path: Path, chunk_ids: list[int]) -> None:
        """Copy raw chunk payloads from another LMDB store without decoding them."""
        source = LmdbChunkStore.open(source_path, readonly=True)
        try:
            next_chunk_id = self.next_chunk_id()
            with source.env.begin(buffers=True) as source_txn, self.env.begin(write=True) as target_txn:
                for chunk_id in chunk_ids:
                    data = source_txn.get(_int_key(chunk_id), db=source.chunks_db)
                    if data is None:
                        raise FileNotFoundError("Index chunk store is missing chunk payloads")
                    target_txn.put(_int_key(chunk_id), bytes(data), db=self.chunks_db)
                    next_chunk_id = max(next_chunk_id, chunk_id + 1)
                target_txn.put(_NEXT_CHUNK_ID, str(next_chunk_id).encode(), db=self.meta_db)
        finally:
            source.close()

    def copy_rebuild_cache_from(self, source_path: Path, cache_keys: list[dict[str, str]]) -> None:
        """Copy raw rebuild-cache payloads from another LMDB store without decoding them."""
        source = LmdbChunkStore.open(source_path, readonly=True)
        try:
            if source.embeddings_db is None or source.bm25_docs_db is None:
                raise FileNotFoundError("Index chunk store is missing rebuild cache payloads")
            if self.embeddings_db is None or self.bm25_docs_db is None:
                raise RuntimeError("Rebuild cache store is not available")
            with source.env.begin(buffers=True) as source_txn, self.env.begin(write=True) as target_txn:
                for keys in cache_keys:
                    embedding_data = source_txn.get(str(keys["embedding_key"]).encode(), db=source.embeddings_db)
                    if embedding_data is None:
                        if not keys.get("reuses_embedding"):
                            raise FileNotFoundError("Index chunk store is missing rebuild cache payloads")
                    else:
                        target_txn.put(
                            str(keys["embedding_key"]).encode(),
                            bytes(embedding_data),
                            db=self.embeddings_db,
                        )
                    bm25_data = source_txn.get(str(keys["bm25_key"]).encode(), db=source.bm25_docs_db)
                    if bm25_data is not None:
                        target_txn.put(str(keys["bm25_key"]).encode(), bytes(bm25_data), db=self.bm25_docs_db)
        finally:
            source.close()

    def _manifest_for_generation(self, data: bytes, generation: int) -> FileManifest:
        payload = FileManifest.from_dict(orjson.loads(data)).to_dict()
        payload["generation"] = generation
        return FileManifest.from_dict(payload)

    def copy_file_manifests_from(self, source_path: Path, file_paths: list[str], generation: int) -> None:
        """Copy file manifests from another LMDB store while assigning a new generation."""
        if source_path.resolve() == self.path.resolve():
            with self.env.begin(buffers=True) as source_txn:
                manifests = []
                for file_path in file_paths:
                    data = source_txn.get(file_path.encode(), db=self.files_db)
                    if data is None:
                        raise FileNotFoundError("Index chunk store is missing file manifests")
                    manifests.append((file_path, self._manifest_for_generation(bytes(data), generation)))
            with self.env.begin(write=True) as target_txn:
                for _, manifest in manifests:
                    self._put_file_manifest(target_txn, manifest)
            return

        source = LmdbChunkStore.open(source_path, readonly=True)
        try:
            with source.env.begin(buffers=True) as source_txn, self.env.begin(write=True) as target_txn:
                for file_path in file_paths:
                    data = source_txn.get(file_path.encode(), db=source.files_db)
                    if data is None:
                        raise FileNotFoundError("Index chunk store is missing file manifests")
                    self._put_file_manifest(target_txn, self._manifest_for_generation(bytes(data), generation))
        finally:
            source.close()

    def write_file_manifest(self, manifest: FileManifest) -> None:
        """Persist one file manifest."""
        with self.env.begin(write=True) as txn:
            self._put_file_manifest(txn, manifest)

    def get_file_manifest(self, file_path: str) -> FileManifest | None:
        """Return one file manifest by repo-relative path."""
        active_generation = self.active_generation()
        with self.env.begin(buffers=True) as txn:
            data = txn.get(file_path.encode(), db=self.files_db)
            if data is None:
                return None
            manifest = FileManifest.from_dict(orjson.loads(bytes(data)))
            return manifest if manifest.generation == active_generation else None

    def get_file_manifest_by_hash(self, file_hash: str) -> FileManifest | None:
        """Return one active file manifest by content identity."""
        return self.get_file_manifests_by_hashes([file_hash]).get(file_hash)

    def get_file_manifests_by_hashes(self, file_hashes: Sequence[str]) -> dict[str, FileManifest]:
        """Return active file manifests by content identity."""
        active_generation = self.active_generation()
        wanted = set(file_hashes)
        manifests: dict[str, FileManifest] = {}
        if not wanted:
            return manifests
        with self.env.begin(buffers=True) as txn:
            if self.file_hashes_db is None:
                with txn.cursor(db=self.files_db) as cursor:
                    for _, manifest_data in cursor:
                        manifest = FileManifest.from_dict(orjson.loads(bytes(manifest_data)))
                        if manifest.file_hash in wanted and manifest.generation == active_generation:
                            manifests.setdefault(manifest.file_hash, manifest)
                return manifests

            for file_hash in wanted:
                data = txn.get(file_hash.encode(), db=self.file_hashes_db)
                if data is None:
                    continue
                for file_path in orjson.loads(bytes(data)):
                    manifest_data = txn.get(str(file_path).encode(), db=self.files_db)
                    if manifest_data is None:
                        continue
                    manifest = FileManifest.from_dict(orjson.loads(bytes(manifest_data)))
                    if manifest.generation == active_generation:
                        manifests[file_hash] = manifest
                        break
        return manifests

    def iter_file_manifests(self) -> Iterator[FileManifest]:
        """Yield persisted file manifests."""
        yield from self.iter_file_manifests_for_generation(self.active_generation())

    def iter_file_manifests_for_generation(self, generation: int) -> Iterator[FileManifest]:
        """Yield file manifests for a specific generation."""
        with self.env.begin(buffers=True) as txn:
            with txn.cursor(db=self.files_db) as cursor:
                for _, data in cursor:
                    manifest = FileManifest.from_dict(orjson.loads(bytes(data)))
                    if manifest.generation == generation:
                        yield manifest

    def write_embedding(self, key: str, vector: npt.NDArray[np.float32]) -> None:
        """Persist one embedding vector by cache key."""
        if self.embeddings_db is None:
            raise RuntimeError("Embedding store is not available")
        payload = np.asarray(vector, dtype=np.float32).tobytes()
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), payload, db=self.embeddings_db)

    def get_embedding(self, key: str) -> npt.NDArray[np.float32] | None:
        """Return one embedding vector by cache key."""
        if self.embeddings_db is None:
            return None
        with self.env.begin(buffers=True) as txn:
            data = txn.get(key.encode(), db=self.embeddings_db)
            if data is None:
                return None
            return np.frombuffer(bytes(data), dtype=np.float32).copy()

    def get_embeddings(self, keys: Sequence[str]) -> dict[str, npt.NDArray[np.float32]]:
        """Return embedding vectors for existing keys."""
        if self.embeddings_db is None:
            return {}
        embeddings: dict[str, npt.NDArray[np.float32]] = {}
        with self.env.begin(buffers=True) as txn:
            for key in keys:
                data = txn.get(key.encode(), db=self.embeddings_db)
                if data is not None:
                    embeddings[key] = np.frombuffer(bytes(data), dtype=np.float32).copy()
        return embeddings

    def write_embeddings(self, embeddings: dict[str, npt.NDArray[np.float32]]) -> None:
        """Persist embedding vectors by cache key."""
        if self.embeddings_db is None:
            raise RuntimeError("Embedding store is not available")
        with self.env.begin(write=True) as txn:
            for key, vector in embeddings.items():
                payload = np.asarray(vector, dtype=np.float32).tobytes()
                txn.put(key.encode(), payload, db=self.embeddings_db)

    def write_query_embedding(self, key: str, vector: npt.NDArray[np.float32]) -> None:
        """Persist one query embedding vector by cache key."""
        if self.query_embeddings_db is None:
            raise RuntimeError("Query embedding store is not available")
        payload = np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), payload, db=self.query_embeddings_db)

    def get_query_embedding(self, key: str) -> npt.NDArray[np.float32] | None:
        """Return one cached query embedding as a single-query matrix."""
        if self.query_embeddings_db is None:
            return None
        with self.env.begin(buffers=True) as txn:
            data = txn.get(key.encode(), db=self.query_embeddings_db)
            if data is None:
                return None
            return np.frombuffer(bytes(data), dtype=np.float32).copy().reshape(1, -1)

    def write_bm25_document(self, key: str, document: list[str]) -> None:
        """Persist one tokenized BM25 document by cache key."""
        if self.bm25_docs_db is None:
            raise RuntimeError("BM25 document store is not available")
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), orjson.dumps(document), db=self.bm25_docs_db)

    def write_rebuild_cache(
        self,
        manifests: list[FileManifest],
        embeddings: dict[str, npt.NDArray[np.float32]],
        bm25_documents: dict[str, list[str]],
    ) -> None:
        """Persist rebuild-cache payloads in one write transaction."""
        self.write_chunks_and_rebuild_cache([], manifests, embeddings, bm25_documents)

    def write_chunks_and_rebuild_cache(
        self,
        chunks: list[Chunk],
        manifests: list[FileManifest],
        embeddings: dict[str, npt.NDArray[np.float32]],
        bm25_documents: dict[str, list[str]],
    ) -> None:
        """Persist chunk payloads and rebuild-cache payloads in one write transaction."""
        if self.embeddings_db is None:
            raise RuntimeError("Embedding store is not available")
        if self.bm25_docs_db is None:
            raise RuntimeError("BM25 document store is not available")
        with self.env.begin(write=True) as txn:
            next_chunk_id = self.next_chunk_id()
            for chunk in chunks:
                if chunk.chunk_id is None:
                    continue
                txn.put(_int_key(chunk.chunk_id), _serialize_chunk(chunk), db=self.chunks_db)
                next_chunk_id = max(next_chunk_id, chunk.chunk_id + 1)
            if chunks:
                txn.put(_NEXT_CHUNK_ID, str(next_chunk_id).encode(), db=self.meta_db)
            for key, vector in embeddings.items():
                payload = np.asarray(vector, dtype=np.float32).tobytes()
                txn.put(key.encode(), payload, db=self.embeddings_db)
            for key, document in bm25_documents.items():
                txn.put(key.encode(), orjson.dumps(document), db=self.bm25_docs_db)
            for manifest in manifests:
                self._put_file_manifest(txn, manifest)

    def get_bm25_document(self, key: str) -> list[str] | None:
        """Return one tokenized BM25 document by cache key."""
        return self.get_bm25_documents([key]).get(key)

    def get_bm25_documents(self, keys: Sequence[str]) -> dict[str, list[str]]:
        """Return tokenized BM25 documents for existing keys."""
        if self.bm25_docs_db is None:
            return {}
        documents: dict[str, list[str]] = {}
        with self.env.begin(buffers=True) as txn:
            for key in keys:
                data = txn.get(key.encode(), db=self.bm25_docs_db)
                if data is not None:
                    documents[key] = list(orjson.loads(bytes(data)))
        return documents

    def next_chunk_id(self) -> int:
        """Return the next stable chunk_id to allocate."""
        with self.env.begin(buffers=True) as txn:
            data = txn.get(_NEXT_CHUNK_ID, db=self.meta_db)
            return 0 if data is None else int(bytes(data))

    def active_generation(self) -> int:
        """Return the committed LMDB generation visible to readers."""
        generation = self._generation_value(_ACTIVE_GENERATION)
        return 0 if generation is None else generation

    def active_generation_snapshot_id(self) -> str | None:
        """Return the SourceSnapshot id bound to the active generation."""
        if self.generations_db is None:
            return None
        with self.env.begin(buffers=True) as txn:
            data = txn.get(_ACTIVE_GENERATION_SNAPSHOT_ID, db=self.generations_db)
            return None if data is None else bytes(data).decode()

    def pending_generation(self) -> int | None:
        """Return the in-progress LMDB generation, if any."""
        return self._generation_value(_PENDING_GENERATION)

    def begin_generation(self, generation: int) -> None:
        """Record an in-progress generation before derived indexes are written."""
        if self.generations_db is None:
            raise RuntimeError("Generation store is not available")
        with self.env.begin(write=True) as txn:
            txn.put(_PENDING_GENERATION, str(generation).encode(), db=self.generations_db)

    def commit_generation(self, generation: int, source_snapshot_id: str | None = None) -> None:
        """Promote a pending generation to active."""
        if self.generations_db is None:
            raise RuntimeError("Generation store is not available")
        with self.env.begin(write=True) as txn:
            txn.put(_ACTIVE_GENERATION, str(generation).encode(), db=self.generations_db)
            if source_snapshot_id is None:
                txn.delete(_ACTIVE_GENERATION_SNAPSHOT_ID, db=self.generations_db)
            else:
                txn.put(_ACTIVE_GENERATION_SNAPSHOT_ID, source_snapshot_id.encode(), db=self.generations_db)
            txn.delete(_PENDING_GENERATION, db=self.generations_db)

    def recover_generation(self) -> None:
        """Discard pending generation state left by an interrupted save."""
        if self.generations_db is None:
            return
        active_generation = self.active_generation()
        with self.env.begin(write=True) as txn:
            with txn.cursor(db=self.files_db) as cursor:
                for _, data in cursor:
                    manifest = FileManifest.from_dict(orjson.loads(bytes(data)))
                    if manifest.generation > active_generation:
                        cursor.delete()
            txn.delete(_PENDING_GENERATION, db=self.generations_db)

    def _put_file_manifest(self, txn: lmdb.Transaction, manifest: FileManifest) -> None:
        if self.file_hashes_db is not None:
            existing_data = txn.get(manifest.file_path.encode(), db=self.files_db)
            if existing_data is not None:
                existing = FileManifest.from_dict(orjson.loads(bytes(existing_data)))
                if existing.file_hash != manifest.file_hash:
                    self._remove_file_hash_path(txn, existing.file_hash, manifest.file_path)
        txn.put(manifest.file_path.encode(), orjson.dumps(manifest), db=self.files_db)
        if self.file_hashes_db is None:
            return
        key = manifest.file_hash.encode()
        data = txn.get(key, db=self.file_hashes_db)
        paths = [] if data is None else list(orjson.loads(bytes(data)))
        if manifest.file_path not in paths:
            paths.append(manifest.file_path)
            paths.sort()
        txn.put(key, orjson.dumps(paths), db=self.file_hashes_db)

    def _remove_file_hash_path(self, txn: lmdb.Transaction, file_hash: str, file_path: str) -> None:
        if self.file_hashes_db is None:
            return
        key = file_hash.encode()
        data = txn.get(key, db=self.file_hashes_db)
        if data is None:
            return
        paths = [path for path in orjson.loads(bytes(data)) if path != file_path]
        if paths:
            txn.put(key, orjson.dumps(paths), db=self.file_hashes_db)
        else:
            txn.delete(key, db=self.file_hashes_db)

    def _generation_value(self, key: bytes) -> int | None:
        if self.generations_db is None:
            return None
        with self.env.begin(buffers=True) as txn:
            data = txn.get(key, db=self.generations_db)
            return None if data is None else int(bytes(data))


def _open_db(env: lmdb.Environment, name: bytes, readonly: bool) -> lmdb._Database | None:
    try:
        return env.open_db(name, create=not readonly)
    except lmdb.NotFoundError:
        return None


def _int_key(value: int) -> bytes:
    return value.to_bytes(8, "big", signed=False)
