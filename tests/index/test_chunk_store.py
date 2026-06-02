from pathlib import Path

import numpy as np

from semble.index.chunk_store import FileManifest, LmdbChunkStore
from semble.types import Chunk
from tests.conftest import make_chunk


def chunk_with_id(content: str, file_path: str, chunk_id: int) -> Chunk:
    """Create a test chunk with a stable chunk ID."""
    chunk = make_chunk(content, file_path)
    return Chunk(
        content=chunk.content,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        language=chunk.language,
        chunk_id=chunk_id,
    )


def test_query_embedding_round_trip(tmp_path: Path) -> None:
    """Query embeddings should persist as single-query matrices for hot search reuse."""
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    try:
        vector = np.array([[0.25, 0.5, 0.75]], dtype=np.float32)
        store.write_query_embedding("query-key", vector)
        loaded = store.get_query_embedding("query-key")
    finally:
        store.close()

    np.testing.assert_allclose(loaded, vector)

    """Chunk payloads are looked up by stable chunk_id without loading a JSON list."""
    chunks = [
        chunk_with_id("def authenticate(token):\n    return token", "auth.py", 7),
        chunk_with_id("def format_date(dt):\n    return str(dt)", "utils.py", 9),
    ]
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    store.write_chunks(chunks)
    store.close()

    loaded = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)

    assert loaded.get_chunk(7) == chunks[0]
    assert loaded.get_chunks([9, 7]) == [chunks[1], chunks[0]]
    assert loaded.next_chunk_id() == 10
    loaded.close()


def test_file_manifest_round_trips_source_root_metadata() -> None:
    """File manifests should preserve source inventory ownership metadata."""
    manifest = FileManifest(
        file_path="sources/service/auth.py",
        file_hash="abc123",
        file_size=128,
        file_mtime_ns=456,
        chunk_ids=[7, 8],
        source_root="sources/service",
        git_path="auth.py",
        tracked=False,
        generation=3,
    )

    assert FileManifest.from_dict(manifest.to_dict()) == manifest


def test_file_manifest_reads_legacy_metadata_without_source_root() -> None:
    """Old manifests should remain readable after source-root metadata is added."""
    manifest = FileManifest.from_dict(
        {
            "file_path": "auth.py",
            "file_hash": "abc123",
            "file_size": 128,
            "file_mtime_ns": 456,
            "chunk_ids": [7, 8],
        }
    )

    assert manifest.source_root == ""
    assert manifest.git_path == "auth.py"
    assert manifest.tracked is True
    assert manifest.generation == 0


def test_lmdb_chunk_store_persists_file_manifest(tmp_path: Path) -> None:
    """File manifests preserve file metadata and live chunk IDs."""
    manifest = FileManifest(
        file_path="auth.py",
        file_hash="abc123",
        file_size=128,
        file_mtime_ns=456,
        chunk_ids=[7, 8],
    )
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    store.write_file_manifest(manifest)
    store.close()

    loaded = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)

    assert loaded.get_file_manifest("auth.py") == manifest
    loaded.close()


def test_lmdb_chunk_store_iterates_file_manifests(tmp_path: Path) -> None:
    """Incremental rebuilds need every file manifest without reading chunk payload JSON."""
    manifests = [
        FileManifest("auth.py", "abc123", 128, 456, [7, 8]),
        FileManifest("utils.py", "def456", 64, 789, [9]),
    ]
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    for manifest in manifests:
        store.write_file_manifest(manifest)
    store.close()

    loaded = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)

    assert sorted(loaded.iter_file_manifests(), key=lambda item: item.file_path) == manifests
    loaded.close()


def test_lmdb_chunk_store_copies_file_manifests_with_new_generation(tmp_path: Path) -> None:
    """Reused file manifests should move between generation stores without rebuilding entries."""
    source_path = tmp_path / "source.lmdb"
    target_path = tmp_path / "target.lmdb"
    source = LmdbChunkStore.open(source_path)
    source.write_rebuild_cache([FileManifest("auth.py", "abc123", 128, 456, [7], generation=1)], {}, {})
    source.commit_generation(1)
    source.close()

    target = LmdbChunkStore.open(target_path)
    target.begin_generation(2)
    target.copy_file_manifests_from(source_path, ["auth.py"], 2)
    target.commit_generation(2)

    assert target.get_file_manifest("auth.py") == FileManifest("auth.py", "abc123", 128, 456, [7], generation=2)
    target.close()


def test_lmdb_chunk_store_refreshes_file_manifest_generation_in_place(tmp_path: Path) -> None:
    """In-place saves should keep reused manifests visible after active generation advances."""
    store_path = tmp_path / "chunks.lmdb"
    store = LmdbChunkStore.open(store_path)
    store.begin_generation(1)
    store.write_rebuild_cache([FileManifest("auth.py", "abc123", 128, 456, [7], generation=1)], {}, {})
    store.commit_generation(1)
    store.begin_generation(2)

    store.copy_file_manifests_from(store_path, ["auth.py"], 2)
    store.commit_generation(2)

    assert store.get_file_manifest("auth.py") == FileManifest("auth.py", "abc123", 128, 456, [7], generation=2)
    store.close()


def test_lmdb_chunk_store_persists_embedding_by_key(tmp_path: Path) -> None:
    """Embedding cache belongs in LMDB so rebuilds do not load full semantic_index vectors."""
    vector = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    store.write_embedding("embedding-key", vector)
    store.close()

    loaded = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)

    np.testing.assert_allclose(loaded.get_embedding("embedding-key"), vector)
    assert loaded.get_embedding("missing") is None
    loaded.close()


def test_lmdb_chunk_store_persists_bm25_document_by_key(tmp_path: Path) -> None:
    """BM25 term cache belongs in LMDB so removing chunk_cache.json keeps token reuse."""
    document = ["authenticate", "token"]
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    store.write_bm25_document("bm25-key", document)
    store.close()

    loaded = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)

    assert loaded.get_bm25_document("bm25-key") == document
    assert loaded.get_bm25_document("missing") is None
    loaded.close()


def test_lmdb_chunk_store_tracks_generation_boundary(tmp_path: Path) -> None:
    """Generation metadata should expose pending and active commit boundaries."""
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")

    assert store.active_generation() == 0
    assert store.pending_generation() is None
    assert store.active_generation_snapshot_id() is None

    store.begin_generation(1)
    assert store.active_generation() == 0
    assert store.pending_generation() == 1

    store.commit_generation(1, source_snapshot_id="snapshot-1")
    assert store.active_generation() == 1
    assert store.pending_generation() is None
    assert store.active_generation_snapshot_id() == "snapshot-1"

    store.begin_generation(2)
    store.recover_generation()
    assert store.active_generation() == 1
    assert store.pending_generation() is None
    assert store.active_generation_snapshot_id() == "snapshot-1"
    store.close()


def test_lmdb_chunk_store_hides_inactive_generation_manifests(tmp_path: Path) -> None:
    """Incremental rebuild 不应读取旧 generation 的 file manifest."""
    stale = FileManifest("auth.py", "abc123", 128, 456, [7], generation=1)
    current = FileManifest("utils.py", "def456", 64, 789, [9], generation=2)
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")

    store.begin_generation(1)
    store.write_rebuild_cache([stale], {}, {})
    store.commit_generation(1)
    store.begin_generation(2)
    store.write_rebuild_cache([current], {}, {})
    store.commit_generation(2)

    assert list(store.iter_file_manifests()) == [current]
    assert store.get_file_manifest("auth.py") is None
    assert store.get_file_manifest("utils.py") == current
    store.close()
