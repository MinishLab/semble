import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from vicinity.backends.basic import BasicArgs

from semble.index.dense import SelectableBasicBackend
from semble.index.turbovec import TurboVecBasicBackend, TurboVecBuildWriter


def _assert_single_native_add(index: Any, expected_ids: list[int]) -> None:
    index.add_with_ids.assert_called_once()
    vectors, ids = index.add_with_ids.call_args.args
    assert vectors.shape == (len(expected_ids), 8)
    assert ids.tolist() == expected_ids


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test save and load roundtrip."""
    vecs = np.random.default_rng(seed=42).normal(size=(10, 32))
    args = BasicArgs()
    selectable = SelectableBasicBackend(vecs, args)
    selectable.save(tmp_path)

    selectable_2 = SelectableBasicBackend.load(tmp_path)
    assert np.allclose(selectable.vectors, selectable_2.vectors)


def test_turbovec_build_writer_batches_rows_into_one_native_add() -> None:
    """First-cold streaming should not retrain TurboVec codebooks once per row."""
    rows = [np.eye(1, 8, dtype=np.float32)[0], np.eye(1, 8, dtype=np.float32)[0]]
    writer = TurboVecBuildWriter(BasicArgs())

    with patch("semble.index.turbovec.IdMapIndex") as index_class:
        index = index_class.return_value
        writer.add_rows(rows, [7, 8])
        writer.finish()

    _assert_single_native_add(index, [7, 8])


def test_turbovec_build_writer_coalesces_small_streaming_batches() -> None:
    """Small streaming batches should not retrain TurboVec codebooks per file."""
    row_a = np.eye(1, 8, dtype=np.float32)[0]
    row_b = np.eye(1, 8, dtype=np.float32)[0]
    writer = TurboVecBuildWriter(BasicArgs())

    with patch("semble.index.turbovec.IdMapIndex") as index_class:
        index = index_class.return_value
        writer.add_rows([row_a], [7])
        writer.add_rows([row_b], [8])
        assert index.add_with_ids.call_count == 0
        writer.finish()

    _assert_single_native_add(index, [7, 8])


def test_turbovec_build_writer_defers_native_add_after_streaming_flush_threshold() -> None:
    """Streaming thresholds should not retrain TurboVec codebooks before finish."""
    row_a = np.eye(1, 8, dtype=np.float32)[0]
    row_b = np.eye(1, 8, dtype=np.float32)[0]
    writer = TurboVecBuildWriter(BasicArgs(), add_batch_size=2)

    with patch("semble.index.turbovec.IdMapIndex") as index_class:
        index = index_class.return_value
        writer.add_rows([row_a], [7])
        writer.add_rows([row_b], [8])
        assert index.add_with_ids.call_count == 0
        writer.finish()

    _assert_single_native_add(index, [7, 8])


def test_turbovec_build_writer_saves_streamed_rows_without_per_row_normalization(tmp_path: Path) -> None:
    """First-cold dense persistence should not normalize streamed rows one at a time."""
    rows = [
        np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 5.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]
    writer = TurboVecBuildWriter(BasicArgs())
    writer.add_rows(rows, [7, 8])
    backend = writer.finish()

    with patch(
        "semble.index.turbovec.normalize_or_copy",
        side_effect=AssertionError("dense save should not normalize rows during persistence"),
    ):
        backend.save(tmp_path)

    loaded = TurboVecBasicBackend.load(tmp_path)
    indices, distances = loaded.query(np.array([rows[0]], dtype=np.float32), 1)[0]
    assert indices.tolist() == [7]
    assert float(distances[0]) == 0.0


def test_turbovec_from_embedding_rows_batches_rows_into_one_native_add() -> None:
    """Non-streaming dense builds should not retrain TurboVec codebooks once per row."""
    rows = [np.eye(1, 8, dtype=np.float32)[0], np.eye(1, 8, dtype=np.float32)[0]]

    with patch("semble.index.turbovec.IdMapIndex") as index_class:
        index = index_class.return_value
        TurboVecBasicBackend.from_embedding_rows(rows, BasicArgs(), vector_ids=np.array([7, 8], dtype=np.uint64))

    _assert_single_native_add(index, [7, 8])


def test_turbovec_from_embedding_rows_saves_without_per_row_normalization(tmp_path: Path) -> None:
    """Non-streaming dense persistence should reuse batch-normalized rows."""
    rows = [
        np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 5.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]
    backend = TurboVecBasicBackend.from_embedding_rows(rows, BasicArgs(), vector_ids=np.array([7, 8], dtype=np.uint64))

    with patch(
        "semble.index.turbovec.normalize_or_copy",
        side_effect=AssertionError("dense save should not normalize rows during persistence"),
    ):
        backend.save(tmp_path)

    loaded = TurboVecBasicBackend.load(tmp_path)
    indices, distances = loaded.query(np.array([rows[0]], dtype=np.float32), 1)[0]
    assert indices.tolist() == [7]
    assert float(distances[0]) == 0.0


def test_turbovec_backend_persists_native_segment_metadata(tmp_path: Path) -> None:
    """TurboVec dense persistence should include native segment files beside legacy vectors."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors, BasicArgs())

    backend.save(tmp_path, created_generation=3, model_path="mock-model")

    assert (tmp_path / "dense_segments" / "segment-0000.tvim").exists()
    segment_metadata = json.loads((tmp_path / "dense_segments" / "segment-0000.meta.json").read_text())
    assert segment_metadata["created_generation"] == 3
    assert segment_metadata["model_path"] == "mock-model"
    loaded = TurboVecBasicBackend.load(tmp_path)
    indices, distances = loaded.query(vectors[[0]], 2)[0]
    assert int(indices[0]) == 0
    assert float(distances[0]) == 0.0


def test_turbovec_load_queries_native_segments_without_loading_raw_vectors(tmp_path: Path) -> None:
    """Hot search loads should use native TurboVec segments instead of vectors.npy."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.save(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded = TurboVecBasicBackend.load(tmp_path)

    indices, distances = loaded.query(vectors[[1]], 2)[0]
    assert int(indices[0]) == 11
    assert float(distances[0]) == 0.0


def test_turbovec_loaded_segment_rows_do_not_load_raw_vectors(tmp_path: Path) -> None:
    """Save fallback should read requested segment rows without loading full vectors.npy."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        rows = loaded.vector_rows_for_ids([11])

    assert rows is not None
    np.testing.assert_array_equal(rows[0], vectors[1])


def test_turbovec_filtered_query_uses_native_segments_without_loading_raw_vectors(tmp_path: Path) -> None:
    """Filtered hot search should apply selectors to stable IDs without vectors.npy."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.save(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded = TurboVecBasicBackend.load(tmp_path)
        indices, distances = loaded.query(vectors[[1]], 2, selector=np.array([11], dtype=np.int_))[0]

    assert indices.tolist() == [11]
    assert float(distances[0]) == 0.0


def test_turbovec_hot_loaded_append_segment_without_loading_raw_vectors(tmp_path: Path) -> None:
    """Incremental dense updates should append new native segments without loading vectors.npy."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors[:2], BasicArgs(), vector_ids=np.array([10, 11], dtype=np.uint64))
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded.add_segment(vectors[2:], np.array([12], dtype=np.uint64))
        old_indices, _ = loaded.query(vectors[[0]], 3)[0]
        new_indices, _ = loaded.query(vectors[[2]], 3)[0]

    assert int(old_indices[0]) == 10
    assert int(new_indices[0]) == 12


def test_turbovec_saves_appended_segments_without_loading_raw_vectors(tmp_path: Path) -> None:
    """Persisting incremental dense segments should preserve old and new native segment files."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors[:2], BasicArgs(), vector_ids=np.array([10, 11], dtype=np.uint64))
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded.add_segment(vectors[2:], np.array([12], dtype=np.uint64))
        loaded.save(tmp_path)
        reloaded = TurboVecBasicBackend.load(tmp_path)
        old_indices, _ = reloaded.query(vectors[[0]], 3)[0]
        new_indices, _ = reloaded.query(vectors[[2]], 3)[0]

    assert (tmp_path / "dense_segments" / "segment-0000.tvim").exists()
    assert (tmp_path / "dense_segments" / "segment-0001.tvim").exists()
    assert int(old_indices[0]) == 10
    assert int(new_indices[0]) == 12


def test_turbovec_saves_appended_segments_without_rewriting_existing_segment(tmp_path: Path) -> None:
    """Incremental dense save should not rewrite unchanged native segment files."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors[:2], BasicArgs(), vector_ids=np.array([10, 11], dtype=np.uint64))
    backend.save(tmp_path)
    existing_segment = tmp_path / "dense_segments" / "segment-0000.tvim"
    old_time_ns = 1_000_000_000
    os.utime(existing_segment, ns=(old_time_ns, old_time_ns))
    loaded = TurboVecBasicBackend.load(tmp_path)

    loaded.add_segment(vectors[2:], np.array([12], dtype=np.uint64))
    loaded.save(tmp_path)

    assert existing_segment.stat().st_mtime_ns == old_time_ns
    assert (tmp_path / "dense_segments" / "segment-0001.tvim").exists()


def test_turbovec_hot_loaded_save_preserves_raw_vectors_for_compact(tmp_path: Path) -> None:
    """Hot-loaded appended dense segments should stay compactable after save/reload."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors[:2], BasicArgs(), vector_ids=np.array([10, 11], dtype=np.uint64))
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)
    loaded.add_segment(vectors[2:], np.array([12], dtype=np.uint64))
    loaded.save(tmp_path)
    reloaded = TurboVecBasicBackend.load(tmp_path)
    reloaded.delete_ids({10})

    reloaded.compact()

    assert reloaded.vector_ids.tolist() == [11, 12]
    indices, _ = reloaded.query(vectors[[2]], 2)[0]
    assert int(indices[0]) == 12


def test_turbovec_appended_segments_use_cosine_normalization(tmp_path: Path) -> None:
    """Appended dense segments should keep cosine-distance semantics for non-unit vectors."""
    base = np.eye(1, 8, dtype=np.float32)
    appended = np.array([[100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    query = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    backend = TurboVecBasicBackend(base, BasicArgs(), vector_ids=np.array([10], dtype=np.uint64))
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)

    loaded.add_segment(appended, np.array([12], dtype=np.uint64))
    indices, distances = loaded.query(query, 1, selector=np.array([12], dtype=np.int_))[0]

    assert indices.tolist() == [12]
    assert float(distances[0]) > 0.9


def test_turbovec_vectors_assignment_rebuilds_native_segments() -> None:
    """Replacing raw vectors should rebuild native segments instead of querying stale vectors."""
    backend = TurboVecBasicBackend(np.eye(2, 8, dtype=np.float32), BasicArgs())
    replacement = np.eye(2, 8, dtype=np.float32)[::-1]

    backend.vectors = replacement
    indices, _ = backend.query(replacement[[0]], 2)[0]

    assert int(indices[0]) == 0


def test_turbovec_hot_loaded_delete_ids_keeps_native_segments_queryable(tmp_path: Path) -> None:
    """Deleting from a hot-loaded native index should keep non-deleted IDs searchable."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded.delete_ids({10})
        indices, _ = loaded.query(vectors[[1]], 3)[0]

    assert 10 not in indices.tolist()
    assert int(indices[0]) == 11


def test_turbovec_load_preserves_all_tombstones_without_native_segment_files(tmp_path: Path) -> None:
    """Reloading an index with no live segments should not resurrect tombstoned IDs."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.delete_ids({10, 11, 12})
    backend.save(tmp_path)

    with patch.object(SelectableBasicBackend, "load", side_effect=AssertionError("raw vectors should not load")):
        loaded = TurboVecBasicBackend.load(tmp_path)
        indices, _ = loaded.query(vectors[[0]], 3)[0]

    assert indices.tolist() == []


def test_turbovec_backend_tombstones_and_compacts_vectors() -> None:
    """Tombstoned dense IDs should be excluded until compact removes their vectors."""
    vectors = np.eye(3, 8, dtype=np.float32)
    backend = TurboVecBasicBackend(vectors, BasicArgs())

    backend.delete_ids({0})

    indices, _ = backend.query(vectors[[0]], 3)[0]
    assert 0 not in set(indices.tolist())

    backend.compact()

    assert len(backend.vectors) == 2
    indices, _ = backend.query(vectors[[1]], 2)[0]
    assert int(indices[0]) == 1

    indices, _ = backend.query(vectors[[1]], 2, selector=np.array([1]))[0]
    assert indices.tolist() == [1]


def test_turbovec_hot_loaded_compact_loads_vectors_before_applying_tombstones(tmp_path: Path) -> None:
    """Hot-loaded dense indexes should compact using persisted vectors, not the empty lazy placeholder."""
    vectors = np.eye(3, 8, dtype=np.float32)
    vector_ids = np.array([10, 11, 12], dtype=np.uint64)
    backend = TurboVecBasicBackend(vectors, BasicArgs(), vector_ids=vector_ids)
    backend.save(tmp_path)
    loaded = TurboVecBasicBackend.load(tmp_path)
    loaded.delete_ids({10})

    loaded.compact()

    assert loaded.vector_ids.tolist() == [11, 12]
    indices, _ = loaded.query(vectors[[1]], 2)[0]
    assert int(indices[0]) == 11
