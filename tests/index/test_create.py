from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import orjson
import pytest

from semble.cache import load_previous_for_incremental
from semble.index.create import create_index_from_path
from semble.index.index import SembleIndex
from semble.index.types import PreviousIndex, make_chunk_id
from semble.types import ContentType


def _write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_incremental_reindex_reuses_updates_and_prunes(mock_model: Any, tmp_path: Path) -> None:
    """One incremental pass reuses unchanged vectors, re-embeds changes, and keeps BM25 slots current."""
    _write_files(
        tmp_path,
        {
            "a.py": "def stable_anchor():\n    return 1\n",
            "b.py": "def changed_value():\n    return 2\n",
            "c.py": "def unique_gone():\n    return 3\n",
            "emptying.py": "def becomes_empty():\n    return 4\n",
        },
    )
    bm25_before, semantic_before, chunks_before, manifest_before = create_index_from_path(
        tmp_path, mock_model, display_root=tmp_path
    )
    a_entry = manifest_before["a.py"]
    b_entry = manifest_before["b.py"]
    a_vectors_before = semantic_before.vectors[a_entry.start : a_entry.end].copy()
    b_vectors_before = semantic_before.vectors[b_entry.start : b_entry.end].copy()
    previous = PreviousIndex(
        chunks=chunks_before,
        vectors=semantic_before.vectors,
        manifest=manifest_before,
        bm25_index=bm25_before,
    )
    _, semantic_unchanged, _, _ = create_index_from_path(tmp_path, mock_model, display_root=tmp_path, previous=previous)
    assert semantic_unchanged.vectors is semantic_before.vectors

    (tmp_path / "b.py").write_text("def changed_value():\n    return 999\n")
    bm25_before, semantic_before, chunks_before, manifest_before = create_index_from_path(
        tmp_path, mock_model, display_root=tmp_path, previous=previous
    )
    assert semantic_before.vectors is previous.vectors
    previous = PreviousIndex(
        chunks=chunks_before,
        vectors=semantic_before.vectors,
        manifest=manifest_before,
        bm25_index=bm25_before,
    )

    (tmp_path / "c.py").unlink()
    (tmp_path / "emptying.py").write_text(" " * 128)
    _write_files(tmp_path, {"d.py": "def brand_new_term():\n    return 4\n"})
    bm25_after, semantic_after, _, manifest_after = create_index_from_path(
        tmp_path, mock_model, display_root=tmp_path, previous=previous
    )

    a_entry_after = manifest_after["a.py"]
    b_entry_after = manifest_after["b.py"]
    np.testing.assert_array_equal(semantic_after.vectors[a_entry_after.start : a_entry_after.end], a_vectors_before)
    assert not np.array_equal(
        b_vectors_before,
        semantic_after.vectors[b_entry_after.start : b_entry_after.end],
    )
    assert "c.py" not in manifest_after
    assert "d.py" in manifest_after
    assert manifest_after["emptying.py"].count == 0
    assert bm25_after.get_scores(["unique_gone"]).sum() == 0
    assert bm25_after.get_scores(["becomes_empty"]).sum() == 0
    assert bm25_after.get_scores(["brand", "new", "term"]).sum() > 0
    expected_ids = {
        make_chunk_id(indexed_path, slot)
        for indexed_path, entry in manifest_after.items()
        for slot in range(entry.count)
    }
    assert set(bm25_after.doc_order) == expected_ids


def _build_valid_cache(index_path: Path, mock_model: Any) -> dict:
    """Build a real, well-formed on-disk index and return its metadata dict for mutation."""
    src = index_path.parent / "src"
    _write_files(src, {"a.py": "def a():\n    return 1\n", "b.py": "def b():\n    return 2\n"})

    with patch("semble.index.index.load_model", return_value=(mock_model, "my/model")):
        SembleIndex.from_path(src).save(index_path)
    return orjson.loads((index_path / "metadata.json").read_bytes())


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_cache",
        "missing_files_key",
        "metadata_mismatch",
        "component_length_mismatch",
        "length_mismatch",
        "overlapping_entries",
        "bm25_order_mismatch",
        "corrupt_json",
    ],
)
def test_load_previous_for_incremental_fails_closed(corrupt: str, tmp_path: Path, mock_model: Any) -> None:
    """Any structurally invalid or missing cache state yields None instead of raising."""
    index_path = tmp_path / "index"

    if corrupt != "missing_cache":
        metadata = _build_valid_cache(index_path, mock_model)
        if corrupt == "missing_files_key":
            del metadata["files"]
        elif corrupt == "metadata_mismatch":
            metadata["model_path"] = "other/model"
        elif corrupt == "component_length_mismatch":
            chunks_path = index_path / "chunks.json"
            chunks = orjson.loads(chunks_path.read_bytes())
            chunks_path.write_bytes(orjson.dumps(chunks[:-1]))
        elif corrupt == "length_mismatch":
            metadata["files"]["a.py"]["count"] += 5
        elif corrupt == "overlapping_entries":
            metadata["files"]["b.py"]["start"] = metadata["files"]["a.py"]["start"]
        elif corrupt == "bm25_order_mismatch":
            bm25_path = index_path / "bm25_index" / "index.json"
            bm25 = orjson.loads(bm25_path.read_bytes())
            bm25["doc_order"].reverse()
            bm25_path.write_bytes(orjson.dumps(bm25))
        elif corrupt == "corrupt_json":
            (index_path / "metadata.json").write_bytes(b"{not json")
            with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
                assert load_previous_for_incremental("/some/path", "my/model", [ContentType.CODE]) is None
            return
        (index_path / "metadata.json").write_bytes(orjson.dumps(metadata))

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = load_previous_for_incremental("/some/path", "my/model", [ContentType.CODE])
    assert result is None


def test_load_previous_for_incremental_happy_path(mock_model: Any, tmp_path: Path) -> None:
    """A well-formed cache round-trips into a usable PreviousIndex."""
    index_path = tmp_path / "cache" / "index"
    _build_valid_cache(index_path, mock_model)

    with (
        patch("semble.cache.find_index_from_cache_folder", return_value=index_path),
        patch("semble.cache.resolve_model_name", return_value="my/model"),
    ):
        previous = load_previous_for_incremental(str(index_path.parent / "src"), None, [ContentType.CODE])

    assert previous is not None
    assert len(previous.chunks) == previous.vectors.shape[0] == len(previous.bm25_index.doc_order)
    assert "a.py" in previous.manifest
