from pathlib import Path

import numpy as np
import pytest

from semble.index.chunks import ChunkTable
from semble.types import Chunk

_CHUNKS = [
    Chunk("def a():\n    return 1", "src/a.py", 1, 2, "python"),
    Chunk("naïve → ünïcode", "docs/b.md", 3, 3, None),
    Chunk("", "src/a.py", 4, 4, "python"),
]


def test_round_trip_and_access(tmp_path: Path) -> None:
    """A saved and reloaded table builds the same chunks and supports negative indices, slicing and lookups."""
    ChunkTable.from_chunks(_CHUNKS).save(tmp_path / "chunks.npz")
    table = ChunkTable.load(tmp_path / "chunks.npz")

    assert list(table) == _CHUNKS
    assert table[-1] == _CHUNKS[-1]
    assert table[1:] == _CHUNKS[1:]
    assert table.indices_by_file == {"src/a.py": [0, 2], "docs/b.md": [1]}
    assert table.indices_by_language == {"python": [0, 2]}


@pytest.mark.parametrize(
    "corrupt",
    [{"offsets": np.array([0, 30, 21, 41])}, {"file_ids": np.array([0, 9, 0], dtype=np.int32)}, None],
    ids=["decreasing_offsets", "unknown_file", "truncated_file"],
)
def test_load_rejects_corrupt_table(corrupt: dict[str, np.ndarray] | None, tmp_path: Path) -> None:
    """A saved table whose arrays don't line up, or that can't be read, is rejected."""
    path = tmp_path / "chunks.npz"
    ChunkTable.from_chunks(_CHUNKS).save(path)
    if corrupt is None:
        path.write_bytes(path.read_bytes()[:50])
    else:
        with np.load(path) as arrays:
            saved = dict(arrays)
        np.savez(path, **{**saved, **corrupt})

    with pytest.raises(ValueError, match="Persisted"):
        ChunkTable.load(path)
