import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import orjson
import pytest
from model2vec import StaticModel
from vicinity.backends.basic import BasicArgs

from semble import SembleIndex
from semble.cache import GIT_CACHE_ROOTS_VERSION, get_validated_cache
from semble.index.chunk_store import LmdbChunkStore, _int_key
from semble.index.create import IndexBuild, create_index_build_from_path, create_index_from_path
from semble.index.dense import SelectableBasicBackend
from semble.index.files import _MAX_FILE_BYTES, FileStatus, get_file_status
from semble.index.source_inventory import GitWalkPlan
from semble.index.sparse import TantivySparseIndex
from semble.types import ContentType
from tests.conftest import make_chunk

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "t@t.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "t@t.com",
}


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)


def _git_commit_all(path: Path, message: str = "commit") -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, env=_GIT_ENV)
    subprocess.run(["git", "-C", str(path), "commit", "-m", message], check=True, capture_output=True, env=_GIT_ENV)


def _git_head(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True)
    return result.stdout.decode().strip()


@pytest.fixture
def indexed_index(mock_model: Any, tmp_project: Path) -> SembleIndex:
    """SembleIndex built from tmp_project."""
    with patch("semble.index.index.load_model", return_value=(mock_model, "")):
        return SembleIndex.from_path(tmp_project)


@pytest.mark.parametrize(
    ("content", "md_in_results"),
    [
        ([ContentType.CODE], False),
        ([ContentType.DOCS], True),
        ([ContentType.CODE, ContentType.DOCS], True),
    ],
)
def test_index_markdown_inclusion(
    mock_model: StaticModel, tmp_project: Path, content: list[ContentType], md_in_results: bool
) -> None:
    """Markdown files are excluded for code-only and included when docs is requested."""
    _, _, chunks = create_index_from_path(tmp_project, mock_model, content=content)
    has_md = ".md" in {Path(c.file_path).suffix for c in chunks}
    assert has_md is md_in_results


def test_create_index_chunks_files_concurrently_without_changing_order(mock_model: StaticModel, tmp_path: Path) -> None:
    """Concurrent chunking must preserve serial walk order and per-file chunk order."""
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n")
    (tmp_path / "c.py").write_text("def c():\n    return 3\n")
    call_order: list[str] = []

    def fake_chunk_source(source: str, file_path: str, language: str | None):
        call_order.append(file_path)
        return [make_chunk(f"first {file_path}", file_path), make_chunk(f"second {file_path}", file_path)]

    with (
        patch("semble.index.create._CHUNK_WORKER_COUNT", 2),
        patch("semble.index.create.chunk_source", side_effect=fake_chunk_source),
    ):
        _, _, chunks = create_index_from_path(tmp_path, mock_model, display_root=tmp_path)

    assert [chunk.content for chunk in chunks] == [
        "first a.py",
        "second a.py",
        "first b.py",
        "second b.py",
        "first c.py",
        "second c.py",
    ]
    assert set(call_order) == {"a.py", "b.py", "c.py"}


def test_create_index_build_uses_tantivy_sparse_and_collects_file_sizes(
    mock_model: StaticModel, tmp_path: Path
) -> None:
    """Full cold build should use Tantivy sparse index and reuse first-read file sizes."""
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return token\n")

    build = create_index_build_from_path(tmp_path, mock_model, display_root=tmp_path)

    assert isinstance(build, IndexBuild)
    assert isinstance(build.sparse_index, TantivySparseIndex)
    assert build.file_sizes == {"auth.py": len("def authenticate(token):\n    return token\n")}
    assert create_index_from_path(tmp_path, mock_model, display_root=tmp_path)[0].__class__ is TantivySparseIndex


def test_create_index_build_records_file_hashes(mock_model: StaticModel, tmp_path: Path) -> None:
    """Full builds should persist content hashes so changed rebuilds can reuse unchanged work."""
    source = "def authenticate(token):\n    return token\n"
    (tmp_path / "auth.py").write_text(source)

    build = create_index_build_from_path(tmp_path, mock_model, display_root=tmp_path)

    assert build.file_hashes == {"auth.py": hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()}


def test_include_text_files_deprecated(mock_model: Any, tmp_project: Path) -> None:
    """include_text_files=True warns and expands to all content types; False warns and resets to code-only."""
    from semble.index.index import _ALL_CONTENT, _DEFAULT_CONTENT

    with patch("semble.index.index.load_model", return_value=(mock_model, "")):
        with pytest.warns(DeprecationWarning, match="include_text_files is deprecated"):
            idx = SembleIndex.from_path(tmp_project, include_text_files=True)
        assert idx._content == _ALL_CONTENT

        with pytest.warns(DeprecationWarning, match="include_text_files is deprecated"):
            idx = SembleIndex.from_path(tmp_project, include_text_files=False)
        assert idx._content == _DEFAULT_CONTENT


def test_from_git_include_text_files_deprecated(mock_model: Any, tmp_project: Path) -> None:
    """from_git raises DeprecationWarning when include_text_files is passed."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    with patch("semble.index.index.load_model", return_value=(mock_model, "")):
        with patch("subprocess.run", return_value=fake_result):
            with patch("semble.index.index.create_index_build_from_path") as mock_create:
                mock_create.return_value = IndexBuild(MagicMock(), MagicMock(), [make_chunk("x = 1", "f.py")], {})
                with pytest.warns(DeprecationWarning, match="include_text_files is deprecated"):
                    SembleIndex.from_git("https://example.com/repo", include_text_files=True)


def test_index_empty_returns_zero_chunks(mock_model: StaticModel, tmp_path: Path) -> None:
    """Indexing an empty directory yields zero files and chunks."""
    with pytest.raises(ValueError):
        create_index_from_path(tmp_path, mock_model)


def test_oversized_file_is_skipped(mock_model: StaticModel, tmp_path: Path) -> None:
    """Files exceeding _MAX_FILE_BYTES are silently skipped during indexing."""
    (tmp_path / "big.py").write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError):  # no indexable content remains
        create_index_from_path(tmp_path, mock_model)


def test_tiny_invalid_utf8_file_status_does_not_crash(tmp_path: Path) -> None:
    """Tiny files with invalid UTF-8 bytes are treated as non-empty."""
    path = tmp_path / "latin1.py"
    path.write_bytes(b"\xff")
    assert get_file_status(path, None) is FileStatus.VALID


def test_index_language_counts(indexed_index: SembleIndex) -> None:
    """Language breakdown in stats includes python with at least one chunk."""
    stats = indexed_index.stats
    assert "python" in stats.languages
    assert stats.languages["python"] > 0


@pytest.mark.parametrize(
    "query",
    [("authenticate token"), ("authenticate"), ("authentication")],
)
def test_search_modes(indexed_index: SembleIndex, query: str) -> None:
    """Each search mode returns a valid list of at most top_k results."""
    results = indexed_index.search(query, top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_constraints(indexed_index: SembleIndex) -> None:
    """search: top_k is respected; no duplicate chunks are returned."""
    assert len(indexed_index.search("function", top_k=1)) <= 1

    results = indexed_index.search("authenticate", top_k=5)
    assert len(results) == len(set(r.chunk for r in results))


def test_search_with_filter_paths_does_not_crash(indexed_index: SembleIndex) -> None:
    """Filtered search works regardless of where the selected chunk lives in the corpus."""
    target_path = indexed_index.chunks[-1].file_path
    results = indexed_index.search("function", top_k=3, filter_paths=[target_path])
    assert all(r.chunk.file_path == target_path for r in results)


def test_search_without_reranking(indexed_index: SembleIndex) -> None:
    """Filtered search works regardless of where the selected chunk lives in the corpus."""
    with patch("semble.search.rerank_topk") as mock:
        indexed_index.search("function", top_k=3, rerank=False)
        mock.assert_not_called()
    with patch("semble.search.rerank_topk") as mock:
        indexed_index.search("function", top_k=3, rerank=True)
        mock.assert_called()


@pytest.mark.parametrize(
    ("content", "expect_rerank"),
    [
        ([ContentType.CODE], True),
        ([ContentType.CODE, ContentType.DOCS], True),
        ([ContentType.DOCS], False),
        ([ContentType.CONFIG], False),
    ],
)
def test_search_rerank_default_by_content_type(
    mock_model: Any, content: list[ContentType], expect_rerank: bool
) -> None:
    """Reranking is on by default when code is indexed, off for non-code-only content."""
    index = SembleIndex(mock_model, MagicMock(), MagicMock(), [make_chunk("x = 1", "f.py")], "", content=content)
    with patch("semble.index.index.search", return_value=[]) as mock_search:
        index.search("function", top_k=3)
    assert mock_search.call_args.kwargs["rerank"] == expect_rerank


@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_search_empty_query_returns_empty(indexed_index: SembleIndex, query: str) -> None:
    """Empty / whitespace-only queries return [] across all modes."""
    assert indexed_index.search(query) == []


@pytest.mark.parametrize(
    ("disk_files", "chunk_paths", "expected"),
    [
        ({"foo.py": "hello world"}, ["foo.py", "foo.py"], {"foo.py": 11}),
        ({}, ["nonexistent.py"], {}),
    ],
    ids=["dedup-same-file", "missing-file-skipped"],
)
def test_compute_file_sizes(
    tmp_path: Path, disk_files: dict[str, str], chunk_paths: list[str], expected: dict[str, int]
) -> None:
    """_compute_file_sizes deduplicates paths and silently skips missing files."""
    for name, content in disk_files.items():
        (tmp_path / name).write_text(content)
    index = SembleIndex.__new__(SembleIndex)
    index.chunks = [make_chunk("c", p) for p in chunk_paths]
    assert index._compute_file_sizes(tmp_path) == expected


def test_find_related(indexed_index: SembleIndex) -> None:
    """find_related returns related chunks for a Chunk or SearchResult seed."""
    chunk = indexed_index.chunks[0]
    via_chunk = indexed_index.find_related(chunk, top_k=3)
    assert isinstance(via_chunk, list)
    assert len(via_chunk) <= 3
    assert all(r.chunk != chunk for r in via_chunk)

    # SearchResult form returns the same results as Chunk form.
    result = indexed_index.search("authenticate", top_k=1)[0]
    assert [r.chunk for r in indexed_index.find_related(result, top_k=3)] == [
        r.chunk for r in indexed_index.find_related(result.chunk, top_k=3)
    ]


def test_roundtrip(tmp_path: Path, indexed_index: SembleIndex) -> None:
    """Test that saving and loading a folder leads to the same data."""
    indexed_index.save(tmp_path)
    with patch.object(StaticModel, "from_pretrained"):
        index_2 = SembleIndex.load_from_disk(tmp_path)
    assert index_2.chunks == indexed_index.chunks
    assert index_2._root == indexed_index._root


def test_save_uses_lmdb_chunks_and_tantivy_sparse_backend(
    tmp_path: Path, indexed_index: SembleIndex, mock_model: StaticModel
) -> None:
    """Saved full-cold indexes should avoid chunks.json and persist Tantivy sparse metadata."""
    indexed_index.save(tmp_path)

    metadata = orjson.loads((tmp_path / "metadata.json").read_bytes())
    assert metadata["sparse_backend"] == "tantivy"
    assert metadata["chunk_ids"] == list(range(len(indexed_index.chunks)))
    assert not (tmp_path / "chunks.json").exists()
    assert (tmp_path / "chunks.lmdb").exists()

    with patch.object(LmdbChunkStore, "get_chunks", side_effect=AssertionError("load should not bulk-read chunks")):
        with patch("semble.index.index.load_model", return_value=(mock_model, indexed_index._model_path)):
            loaded = SembleIndex.load_from_disk(tmp_path)
    assert isinstance(loaded._bm25_index, TantivySparseIndex)
    assert loaded.search("authenticate", top_k=1)


def test_loaded_stats_use_mappings_without_loading_chunk_payloads(
    tmp_path: Path, indexed_index: SembleIndex, mock_model: StaticModel
) -> None:
    """Stats for persisted indexes should use metadata mappings, not lazy chunk payload reads."""
    indexed_index.save(tmp_path)
    expected_stats = indexed_index.stats

    with patch("semble.index.index.load_model", return_value=(mock_model, indexed_index._model_path)):
        loaded = SembleIndex.load_from_disk(tmp_path)

    with patch.object(loaded.chunks, "chunk_by_id", side_effect=AssertionError("stats should not load chunks")):
        stats = loaded.stats

    assert stats == expected_stats


def test_loaded_chunks_by_id_batches_payload_reads(
    tmp_path: Path, indexed_index: SembleIndex, mock_model: StaticModel
) -> None:
    """Lazy chunk batch lookup should avoid per-id payload reads through chunk_by_id."""
    indexed_index.save(tmp_path)

    with patch("semble.index.index.load_model", return_value=(mock_model, indexed_index._model_path)):
        loaded = SembleIndex.load_from_disk(tmp_path)

    chunk_ids = [0, min(1, len(indexed_index.chunks) - 1)]
    expected = [indexed_index.chunks[chunk_id] for chunk_id in chunk_ids]
    with patch.object(
        loaded.chunks,
        "chunk_by_id",
        side_effect=AssertionError("batch lookup should not call chunk_by_id"),
    ):
        chunks = loaded.chunks.chunks_by_id(chunk_ids)

    assert chunks == expected


def test_save_writes_git_cache_metadata_for_hot_validation(tmp_path: Path, mock_model: StaticModel) -> None:
    """Fresh full saves in git repos should include HEAD metadata for hot cache validation."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate(token):\n    return token\n")
    _init_git_repo(repo)
    _git_commit_all(repo)

    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        index = SembleIndex.from_path(repo)
    save_path = tmp_path / "index"
    index.save(save_path)

    metadata = orjson.loads((save_path / "metadata.json").read_bytes())
    assert metadata["git_roots_version"] == GIT_CACHE_ROOTS_VERSION
    assert metadata["git_roots"] == [{"path": "", "head": _git_head(repo)}]
    assert metadata["tracked_paths"] == ["auth.py"]
    with patch("semble.cache.find_index_from_cache_folder", return_value=save_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git metadata should avoid full walk")):
            assert get_validated_cache(str(repo), "mock-model", [ContentType.CODE]) == save_path


def test_full_save_replaces_lmdb_chunk_store(tmp_path: Path, mock_model: StaticModel) -> None:
    """Full saves should not leave stale chunk payloads in an existing LMDB store."""
    first = make_chunk("def authenticate(token):\n    return token", "auth.py")
    stale = make_chunk("def stale():\n    return None", "stale.py")
    first_index = SembleIndex(
        mock_model,
        TantivySparseIndex.build_temporary([first, stale]),
        SelectableBasicBackend(np.ones((2, mock_model.dim), dtype=np.float32), BasicArgs()),
        [first, stale],
        "mock-model",
    )
    first_index.save(tmp_path)

    second_index = SembleIndex(
        mock_model,
        TantivySparseIndex.build_temporary([first]),
        SelectableBasicBackend(np.ones((1, mock_model.dim), dtype=np.float32), BasicArgs()),
        [first],
        "mock-model",
    )
    second_index.save(tmp_path)

    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)
    try:
        assert store.get_chunk(0) == first
        assert store.get_chunk(1) is None
    finally:
        store.close()


def test_full_save_keeps_existing_lmdb_chunk_store_when_rewrite_fails(tmp_path: Path, mock_model: StaticModel) -> None:
    """Failed full saves should not replace a valid existing LMDB store with an incomplete one."""
    first = make_chunk("def authenticate(token):\n    return token", "auth.py")
    stale = make_chunk("def stale():\n    return None", "stale.py")
    existing_index = SembleIndex(
        mock_model,
        TantivySparseIndex.build_temporary([first, stale]),
        SelectableBasicBackend(np.ones((2, mock_model.dim), dtype=np.float32), BasicArgs()),
        [first, stale],
        "mock-model",
    )
    existing_index.save(tmp_path)

    replacement_index = SembleIndex(
        mock_model,
        TantivySparseIndex.build_temporary([first]),
        SelectableBasicBackend(np.ones((1, mock_model.dim), dtype=np.float32), BasicArgs()),
        [first],
        "mock-model",
    )
    with (
        patch.object(LmdbChunkStore, "write_chunks_with_ids", side_effect=RuntimeError("write failed")),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        replacement_index.save(tmp_path)

    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb", readonly=True)
    try:
        assert store.get_chunk(0) == first
        assert store.get_chunk(1) == stale
    finally:
        store.close()


def test_save_preserves_stable_chunk_ids_for_loaded_search(tmp_path: Path, mock_model: StaticModel) -> None:
    """Saved Tantivy and LMDB IDs must match when chunks already carry stable chunk_id values."""
    chunk = replace(make_chunk("def authenticate(token):\n    return token", "auth.py"), chunk_id=10)
    semantic_index = SelectableBasicBackend(np.ones((1, mock_model.dim), dtype=np.float32), BasicArgs())
    index = SembleIndex(
        mock_model,
        TantivySparseIndex.build_temporary([chunk]),
        semantic_index,
        [chunk],
        "mock-model",
    )

    index.save(tmp_path)

    metadata = orjson.loads((tmp_path / "metadata.json").read_bytes())
    assert metadata["chunk_ids"] == [10]
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        loaded = SembleIndex.load_from_disk(tmp_path)

    assert loaded.search("authenticate token", top_k=1)[0].chunk == chunk


def test_loaded_tantivy_index_fails_loud_when_lmdb_chunk_payload_is_missing(
    tmp_path: Path, indexed_index: SembleIndex, mock_model: StaticModel
) -> None:
    """Tantivy cache corruption should not silently return fewer sparse results."""
    indexed_index.save(tmp_path)
    metadata = orjson.loads((tmp_path / "metadata.json").read_bytes())
    missing_id = int(metadata["chunk_ids"][0])
    store = LmdbChunkStore.open(tmp_path / "chunks.lmdb")
    try:
        with store.env.begin(write=True) as txn:
            txn.delete(_int_key(missing_id), db=store.chunks_db)
    finally:
        store.close()

    with patch("semble.index.index.load_model", return_value=(mock_model, indexed_index._model_path)):
        loaded = SembleIndex.load_from_disk(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing chunk payload"):
        loaded._bm25_index.search("authenticate", top_k=1)


def test_load_from_disk_fails_loud_when_lmdb_chunk_payload_is_missing(tmp_path: Path, mock_model: StaticModel) -> None:
    """LMDB chunk_ids metadata must not load when payloads are missing."""
    index_path = tmp_path / "bm25-lmdb"
    index_path.mkdir()
    (index_path / "bm25_index").mkdir()
    (index_path / "semantic_index").mkdir()
    chunk = make_chunk("def authenticate(token):\n    return token", "auth.py")
    store = LmdbChunkStore.open(index_path / "chunks.lmdb")
    try:
        store.write_chunks_with_ids([chunk], [0])
    finally:
        store.close()
    (index_path / "metadata.json").write_bytes(
        orjson.dumps(
            {
                "root_path": None,
                "time": 1.0,
                "model_path": "mock-model",
                "content_type": ["code"],
                "file_paths": ["auth.py"],
                "chunk_ids": [0, 1],
                "sparse_backend": "bm25s",
            }
        )
    )

    with (
        patch("semble.index.index.BM25.load", return_value=MagicMock()),
        patch("semble.index.index.SelectableBasicBackend.load", return_value=MagicMock()),
        patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")),
        pytest.raises(FileNotFoundError, match="missing chunk payloads"),
    ):
        SembleIndex.load_from_disk(index_path)


def test_load_save_roundtrip_preserves_manifest(tmp_path: Path, indexed_index: SembleIndex) -> None:
    """load_from_disk followed by save must not clobber file_paths with an empty list."""
    save_a = tmp_path / "a"
    save_b = tmp_path / "b"
    indexed_index.save(save_a)
    with patch.object(StaticModel, "from_pretrained"):
        loaded = SembleIndex.load_from_disk(save_a)
    loaded.save(save_b)
    import json

    manifest_a = json.loads((save_a / "metadata.json").read_text())["file_paths"]
    manifest_b = json.loads((save_b / "metadata.json").read_text())["file_paths"]
    assert manifest_b == manifest_a
    assert len(manifest_b) > 0


def test_load_non_existent(tmp_path: Path, indexed_index: SembleIndex) -> None:
    """Test that saving and loading a folder leads to the same data."""
    with pytest.raises(FileNotFoundError):
        SembleIndex.load_from_disk(tmp_path / "temp")


def test_legacy_bm25_chunks_json_cache_still_loads(tmp_path: Path, indexed_index: SembleIndex) -> None:
    """Legacy bm25s/chunks.json caches should remain loadable after Tantivy becomes default."""
    paths = tmp_path / "legacy"
    paths.mkdir()
    indexed_index._bm25_index = MagicMock()
    (paths / "bm25_index").write_text("legacy")
    indexed_index._semantic_index.save(paths / "semantic_index")
    (paths / "chunks.json").write_bytes(orjson.dumps([chunk.to_dict() for chunk in indexed_index.chunks]))
    (paths / "metadata.json").write_bytes(
        orjson.dumps(
            {
                "root_path": None,
                "time": 1.0,
                "model_path": indexed_index._model_path,
                "content_type": [item.value for item in indexed_index._content],
                "file_paths": sorted(indexed_index._file_mapping),
            }
        )
    )

    with (
        patch("semble.index.index.BM25.load", return_value=indexed_index._bm25_index),
        patch.object(StaticModel, "from_pretrained"),
    ):
        loaded = SembleIndex.load_from_disk(paths)

    assert loaded.chunks == indexed_index.chunks
    assert loaded._bm25_index is indexed_index._bm25_index


def test_load_from_disk_missing_files_reports_them(tmp_path: Path) -> None:
    """When the directory exists but required index files are missing, the error lists them."""
    index_dir = tmp_path / "incomplete_index"
    index_dir.mkdir()
    # Create only one of the four expected files so the rest are reported as missing.
    (index_dir / "chunks.json").write_text("[]")

    with pytest.raises(FileNotFoundError, match="Missing:") as exc_info:
        SembleIndex.load_from_disk(index_dir)

    error_msg = str(exc_info.value)
    # The three missing files should all appear in the error message.
    assert "bm25_index" in error_msg
    assert "semantic_index" in error_msg
    assert "metadata.json" in error_msg
    # The file we did create should NOT be listed as missing.
    assert "chunks.json" not in error_msg


def test_from_path_uses_cache_when_valid(tmp_project: Path) -> None:
    """from_path returns the cached index directly when get_validated_cache hits."""
    fake_cached = MagicMock(spec=SembleIndex)
    with patch("semble.index.index.get_validated_cache", return_value=tmp_project / "cache"):
        with patch.object(SembleIndex, "load_from_disk", return_value=fake_cached):
            result = SembleIndex.from_path(tmp_project)
    assert result is fake_cached


def test_from_path_rebuilds_changed_files_from_hash_seed(tmp_path: Path, mock_model: StaticModel) -> None:
    """Changed-source rebuilds should reuse unchanged chunk vectors and embed only changed files."""
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return token\n")
    (tmp_path / "utils.py").write_text("def format_date(dt):\n    return str(dt)\n")
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        original = SembleIndex.from_path(tmp_path)
    cache_path = tmp_path / "cache"
    original.save(cache_path)
    original_hashes = dict(original._file_hashes)

    (tmp_path / "utils.py").write_text("def changed_format(dt):\n    return repr(dt)\n")

    def fake_embed(model: StaticModel, chunks: list[Any]):
        assert {chunk.file_path for chunk in chunks} == {"utils.py"}
        return np.zeros((len(chunks), model.dim), dtype=np.float32)

    with (
        patch("semble.index.index.get_validated_cache", return_value=None),
        patch("semble.index.index.get_rebuild_cache", return_value=cache_path),
        patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")),
        patch("semble.index.index.embed_chunks", side_effect=fake_embed),
    ):
        rebuilt = SembleIndex.from_path(tmp_path)

    assert rebuilt.loaded_from_disk is False
    assert rebuilt.stats.indexed_files == 2
    assert rebuilt.stats.total_chunks == original.stats.total_chunks
    assert rebuilt._file_hashes["auth.py"] == original_hashes["auth.py"]
    assert rebuilt._file_hashes["utils.py"] != original_hashes["utils.py"]
    assert any(chunk.file_path == "auth.py" for chunk in rebuilt.chunks)
    assert any("changed_format" in chunk.content for chunk in rebuilt.chunks if chunk.file_path == "utils.py")


def test_from_path_rebuilds_changed_files_from_git_plan_without_full_walk(
    tmp_path: Path,
    mock_model: StaticModel,
) -> None:
    """Git plans should let changed rebuilds avoid full filesystem walks."""
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return token\n")
    (tmp_path / "utils.py").write_text("def format_date(dt):\n    return str(dt)\n")
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        original = SembleIndex.from_path(tmp_path)
    cache_path = tmp_path / "cache"
    original.save(cache_path)
    metadata_path = cache_path / "metadata.json"
    metadata = orjson.loads(metadata_path.read_bytes())
    metadata["git_roots"] = [
        {"path": "", "head": "old-head"},
        {"path": "empty-nested", "head": "old-empty-head"},
    ]
    metadata["git_roots_version"] = GIT_CACHE_ROOTS_VERSION
    metadata["tracked_paths"] = ["auth.py", "utils.py"]
    metadata_path.write_bytes(orjson.dumps(metadata))
    (tmp_path / "utils.py").write_text("def changed_format(dt):\n    return repr(dt)\n")
    plan = GitWalkPlan(
        current_paths=("auth.py", "utils.py"),
        changed_paths=frozenset({"utils.py"}),
        deleted_paths=frozenset(),
        source_roots=(),
        git_cache_metadata=({"path": "", "head": "old-head"},),
    )

    with (
        patch("semble.index.index.get_validated_cache", return_value=None),
        patch("semble.index.index.get_rebuild_cache", return_value=cache_path),
        patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")),
        patch("semble.index.index.build_git_walk_plan", return_value=plan),
        patch.object(LmdbChunkStore, "get_chunks", side_effect=AssertionError("git plan should avoid seed chunk load")),
        patch("semble.index.index.TantivySparseIndex.build_temporary") as build_temporary,
        patch("semble.index.index.walk_files", side_effect=AssertionError("git plan should avoid full walk")),
    ):
        rebuilt = SembleIndex.from_path(tmp_path)

    build_temporary.assert_not_called()
    with patch.object(rebuilt.chunks, "chunk_by_id", side_effect=AssertionError("save should not walk lazy chunks")):
        rebuilt.save(tmp_path / "rebuilt-cache")
    assert rebuilt.stats.indexed_files == 2
    assert any("changed_format" in chunk.content for chunk in rebuilt.chunks if chunk.file_path == "utils.py")
    assert rebuilt.search("changed_format", top_k=1)
    assert rebuilt._git_cache_metadata == (
        {"path": "", "head": "old-head"},
        {"path": "empty-nested", "head": "old-empty-head"},
    )


def test_git_plan_lazy_save_removes_deleted_chunk_payloads(
    tmp_path: Path,
    mock_model: StaticModel,
) -> None:
    """Lazy incremental saves should remove payloads for files deleted from the rebuilt index."""
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return token\n")
    (tmp_path / "deleted.py").write_text("def deleted():\n    return 'stale'\n")
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        original = SembleIndex.from_path(tmp_path)
    cache_path = tmp_path / "cache"
    original.save(cache_path)
    metadata_path = cache_path / "metadata.json"
    metadata = orjson.loads(metadata_path.read_bytes())
    deleted_chunk_ids = [
        chunk_id
        for chunk_id, file_path in zip(metadata["chunk_ids"], metadata["chunk_file_paths"])
        if file_path == "deleted.py"
    ]
    assert deleted_chunk_ids
    metadata["git_roots"] = [{"path": "", "head": "old-head"}]
    metadata["git_roots_version"] = GIT_CACHE_ROOTS_VERSION
    metadata["tracked_paths"] = ["auth.py", "deleted.py"]
    metadata_path.write_bytes(orjson.dumps(metadata))
    (tmp_path / "deleted.py").unlink()
    plan = GitWalkPlan(
        current_paths=("auth.py",),
        changed_paths=frozenset(),
        deleted_paths=frozenset({"deleted.py"}),
        source_roots=(),
        git_cache_metadata=({"path": "", "head": "old-head"},),
    )

    with (
        patch("semble.index.index.get_validated_cache", return_value=None),
        patch("semble.index.index.get_rebuild_cache", return_value=cache_path),
        patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")),
        patch("semble.index.index.build_git_walk_plan", return_value=plan),
    ):
        rebuilt = SembleIndex.from_path(tmp_path)

    saved_path = tmp_path / "rebuilt-cache"
    rebuilt.save(saved_path)
    store = LmdbChunkStore.open(saved_path / "chunks.lmdb", readonly=True)
    try:
        assert all(store.get_chunk(chunk_id) is None for chunk_id in deleted_chunk_ids if chunk_id is not None)
    finally:
        store.close()
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        assert all(chunk.file_path != "deleted.py" for chunk in SembleIndex.load_from_disk(saved_path).chunks)


def test_from_path_rejects_git_plan_missing_root_with_cached_files(
    tmp_path: Path,
    mock_model: StaticModel,
) -> None:
    """Git plans must not ignore missing source roots that still own cached files."""
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return token\n")
    service = tmp_path / "service"
    service.mkdir()
    (service / "lib.py").write_text("def service_lib():\n    return 1\n")
    with patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")):
        original = SembleIndex.from_path(tmp_path)
    cache_path = tmp_path / "cache"
    original.save(cache_path)
    metadata_path = cache_path / "metadata.json"
    metadata = orjson.loads(metadata_path.read_bytes())
    metadata["git_roots"] = [
        {"path": "", "head": "old-head"},
        {"path": "service", "head": "old-service-head"},
    ]
    metadata["git_roots_version"] = GIT_CACHE_ROOTS_VERSION
    metadata_path.write_bytes(orjson.dumps(metadata))
    plan = GitWalkPlan(
        current_paths=("auth.py", "service/lib.py"),
        changed_paths=frozenset(),
        deleted_paths=frozenset(),
        source_roots=(),
        git_cache_metadata=({"path": "", "head": "old-head"},),
    )

    with (
        patch("semble.index.index.get_validated_cache", return_value=None),
        patch("semble.index.index.get_rebuild_cache", return_value=cache_path),
        patch("semble.index.index.load_model", return_value=(mock_model, "mock-model")),
        patch("semble.index.index.build_git_walk_plan", return_value=plan),
        patch("semble.index.index.walk_files", side_effect=AssertionError("must not fallback to full walk")),
        pytest.raises(AssertionError, match="must not fallback"),
    ):
        SembleIndex.from_path(tmp_path)


@pytest.mark.parametrize("ref", [None, "v1.0"])
def test_from_git_uses_cache_when_valid(ref: str | None) -> None:
    """from_git uses the cache for both URL-only and URL@ref cache keys."""
    fake_cached = MagicMock(spec=SembleIndex)
    with patch("semble.index.index.get_validated_cache", return_value=Path("/cache")):
        with patch.object(SembleIndex, "load_from_disk", return_value=fake_cached):
            result = SembleIndex.from_git("https://github.com/org/repo.git", ref=ref)
    assert result is fake_cached
