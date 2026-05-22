from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from model2vec import StaticModel

from semble import SembleIndex
from semble.index.create import _MAX_FILE_BYTES, create_index_from_path
from semble.types import ContentType
from tests.conftest import make_chunk


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
            with patch("semble.index.index.create_index_from_path") as mock_create:
                mock_create.return_value = (MagicMock(), MagicMock(), [make_chunk("x = 1", "f.py")])
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


def test_load_non_existent(tmp_path: Path, indexed_index: SembleIndex) -> None:
    """Test that saving and loading a folder leads to the same data."""
    with pytest.raises(FileNotFoundError):
        SembleIndex.load_from_disk(tmp_path / "temp")


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
