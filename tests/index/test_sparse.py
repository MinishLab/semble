from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from semble.index.chunk_store import LmdbChunkStore
from semble.index.sparse import TantivySparseIndex, filter_spec_to_selector
from semble.types import FilterSpec
from tests.conftest import make_chunk


def test_tantivy_sparse_index_searches_content_and_path_terms(tmp_path: Path) -> None:
    """Tantivy sparse search matches both chunk content and indexed path tokens."""
    chunks = [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "services/auth.py"),
        make_chunk("def format_date(dt):\n    return str(dt)", "utils/date_tools.py"),
    ]

    sparse_index = TantivySparseIndex.from_chunks(chunks, path=tmp_path / "tantivy")

    content_results = sparse_index.search("authenticate token", top_k=2)
    assert content_results[0].chunk is chunks[0]

    path_results = sparse_index.search("date_tools", top_k=2)
    assert path_results[0].chunk is chunks[1]


def test_tantivy_sparse_index_builds_query_from_semble_tokens_without_parse_query() -> None:
    """Tantivy sparse query construction should not re-tokenize through Tantivy's parser."""
    searcher = Mock()
    searcher.search.return_value.hits = []
    index = Mock()
    index.schema = object()
    index.searcher.return_value = searcher
    index.parse_query.side_effect = AssertionError("parse_query should not tokenize")
    sparse_index = TantivySparseIndex(index, [make_chunk("def helper():\n    return True", "date_tools.py")])
    term_queries: list[tuple[str, str, str]] = []

    def fake_term_query(schema: object, field_name: str, field_value: str, index_option: str = "position") -> str:
        assert schema is index.schema
        term_queries.append((field_name, field_value, index_option))
        return f"{field_name}:{field_value}"

    def fake_boost_query(query: object, boost: float) -> tuple[str, object, float]:
        return ("boost", query, boost)

    def fake_disjunction_max_query(subqueries: object) -> tuple[str, object]:
        return ("dismax", subqueries)

    def fake_boolean_query(subqueries: object) -> tuple[str, object]:
        return ("bool", subqueries)

    with (
        patch("semble.index.sparse.tantivy.Query.term_query", side_effect=fake_term_query),
        patch("semble.index.sparse.tantivy.Query.boost_query", side_effect=fake_boost_query),
        patch("semble.index.sparse.tantivy.Query.disjunction_max_query", side_effect=fake_disjunction_max_query),
        patch("semble.index.sparse.tantivy.Query.boolean_query", side_effect=fake_boolean_query),
    ):
        sparse_index.search("date_tools", top_k=1)

    assert ("content", "date_tools", "position") in term_queries
    assert ("path_stem", "date_tools", "position") in term_queries
    assert ("path_dirs", "date_tools", "position") in term_queries
    searcher.search.assert_called_once()
    assert searcher.search.call_args.args[0][0] == "bool"
    assert searcher.search.call_args.args[1] == 1


def test_tantivy_sparse_index_stores_filter_fields(tmp_path: Path) -> None:
    """Tantivy docs should store path and language fields for backend-native filters."""
    chunks = [make_chunk("def authenticate(token):\n    return token == 'secret'", "services/auth.py")]
    sparse_index = TantivySparseIndex.from_chunks(chunks, path=tmp_path / "tantivy")

    searcher = sparse_index.index.searcher()
    query = sparse_index.index.parse_query("authenticate", ["content"])
    _, doc_address = searcher.search(query, 1).hits[0]
    document = searcher.doc(doc_address)

    assert document["file_path"] == ["services/auth.py"]
    assert document["language"] == ["python"]


def test_tantivy_sparse_index_filters_chunk_ids_in_backend(tmp_path: Path) -> None:
    """Chunk-id filters should compose with Tantivy path/language filters."""
    chunks = [
        replace(make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"), chunk_id=10),
        replace(make_chunk("def authenticate_for_test(token):\n    return True", "test_auth.py"), chunk_id=11),
    ]
    sparse_index = TantivySparseIndex.from_chunks(chunks, path=tmp_path / "tantivy")

    with patch("semble.index.sparse._matches_filter_spec", side_effect=AssertionError("should use Tantivy filter")):
        results = sparse_index.search(
            "authenticate token",
            top_k=2,
            filter_spec=FilterSpec(
                file_paths=frozenset({"test_auth.py"}),
                languages=frozenset({"python"}),
                chunk_ids=frozenset({11}),
            ),
        )

    assert [result.chunk for result in results] == [chunks[1]]


def test_tantivy_sparse_index_filters_path_and_language_in_backend(tmp_path: Path) -> None:
    """Path/language filters should use Tantivy fields instead of Python chunk scanning."""
    chunks = [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        make_chunk("def authenticate_for_test(token):\n    return True", "test_auth.py"),
    ]
    sparse_index = TantivySparseIndex.from_chunks(chunks, path=tmp_path / "tantivy")

    with patch("semble.index.sparse._matches_filter_spec", side_effect=AssertionError("should use Tantivy filter")):
        results = sparse_index.search(
            "authenticate token",
            top_k=2,
            filter_spec=FilterSpec(file_paths=frozenset({"test_auth.py"}), languages=frozenset({"python"})),
        )

    assert [result.chunk for result in results] == [chunks[1]]


def test_tantivy_sparse_index_applies_filter_spec(tmp_path: Path) -> None:
    """Tantivy sparse search preserves filtering semantics through FilterSpec."""
    chunks = [
        make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        make_chunk("def authenticate_for_test(token):\n    return True", "test_auth.py"),
    ]
    sparse_index = TantivySparseIndex.from_chunks(chunks, path=tmp_path / "tantivy")

    results = sparse_index.search(
        "authenticate token",
        top_k=2,
        filter_spec=FilterSpec(file_paths=frozenset({"test_auth.py"})),
    )

    assert [result.chunk for result in results] == [chunks[1]]


def test_tantivy_sparse_index_loads_persisted_index(tmp_path: Path) -> None:
    """Persisted Tantivy sparse indexes reload without rebuilding BM25 data."""
    chunks = [make_chunk("def reconcile_account(account_id):\n    return account_id", "recon/accounting.py")]
    index_path = tmp_path / "tantivy"
    TantivySparseIndex.from_chunks(chunks, path=index_path)

    loaded = TantivySparseIndex.load(index_path, chunks)

    results = loaded.search("reconcile account", top_k=1)
    assert results[0].chunk is chunks[0]


def test_tantivy_sparse_index_loads_hit_chunks_from_lmdb_without_bulk_chunk_list(tmp_path: Path) -> None:
    """Hot sparse search should fetch only hit chunk payloads from LMDB."""
    chunks = [
        replace(make_chunk("def authenticate(token):\n    return token", "auth.py"), chunk_id=10),
        replace(make_chunk("def reconcile_account(account_id):\n    return account_id", "recon.py"), chunk_id=11),
    ]
    index_path = tmp_path / "tantivy"
    store_path = tmp_path / "chunks.lmdb"
    TantivySparseIndex.from_chunks(chunks, path=index_path)
    store = LmdbChunkStore.open(store_path)
    try:
        store.write_chunks(chunks)
    finally:
        store.close()

    with patch.object(LmdbChunkStore, "get_chunks", side_effect=AssertionError("bulk chunks should not load")):
        loaded = TantivySparseIndex.load_from_store(index_path, store_path)
        results = loaded.search("authenticate token", top_k=1)

    assert [result.chunk for result in results] == [chunks[0]]


def test_tantivy_sparse_index_updates_changed_chunks_without_rebuild(tmp_path: Path) -> None:
    """Tantivy sparse updates should tombstone deleted chunk IDs and add replacement chunks."""
    auth = replace(make_chunk("def authenticate(token):\n    return token", "auth.py"), chunk_id=1)
    old_utils = replace(
        make_chunk("def obsolete_unique_marker(first, last):\n    return first + last", "utils.py"),
        chunk_id=2,
    )
    new_utils = replace(make_chunk("def changed_name(first, last):\n    return last + first", "utils.py"), chunk_id=3)
    sparse_index = TantivySparseIndex.from_chunks([auth, old_utils], path=tmp_path / "tantivy")

    sparse_index.update_chunks([auth, new_utils], deleted_chunk_ids={2}, added_chunks=[new_utils])

    assert sparse_index.search("obsolete_unique_marker", top_k=1) == []
    assert sparse_index.search("changed_name", top_k=1)[0].chunk is new_utils
    assert sparse_index.search("authenticate", top_k=1)[0].chunk is auth


def test_tantivy_sparse_index_save_copies_updated_index_without_rebuild(tmp_path: Path) -> None:
    """Saving an updated Tantivy index should persist the delete/add result without rebuilding all chunks."""
    auth = replace(make_chunk("def authenticate(token):\n    return token", "auth.py"), chunk_id=1)
    old_utils = replace(make_chunk("def obsolete_unique_marker():\n    return 'old'", "utils.py"), chunk_id=2)
    new_utils = replace(make_chunk("def changed_name():\n    return 'new'", "utils.py"), chunk_id=3)
    source_path = tmp_path / "tantivy"
    saved_path = tmp_path / "saved"
    sparse_index = TantivySparseIndex.from_chunks([auth, old_utils], path=source_path)
    sparse_index.update_chunks([auth, new_utils], deleted_chunk_ids={2}, added_chunks=[new_utils])

    with patch.object(TantivySparseIndex, "from_chunks", side_effect=AssertionError("save should not rebuild")):
        sparse_index.save(saved_path)

    loaded = TantivySparseIndex.load(saved_path, [auth, new_utils])
    assert loaded.search("obsolete_unique_marker", top_k=1) == []
    assert loaded.search("changed_name", top_k=1)[0].chunk is new_utils


def test_filter_spec_chunk_ids_fall_back_to_chunk_positions() -> None:
    """Selector conversion preserves legacy position selectors when chunks lack stable IDs."""
    chunks = [make_chunk("a = 1", "a.py"), make_chunk("b = 1", "b.py")]

    selector = filter_spec_to_selector(FilterSpec(chunk_ids=frozenset({1})), chunks)

    np.testing.assert_array_equal(selector, np.array([1]))
