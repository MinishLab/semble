import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import bm25s
import numpy as np
import numpy.typing as npt
import tantivy

from semble.index.chunk_store import LmdbChunkStore
from semble.tokens import tokenize
from semble.types import Chunk, FilterSpec, SearchResult

_TANTIVY_BODY_FIELD = "body"
_TANTIVY_BUILD_HEAP_SIZE = 128_000_000
_TANTIVY_BUILD_THREADS = 4


class SparseIndex(Protocol):
    def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
        """Return sparse search results ranked by descending relevance."""
        ...


def _chunk_id(chunk: Chunk, index: int) -> int:
    return chunk.chunk_id if chunk.chunk_id is not None else index


def _matches_filter_spec(filter_spec: FilterSpec, chunk: Chunk, index: int) -> bool:
    if filter_spec.file_paths is not None and chunk.file_path not in filter_spec.file_paths:
        return False
    if filter_spec.languages is not None and chunk.language not in filter_spec.languages:
        return False
    if filter_spec.chunk_ids is not None and _chunk_id(chunk, index) not in filter_spec.chunk_ids:
        return False
    return True


def filter_spec_to_selector(filter_spec: FilterSpec | None, chunks: Sequence[Chunk]) -> npt.NDArray[np.int_] | None:
    """Convert a backend-neutral filter spec to current list-position selectors."""
    if filter_spec is None:
        return None
    indices = [index for index, chunk in enumerate(chunks) if _matches_filter_spec(filter_spec, chunk, index)]
    return np.array(indices, dtype=np.int_)


def _tantivy_schema() -> tantivy.Schema:
    return (
        tantivy.SchemaBuilder()
        .add_integer_field("chunk_id", indexed=True, stored=True)
        .add_text_field("file_path", stored=True, tokenizer_name="raw")
        .add_text_field("language", stored=True, tokenizer_name="raw")
        .add_text_field(_TANTIVY_BODY_FIELD, tokenizer_name="whitespace")
        .build()
    )


@dataclass(frozen=True, slots=True)
class PreparedTantivyFields:
    body: str


def _prepare_tantivy_fields(chunk: Chunk) -> PreparedTantivyFields:
    return PreparedTantivyFields(body=" ".join(tokenize(enrich_for_bm25(chunk))))


def _tantivy_document_from_fields(
    chunk: Chunk,
    chunk_id: int,
    fields: PreparedTantivyFields,
) -> tantivy.Document:
    return tantivy.Document(
        chunk_id=chunk_id,
        file_path=[chunk.file_path],
        language=[chunk.language or ""],
        body=[fields.body],
    )


def _tantivy_document(chunk: Chunk, chunk_id: int) -> tantivy.Document:
    return _tantivy_document_from_fields(chunk, chunk_id, _prepare_tantivy_fields(chunk))


def _tantivy_term_filter_query(
    schema: tantivy.Schema,
    field_name: str,
    values: frozenset[str] | frozenset[int] | None,
) -> tantivy.Query | None:
    if values is None:
        return None
    terms = sorted(values)
    if not terms:
        return tantivy.Query.empty_query()
    if len(terms) == 1:
        return tantivy.Query.term_query(schema, field_name, terms[0])
    return tantivy.Query.term_set_query(schema, field_name, terms)


def _tantivy_search_query(schema: tantivy.Schema, query: str) -> tantivy.Query | None:
    token_queries = [tantivy.Query.term_query(schema, _TANTIVY_BODY_FIELD, token) for token in tokenize(query)]
    if not token_queries:
        return None
    if len(token_queries) == 1:
        return token_queries[0]
    return tantivy.Query.boolean_query([(tantivy.Occur.Should, query) for query in token_queries])


def _tantivy_filter_query(schema: tantivy.Schema, filter_spec: FilterSpec | None) -> tantivy.Query | None:
    if filter_spec is None:
        return None

    filters = [
        filter_query
        for filter_query in (
            _tantivy_term_filter_query(schema, "file_path", filter_spec.file_paths),
            _tantivy_term_filter_query(schema, "language", filter_spec.languages),
            _tantivy_term_filter_query(schema, "chunk_id", filter_spec.chunk_ids),
        )
        if filter_query is not None
    ]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return tantivy.Query.boolean_query([(tantivy.Occur.Must, query) for query in filters])


@dataclass(slots=True)
class TantivyBuildWriter:
    path: Path | None = None
    chunks: list[Chunk] = field(default_factory=list)
    document_count: int = 0
    index: Any = field(init=False)
    writer: Any = field(init=False)

    def __post_init__(self) -> None:
        """Open the build-time Tantivy writer."""
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
        self.index = tantivy.Index(
            schema=_tantivy_schema(),
            path=None if self.path is None else str(self.path),
            reuse=False,
        )
        self.writer = self.index.writer(heap_size=_TANTIVY_BUILD_HEAP_SIZE, num_threads=_TANTIVY_BUILD_THREADS)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Append chunk documents to the build-time Tantivy writer."""
        self.add_prepared_chunks(chunks, [None] * len(chunks))

    def add_prepared_chunks(
        self,
        chunks: Sequence[Chunk],
        fields: Sequence[PreparedTantivyFields | None],
    ) -> None:
        """Append chunks with optional pre-tokenized Tantivy fields."""
        for chunk_index, (chunk, prepared) in enumerate(zip(chunks, fields), start=self.document_count):
            document = (
                _tantivy_document(chunk, _chunk_id(chunk, chunk_index))
                if prepared is None
                else _tantivy_document_from_fields(chunk, _chunk_id(chunk, chunk_index), prepared)
            )
            self.writer.add_document(document)
        self.document_count += len(chunks)

    def finish(self, chunks: Sequence[Chunk] | None = None) -> "TantivySparseIndex":
        """Commit the build-time Tantivy writer and return a searchable wrapper."""
        self.writer.commit()
        self.index.reload()
        return TantivySparseIndex(self.index, self.chunks if chunks is None else chunks, self.path)


@dataclass(slots=True)
class TantivySparseIndex:
    index: tantivy.Index
    chunks: Sequence[Chunk]
    path: Path | None = None
    chunk_store_path: Path | None = None
    _temporary_path: Path | None = None
    _chunks_by_id: dict[int, Chunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the stable chunk-id lookup used by Tantivy hits."""
        self._chunks_by_id = {_chunk_id(chunk, index): chunk for index, chunk in enumerate(self.chunks)}

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], path: Path | None = None) -> "TantivySparseIndex":
        """Build and commit a Tantivy sparse index for chunks."""
        writer = TantivyBuildWriter(path)
        writer.add_chunks(chunks)
        return writer.finish(chunks)

    @classmethod
    def build_temporary(cls, chunks: Sequence[Chunk]) -> "TantivySparseIndex":
        """Build a temporary on-disk index so save can copy instead of rebuilding."""
        path = Path(tempfile.mkdtemp(prefix="semble-tantivy-"))
        index = cls.from_chunks(chunks, path=path)
        index._temporary_path = path
        return index

    @classmethod
    def load(cls, path: Path, chunks: Sequence[Chunk]) -> "TantivySparseIndex":
        """Open a persisted Tantivy sparse index."""
        index = tantivy.Index.open(str(path))
        index.reload()
        return cls(index, chunks, path)

    @classmethod
    def load_from_store(cls, path: Path, chunk_store_path: Path) -> "TantivySparseIndex":
        """Open a persisted Tantivy index and resolve hit payloads from LMDB."""
        index = tantivy.Index.open(str(path))
        index.reload()
        return cls(index, (), path, chunk_store_path)

    @classmethod
    def load_copy(cls, path: Path, chunks: Sequence[Chunk]) -> "TantivySparseIndex":
        """Open a writable copy of a persisted Tantivy sparse index."""
        copied_path = Path(tempfile.mkdtemp(prefix="semble-tantivy-"))
        shutil.copytree(path, copied_path, dirs_exist_ok=True)
        index = cls.load(copied_path, chunks)
        index._temporary_path = copied_path
        return index

    @classmethod
    def load_copy_from_store(cls, path: Path, chunk_store_path: Path) -> "TantivySparseIndex":
        """Open a writable Tantivy copy that resolves unchanged hits from an existing chunk store."""
        copied_path = Path(tempfile.mkdtemp(prefix="semble-tantivy-"))
        shutil.copytree(path, copied_path, dirs_exist_ok=True)
        index = cls.load_from_store(copied_path, chunk_store_path)
        index._temporary_path = copied_path
        return index

    def save(self, path: Path) -> None:
        """Persist this sparse index at path without rebuilding when a Tantivy directory exists."""
        if self.path is None:
            self.from_chunks(self.chunks, path=path)
            return
        if self.path.resolve() == path.resolve():
            return
        if path.exists():
            shutil.rmtree(path)
        shutil.copytree(self.path, path)

    def __del__(self) -> None:
        """Remove temporary sparse-index copies once the wrapper is released."""
        if self._temporary_path is not None:
            shutil.rmtree(self._temporary_path, ignore_errors=True)

    def _chunk_by_id(self, chunk_id: int) -> Chunk:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is not None:
            return chunk
        if self.chunk_store_path is None:
            raise FileNotFoundError(f"Sparse index is missing chunk payload for id {chunk_id}")
        store = LmdbChunkStore.open(self.chunk_store_path, readonly=True)
        try:
            loaded = store.get_chunk(chunk_id)
        finally:
            store.close()
        if loaded is None:
            raise FileNotFoundError(f"Sparse index is missing chunk payload for id {chunk_id}")
        return loaded

    def update_chunks(
        self,
        chunks: Sequence[Chunk],
        deleted_chunk_ids: set[int],
        added_chunks: Sequence[Chunk],
    ) -> None:
        """Apply Tantivy delete/add updates for changed chunk IDs."""
        chunk_indices: dict[Chunk, int] | None = None
        writer = self.index.writer(heap_size=15_000_000, num_threads=1)
        for chunk_id in deleted_chunk_ids:
            writer.delete_documents("chunk_id", chunk_id)
        for added_index, chunk in enumerate(added_chunks):
            chunk_id = chunk.chunk_id
            if chunk_id is None:
                if chunk_indices is None:
                    chunk_indices = {candidate: index for index, candidate in enumerate(chunks)}
                chunk_id = _chunk_id(chunk, chunk_indices[chunk])
            writer.add_document(_tantivy_document(chunk, chunk_id))
        writer.commit()
        self.index.reload()
        self.chunks = chunks
        if self.chunk_store_path is None:
            self.__post_init__()
        else:
            self._chunks_by_id = {
                chunk.chunk_id if chunk.chunk_id is not None else index: chunk
                for index, chunk in enumerate(added_chunks)
            }

    def search_ids(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[tuple[int, float]]:
        """Return Tantivy BM25 hits as stable chunk IDs without loading chunk payloads."""
        search_query = _tantivy_search_query(self.index.schema, query)
        if search_query is None:
            return []

        searcher = self.index.searcher()
        native_filter = _tantivy_filter_query(self.index.schema, filter_spec)
        native_query = (
            search_query
            if native_filter is None
            else tantivy.Query.boolean_query([(tantivy.Occur.Must, search_query), (tantivy.Occur.Must, native_filter)])
        )
        return [
            (int(searcher.doc(doc_address)["chunk_id"][0]), float(score))
            for score, doc_address in searcher.search(native_query, top_k).hits
        ]

    def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
        """Return Tantivy BM25 results mapped back to chunks."""
        results: list[SearchResult] = []
        for chunk_id, score in self.search_ids(query, top_k, filter_spec):
            results.append(SearchResult(chunk=self._chunk_by_id(chunk_id), score=score))
            if len(results) == top_k:
                break
        return results


@dataclass(frozen=True, slots=True)
class Bm25sSparseIndex:
    bm25_index: bm25s.BM25
    chunks: Sequence[Chunk]

    def _search_positions(self, query: str, top_k: int, filter_spec: FilterSpec | None) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        selector = filter_spec_to_selector(filter_spec, self.chunks)
        mask = selector_to_mask(selector, len(self.chunks))
        scores: npt.NDArray[np.float32] = self.bm25_index.get_scores(tokens, weight_mask=mask)
        indices = _sort_top_k(scores, top_k)
        return [(int(index), float(scores[int(index)])) for index in indices if scores[int(index)] > 0]

    def search_ids(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[tuple[int, float]]:
        """Return bm25s hits as stable chunk IDs."""
        return [
            (_chunk_id(self.chunks[index], index), score)
            for index, score in self._search_positions(query, top_k, filter_spec)
        ]

    def search(self, query: str, top_k: int, filter_spec: FilterSpec | None = None) -> list[SearchResult]:
        """Return bm25s results mapped back to chunks."""
        return [
            SearchResult(chunk=self.chunks[index], score=score)
            for index, score in self._search_positions(query, top_k, filter_spec)
        ]


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def selector_to_mask(selector: npt.NDArray[np.int_] | None, size: int) -> npt.NDArray[np.bool_] | None:
    """Convert a selector array of indices into a boolean mask of length ``size``."""
    if selector is None:
        return None
    mask = np.zeros(size, dtype=bool)
    mask[selector] = True
    return mask


def enrich_for_bm25(chunk: Chunk) -> str:
    """Append file path components to BM25 content to boost path-based queries.

    Assumes ``chunk.file_path`` is already repo-relative (set by ``create_index_from_path``)
    so machine-specific directory components are never indexed.
    """
    path = Path(chunk.file_path)
    stem = path.stem
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])  # Last 3 directory components
    # Repeat the stem twice to up-weight file-path matches in BM25.
    return f"{chunk.content} {stem} {stem} {dir_text}"
