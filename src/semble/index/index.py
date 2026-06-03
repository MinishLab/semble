from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson
from bm25s import BM25
from model2vec.model import StaticModel

from semble.cache import GIT_CACHE_ROOTS_VERSION, build_git_cache_metadata, get_validated_cache
from semble.index.chunk_store import LmdbChunkStore
from semble.index.create import create_index_build_from_path
from semble.index.dense import SelectableBasicBackend, load_model
from semble.index.files import read_file_text
from semble.index.sparse import SparseIndex, TantivySparseIndex
from semble.index.types import PersistencePath
from semble.search import _search_semantic, search
from semble.stats import save_search_stats
from semble.types import CallType, Chunk, ContentType, FilterSpec, IndexStats, SearchResult

_GIT_CLONE_TIMEOUT = int(os.environ.get("SEMBLE_CLONE_TIMEOUT", 60))
_DEFAULT_CONTENT: tuple[ContentType, ...] = (ContentType.CODE,)
_ALL_CONTENT: tuple[ContentType, ...] = (ContentType.CODE, ContentType.DOCS, ContentType.CONFIG)
_INCLUDE_TEXT_FILES_DEPRECATION_MSG = (
    "include_text_files is deprecated and will be removed in a future version. "
    "Use content=(ContentType.CODE, ContentType.DOCS, ContentType.CONFIG) instead."
)


def _apply_include_text_files(
    content: ContentType | Sequence[ContentType], include_text_files: bool | None
) -> tuple[ContentType, ...]:
    """Apply the deprecated include_text_files override, emitting a DeprecationWarning."""
    if include_text_files is None:
        return (content,) if isinstance(content, ContentType) else tuple(content)
    warnings.warn(
        _INCLUDE_TEXT_FILES_DEPRECATION_MSG,
        DeprecationWarning,
        stacklevel=3,
    )
    return _ALL_CONTENT if include_text_files else _DEFAULT_CONTENT


def _cache_is_loadable_for_git(cache_path: Path) -> bool:
    """Reject hybrid-generation caches that this non-streaming path cannot prove active."""
    metadata_path = PersistencePath.from_path(cache_path).metadata
    try:
        metadata = orjson.loads(metadata_path.read_bytes())
    except OSError:
        return True
    except ValueError:
        return False
    return metadata.get("active_generation") is None


class _StableIdSemanticBackend:
    """Map persisted dense row positions back to stable chunk IDs."""

    def __init__(self, backend: SelectableBasicBackend, chunk_ids: Sequence[int]) -> None:
        """Wrap a row-position backend with stable chunk-id outputs."""
        self._backend = backend
        self._chunk_ids = np.array(chunk_ids, dtype=np.int_)
        self._rows_by_id = {int(chunk_id): row for row, chunk_id in enumerate(chunk_ids)}

    @property
    def vectors(self) -> npt.NDArray[np.float32]:
        """Expose underlying vectors for cache reuse paths."""
        return self._backend.vectors

    def query(
        self,
        vectors: npt.NDArray,
        k: int,
        selector: npt.NDArray[np.int_] | None = None,
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]]:
        """Run dense query with stable chunk-id selectors and results."""
        row_selector = None
        if selector is not None:
            row_selector = np.array(
                [self._rows_by_id[int(chunk_id)] for chunk_id in selector if int(chunk_id) in self._rows_by_id],
                dtype=np.int_,
            )
            if len(row_selector) == 0:
                return []
        results = self._backend.query(vectors, k, selector=row_selector)
        return [(self._chunk_ids[indices], distances) for indices, distances in results]

    def save(self, path: Path) -> None:
        """Persist the underlying dense backend."""
        self._backend.save(path)


class LazyChunkList(Sequence[Chunk]):
    """Sequence that loads persisted chunks by stable ID only when requested."""

    def __init__(
        self,
        chunk_ids: Sequence[int],
        store_path: Path,
        file_paths: Sequence[str] = (),
        languages: Sequence[str | None] = (),
    ) -> None:
        """Create a lazy chunk sequence backed by LMDB payloads."""
        self._chunk_ids = list(chunk_ids)
        self._store_path = store_path
        self._cache: dict[int, Chunk] = {}
        self._file_mapping = self._build_mapping(file_paths)
        self._language_mapping = self._build_language_mapping(languages)

    def __len__(self) -> int:
        """Return total persisted chunk count."""
        return len(self._chunk_ids)

    def __iter__(self) -> Iterator[Chunk]:
        """Yield chunks in persisted order."""
        for chunk_id in self._chunk_ids:
            yield self.chunk_by_id(chunk_id)

    def __getitem__(self, index: int) -> Chunk:
        """Return one chunk by persisted position."""
        return self.chunk_by_id(self._chunk_ids[index])

    def __eq__(self, other: object) -> bool:
        """Compare with another chunk sequence by loaded chunk values."""
        if not isinstance(other, Sequence):
            return False
        return list(self) == list(other)

    def chunk_by_id(self, chunk_id: int) -> Chunk:
        """Return one chunk by stable ID, loading it from LMDB on first access."""
        chunk = self._cache.get(chunk_id)
        if chunk is not None:
            return chunk
        store = LmdbChunkStore.open(self._store_path, readonly=True)
        try:
            loaded = store.get_chunk(chunk_id)
        finally:
            store.close()
        if loaded is None:
            raise FileNotFoundError(f"Index chunk store is missing chunk payload for id {chunk_id}")
        self._cache[chunk_id] = loaded
        return loaded

    def chunks_by_id(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Return chunks for stable IDs, preserving requested order."""
        return [self.chunk_by_id(chunk_id) for chunk_id in chunk_ids]

    def chunk_ids_for_paths(self, file_paths: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks under selected file paths."""
        return [chunk_id for path in file_paths for chunk_id in self._file_mapping.get(path, [])]

    def chunk_ids_for_languages(self, languages: frozenset[str]) -> list[int]:
        """Return stable IDs for persisted chunks in selected languages."""
        return [chunk_id for language in languages for chunk_id in self._language_mapping.get(language, [])]

    def _build_mapping(self, file_paths: Sequence[str]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, file_path in zip(self._chunk_ids, file_paths):
            mapping[file_path].append(chunk_id)
        return dict(mapping)

    def file_mapping(self) -> dict[str, list[int]]:
        """Return persisted file-to-chunk-ID mapping."""
        return self._file_mapping

    def language_mapping(self) -> dict[str, list[int]]:
        """Return persisted language-to-chunk-ID mapping."""
        return self._language_mapping

    def _build_language_mapping(self, languages: Sequence[str | None]) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = defaultdict(list)
        for chunk_id, language in zip(self._chunk_ids, languages):
            if language:
                mapping[language].append(chunk_id)
        return dict(mapping)


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: StaticModel,
        bm25_index: SparseIndex | BM25,
        semantic_index: SelectableBasicBackend,
        chunks: Sequence[Chunk],
        model_path: str,
        root: Path | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        loaded_from_disk: bool = False,
    ) -> None:
        """Initialize a SembleIndex. Should be created with from_path or from_git.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        :param model_path: Path to the model file.
        :param root: Root directory used to read file sizes for token-savings stats.
        :param content: Content type used when indexing; controls the search pipeline.
        :param loaded_from_disk: Whether the index was loaded from disk (cache hit); controls CLI messaging.
        """
        self.model = model
        self.chunks: Sequence[Chunk] = chunks
        self._bm25_index: SparseIndex | BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._model_path: str = model_path
        self._root: Path | None = root
        self._content: tuple[ContentType, ...] = (content,) if isinstance(content, ContentType) else tuple(content)
        self._file_sizes: dict[str, int] = {}
        self._file_mapping, self._language_mapping = self._populate_mapping()
        self.loaded_from_disk: bool = loaded_from_disk

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order."""
        if hasattr(self.chunks, "file_mapping") and hasattr(self.chunks, "language_mapping"):
            return self.chunks.file_mapping(), self.chunks.language_mapping()
        language_to_id = defaultdict(list)
        file_to_id = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            language = chunk.language
            if language:
                language_to_id[language].append(i)
            file_to_id[chunk.file_path].append(i)

        return dict(file_to_id), dict(language_to_id)

    def _compute_file_sizes(self, root: Path) -> dict[str, int]:
        """Return a mapping of repo-relative file path to total character count."""
        sizes: dict[str, int] = {}
        for chunk in self.chunks:
            if chunk.file_path in sizes:
                continue
            try:
                sizes[chunk.file_path] = len(read_file_text(root / chunk.file_path))
            except OSError:
                pass
        return sizes

    @property
    def stats(self) -> IndexStats:
        """Stats of an index."""
        language_counts: dict[str, int] = defaultdict(int)
        for chunk in self.chunks:
            if chunk.language:
                language_counts[chunk.language] += 1

        return IndexStats(
            indexed_files=len(self._file_mapping),
            total_chunks=len(self.chunks),
            languages=dict(language_counts),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
        model_path: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param content: Content types to index, e.g. ContentType.CODE or [ContentType.CODE, ContentType.DOCS].
        :param include_text_files: Deprecated. Pass a content sequence directly instead.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        normalized = _apply_include_text_files(content, include_text_files)
        cache_path = get_validated_cache(str(path), model_path, normalized)
        if cache_path:
            return cls.load_from_disk(cache_path)
        model, model_path = load_model(model_path)

        path = path.resolve()
        build = create_index_build_from_path(
            path,
            model=model,
            content=normalized,
            display_root=path,
        )

        index = SembleIndex(
            model,
            build.sparse_index,
            build.semantic_index,
            build.chunks,
            model_path,
            root=path,
            content=normalized,
        )
        index._file_sizes = build.file_sizes
        return index

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model_path: str | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
    ) -> SembleIndex:
        """Clone a git repository and index it.

        The repository is cloned into a temporary directory that is removed once
        indexing finishes. Chunk content is preserved in-memory, but
        chunk.file_path will not point to a readable file after this call
        returns — it is a repo-relative label, not a filesystem path.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :param content: Content types to index, e.g. (ContentType.CODE,) or (ContentType.CODE, ContentType.DOCS).
        :param include_text_files: Deprecated. Pass content=(ContentType.CODE, ContentType.DOCS, ...) instead.
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH, the clone fails, or times out.
        """
        normalized = _apply_include_text_files(content, include_text_files)
        cache_key = f"{url}@{ref}" if ref else url
        cache_path = get_validated_cache(cache_key, model_path, normalized)
        if cache_path and _cache_is_loadable_for_git(cache_path):
            return cls.load_from_disk(cache_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # `--` prevents `url` from being interpreted as a git option (e.g. `--upload-pack=...`).
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), "--", url, tmp_dir]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=_GIT_CLONE_TIMEOUT
                )
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"git clone timed out for {url!r} (limit: {_GIT_CLONE_TIMEOUT} s)") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")

            model, model_path = load_model(model_path)
            resolved_path = Path(tmp_dir).resolve()
            build = create_index_build_from_path(
                resolved_path,
                model=model,
                content=normalized,
                display_root=resolved_path,
            )

            index = SembleIndex(
                model,
                build.sparse_index,
                build.semantic_index,
                build.chunks,
                model_path,
                root=resolved_path,
                content=normalized,
            )
            index._file_sizes = build.file_sizes
            return index

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        filter_spec = FilterSpec(languages=frozenset([target.language])) if target.language else None
        results = _search_semantic(
            target.content, self.model, self._semantic_index, self.chunks, top_k + 1, filter_spec
        )
        results = [r for r in results if r.chunk != target][:top_k]
        save_search_stats(results, CallType.FIND_RELATED, self._file_sizes)
        return results

    def _get_selector_vector(
        self, filter_languages: list[str] | None = None, filter_paths: list[str] | None = None
    ) -> npt.NDArray[np.int_] | None:
        """Create a vector of chunk indices to restrict retrieval to."""
        selector = []
        for language in filter_languages or []:
            selector.extend(self._language_mapping.get(language, []))
        for filename in filter_paths or []:
            selector.extend(self._file_mapping.get(filename, []))

        return np.unique(selector) if selector else None

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        filter_languages: list[str] | None = None,
        filter_paths: list[str] | None = None,
        rerank: bool | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. None auto-detects from query type.
        :param filter_languages: Optional list of language codes; if set, only chunks in
            these languages are returned.
        :param filter_paths: Optional list of repo-relative file paths; if set, only
            chunks from these files are returned.
        :param rerank: Apply code-tuned reranking (file boost, identifier boost, path penalties).
            Defaults to True when ContentType.CODE was indexed.
        :return: Ranked list of SearchResult objects, best match first.
        """
        if not self.chunks or not query.strip():
            return []

        resolved_rerank = (ContentType.CODE in self._content) if rerank is None else rerank

        filter_spec = None
        if filter_languages or filter_paths:
            filter_spec = FilterSpec(
                file_paths=frozenset(filter_paths) if filter_paths else None,
                languages=frozenset(filter_languages) if filter_languages else None,
            )
        results = search(
            query,
            self.model,
            self._semantic_index,
            self._bm25_index,
            self.chunks,
            top_k,
            alpha=alpha,
            filter_spec=filter_spec,
            rerank=resolved_rerank,
        )
        save_search_stats(results, CallType.SEARCH, self._file_sizes)
        return results

    @classmethod
    def load_from_disk(cls: type[SembleIndex], path: Path | str) -> SembleIndex:
        """Load the index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        persistence_paths = PersistencePath.from_path(path)
        non_existent = persistence_paths.non_existing()
        if non_existent:
            missing = ", ".join(str(p) for p in non_existent)
            raise FileNotFoundError(f"Index not found at {path}. Missing: {missing}")

        semantic_index = SelectableBasicBackend.load(persistence_paths.semantic_index)
        with open(persistence_paths.metadata, "rb") as f:
            metadata = orjson.loads(f.read())

        chunk_ids = metadata.get("chunk_ids")
        if chunk_ids is not None:
            ids = [int(chunk_id) for chunk_id in chunk_ids]
            semantic_index = _StableIdSemanticBackend(semantic_index, ids)
            if metadata.get("sparse_backend") == "tantivy":
                chunks = LazyChunkList(
                    ids,
                    persistence_paths.chunk_store,
                    metadata.get("chunk_file_paths", []),
                    metadata.get("chunk_languages", []),
                )
                bm_25_index: SparseIndex | BM25 = TantivySparseIndex.load_from_store(
                    persistence_paths.bm25_index,
                    persistence_paths.chunk_store,
                )
            else:
                bm_25_index = BM25.load(persistence_paths.bm25_index)
                store = LmdbChunkStore.open(persistence_paths.chunk_store, readonly=True)
                try:
                    chunks = store.get_chunks(ids)
                finally:
                    store.close()
                if len(chunks) != len(ids):
                    raise FileNotFoundError("Index chunk store is missing chunk payloads")
        else:
            bm_25_index = BM25.load(persistence_paths.bm25_index)
            with open(persistence_paths.chunks, "r") as f:
                chunk_data = orjson.loads(f.read())
            chunks = [Chunk.from_dict(chunk_item) for chunk_item in chunk_data]
        root_path = metadata["root_path"]
        model_path = metadata["model_path"]
        content = tuple(ContentType(s) for s in metadata.get("content_type", ["code"]))
        if root_path:
            root_path = Path(root_path)

        model, model_path = load_model(model_path)

        index = cls(
            model,
            bm_25_index,
            semantic_index,
            chunks,
            model_path,
            root=root_path,
            content=content,
            loaded_from_disk=True,
        )
        index._file_sizes = {str(key): int(value) for key, value in metadata.get("file_sizes", {}).items()}
        return index

    def save(self, path: Path | str) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = path.parent / f"{path.name}.tmp-{uuid.uuid4().hex}"
        backup_path = path.parent / f"{path.name}.old-{uuid.uuid4().hex}"

        try:
            self._write_staged_index(staging_path)
            self._replace_saved_index(path, staging_path, backup_path)
        except Exception:
            self._restore_failed_save(path, staging_path, backup_path)
            raise
        if backup_path.exists():
            shutil.rmtree(backup_path)

    def _write_staged_index(self, staging_path: Path) -> None:
        """Write a complete index into a staging directory."""
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(parents=True)
        persistence_paths = PersistencePath.from_path(staging_path)

        if self._root is not None and not self._file_sizes:
            self._file_sizes = self._compute_file_sizes(self._root)
        self._bm25_index.save(persistence_paths.bm25_index)
        self._semantic_index.save(persistence_paths.semantic_index)
        chunk_ids = [chunk.chunk_id if chunk.chunk_id is not None else index for index, chunk in enumerate(self.chunks)]
        chunk_store_id = self._write_chunk_store(persistence_paths, chunk_ids)
        metadata = self._metadata_for_save(chunk_ids, chunk_store_id)
        with open(persistence_paths.metadata, "wb") as f:
            data = orjson.dumps(metadata)
            f.write(data)

    def _write_chunk_store(self, persistence_paths: PersistencePath, chunk_ids: list[int]) -> str:
        """Write LMDB chunks and return the store identifier bound to metadata."""
        chunk_store_id = uuid.uuid4().hex
        store = LmdbChunkStore.open(persistence_paths.chunk_store)
        try:
            store.write_chunks_with_ids(self.chunks, chunk_ids)
            store.write_store_id(chunk_store_id)
        finally:
            store.close()
        return chunk_store_id

    def _metadata_for_save(self, chunk_ids: list[int], chunk_store_id: str) -> dict[str, object]:
        """Build metadata for a staged full-save index."""
        root_str = None if self._root is None else str(self._root)
        file_paths = sorted(self._file_mapping)
        metadata: dict[str, object] = {
            "root_path": root_str,
            "time": datetime.now().timestamp(),
            "model_path": self._model_path,
            "content_type": list(x.value for x in self._content),
            "file_paths": file_paths,
            "chunk_ids": chunk_ids,
            "chunk_file_paths": [chunk.file_path for chunk in self.chunks],
            "chunk_languages": [chunk.language for chunk in self.chunks],
            "chunk_store_id": chunk_store_id,
            "file_sizes": self._file_sizes,
            "sparse_backend": "tantivy" if isinstance(self._bm25_index, TantivySparseIndex) else "bm25s",
        }
        if self._root is not None:
            git_roots = build_git_cache_metadata(self._root, file_paths)
            if git_roots is not None:
                metadata["git_roots"] = git_roots
                metadata["git_roots_version"] = GIT_CACHE_ROOTS_VERSION
        return metadata

    def _replace_saved_index(self, path: Path, staging_path: Path, backup_path: Path) -> None:
        """Replace an existing index with a complete staged index."""
        if path.exists():
            path.rename(backup_path)
        staging_path.rename(path)

    def _restore_failed_save(self, path: Path, staging_path: Path, backup_path: Path) -> None:
        """Remove incomplete staged data and restore the previous index if needed."""
        if staging_path.exists():
            shutil.rmtree(staging_path)
        if backup_path.exists() and not path.exists():
            backup_path.rename(path)
