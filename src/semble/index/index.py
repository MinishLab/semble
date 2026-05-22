from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson
from bm25s import BM25
from model2vec.model import StaticModel

from semble.index.create import create_index_from_path
from semble.index.dense import SelectableBasicBackend, load_model
from semble.index.types import PersistencePath
from semble.search import _search_semantic, search
from semble.stats import save_search_stats
from semble.types import CallType, Chunk, ContentType, IndexStats, SearchResult

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


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: StaticModel,
        bm25_index: BM25,
        semantic_index: SelectableBasicBackend,
        chunks: list[Chunk],
        model_path: str,
        root: Path | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
    ) -> None:
        """Initialize a SembleIndex. Should be created with from_path or from_git.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        :param model_path: Path to the model file.
        :param root: Root directory used to read file sizes for token-savings stats.
        :param content: Content type used when indexing; controls the search pipeline.
        """
        self.model = model
        self.chunks: list[Chunk] = chunks
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: SelectableBasicBackend = semantic_index
        self._model_path: str = model_path
        self._root: Path | None = root
        self._content: tuple[ContentType, ...] = (content,) if isinstance(content, ContentType) else tuple(content)
        self._file_sizes: dict[str, int] = self._compute_file_sizes(root) if root else {}
        self._file_mapping, self._language_mapping = self._populate_mapping()

    def _populate_mapping(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Build (file → chunk indices, language → chunk indices) mappings, in that order."""
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
                sizes[chunk.file_path] = len((root / chunk.file_path).read_text(encoding="utf-8", errors="replace"))
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
        extensions: Sequence[str] | None = None,
        content: ContentType | Sequence[ContentType] = _DEFAULT_CONTENT,
        include_text_files: bool | None = None,
        model_path: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param content: Content types to index, e.g. ContentType.CODE or [ContentType.CODE, ContentType.DOCS].
        :param include_text_files: Deprecated. Pass a content sequence directly instead.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :return: An indexed SembleIndex. Chunk file paths are relative to ``path``.
        :raises FileNotFoundError: If `path` does not exist.
        :raises NotADirectoryError: If `path` exists but is not a directory.
        """
        model, model_path = load_model(model_path)
        normalized = _apply_include_text_files(content, include_text_files)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        path = path.resolve()
        bm25, vicinity, chunks = create_index_from_path(
            path,
            model=model,
            extensions=extensions,
            content=normalized,
            display_root=path,
        )

        return SembleIndex(model, bm25, vicinity, chunks, model_path, root=path, content=normalized)

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        extensions: Sequence[str] | None = None,
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
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param model_path: Path to the model to use. If None, the default model will be used.
        :param content: Content types to index, e.g. (ContentType.CODE,) or (ContentType.CODE, ContentType.DOCS).
        :param include_text_files: Deprecated. Pass content=(ContentType.CODE, ContentType.DOCS, ...) instead.
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH, the clone fails, or times out.
        """
        normalized = _apply_include_text_files(content, include_text_files)
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
            bm25, vicinity, chunks = create_index_from_path(
                resolved_path,
                model=model,
                extensions=extensions,
                content=normalized,
                display_root=resolved_path,
            )

            return SembleIndex(model, bm25, vicinity, chunks, model_path, root=resolved_path, content=normalized)

    def find_related(self, source: Chunk | SearchResult, *, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the given chunk or search result.

        :param source: A SearchResult or Chunk to use as the seed.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = source.chunk if isinstance(source, SearchResult) else source
        selector = self._get_selector_vector(filter_languages=[target.language]) if target.language else None
        results = _search_semantic(target.content, self.model, self._semantic_index, self.chunks, top_k + 1, selector)
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

        selector = self._get_selector_vector(filter_languages, filter_paths)
        results = search(
            query,
            self.model,
            self._semantic_index,
            self._bm25_index,
            self.chunks,
            top_k,
            alpha=alpha,
            selector=selector,
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

        bm_25_index = BM25.load(persistence_paths.bm25_index)
        semantic_index = SelectableBasicBackend.load(persistence_paths.semantic_index)
        with open(persistence_paths.metadata, "r") as f:
            metadata = orjson.loads(f.read())
        with open(persistence_paths.chunks, "r") as f:
            chunk_data = orjson.loads(f.read())

        chunks = []
        for chunk_item in chunk_data:
            chunks.append(Chunk.from_dict(chunk_item))
        root_path = metadata["root_path"]
        model_path = metadata["model_path"]
        if root_path:
            root_path = Path(root_path)

        model, model_path = load_model(model_path)

        return cls(model, bm_25_index, semantic_index, chunks, model_path, root=root_path)

    def save(self, path: Path | str) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        persistence_paths = PersistencePath.from_path(path)

        self._bm25_index.save(persistence_paths.bm25_index)
        self._semantic_index.save(persistence_paths.semantic_index)
        chunks_as_dict = [chunk.to_dict() for chunk in self.chunks]
        with open(persistence_paths.chunks, "wb") as f:
            data = orjson.dumps(chunks_as_dict)
            f.write(data)
        root_str = None if self._root is None else str(self._root)
        metadata = {"root_path": root_str, "time": datetime.now().timestamp(), "model_path": self._model_path}
        with open(persistence_paths.metadata, "wb") as f:
            data = orjson.dumps(metadata)
            f.write(data)
