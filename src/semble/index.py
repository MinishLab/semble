from __future__ import annotations

import contextlib
from pathlib import Path

import bm25s
import numpy as np
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.cache import EmbeddingCache
from semble.chunker import chunk_source
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.sources import language_for_path, resolve_extensions, walk_files
from semble.types import Chunk, EmbeddingMatrix, Encoder, IndexStats, SearchMode, SearchResult
from semble.utils import tokenize

_DEFAULT_MODEL_NAME = "Pringled/potion-code-16M"


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: Encoder | None = None,
        *,
        enable_caching: bool = True,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize a SembleIndex instance."""
        self.model: Encoder | None = model
        self.cache_dir, self.cache_namespace = self._resolve_cache_config(
            model, enable_caching=enable_caching, cache_dir=cache_dir, model_name=model_name
        )
        self.chunks: list[Chunk] = []
        self.stats = IndexStats()
        self._embedding_cache: dict[str, EmbeddingMatrix] = {}
        self._bm25_index: bm25s.BM25 | None = None
        self._semantic_index: Vicinity | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
        enable_caching: bool = True,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :param enable_caching: Whether to persist embeddings to disk between runs.
        :param cache_dir: Override the cache directory. Defaults to ~/.cache/semble.
        :param model_name: Stable identifier for a custom encoder, used as the disk cache namespace.
        :return: An indexed SembleIndex.
        """
        instance = cls(model=model, enable_caching=enable_caching, cache_dir=cache_dir, model_name=model_name)
        instance.index(path, extensions=extensions, ignore=ignore, include_docs=include_docs)
        return instance

    def index(
        self,
        path: str | Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> IndexStats:
        """Index a directory using the backend configured at construction time.

        :param path: Root directory to index.
        :param extensions: File extensions to include.
        :param ignore: Directory names to skip.
        :param include_docs: If True, also index documentation files.
        :return: Statistics about the indexed files and chunks.
        """
        path = Path(path).resolve()
        extensions = resolve_extensions(extensions, include_docs=include_docs)

        all_chunks: list[Chunk] = []
        language_counts: dict[str, int] = {}
        indexed_files = 0

        for file_path in walk_files(path, extensions, ignore):
            language = language_for_path(file_path)
            with contextlib.suppress(OSError):
                source = file_path.read_text(encoding="utf-8", errors="replace")
                indexed_files += 1
                file_chunks = chunk_source(source, str(file_path), language)
                all_chunks.extend(file_chunks)
                for chunk in file_chunks:
                    if chunk.language:
                        language_counts[chunk.language] = language_counts.get(chunk.language, 0) + 1

        self.chunks = all_chunks

        if all_chunks:
            embeddings = self._embed_chunks(all_chunks)
            self._bm25_index = self._build_bm25_index(all_chunks)
            self._semantic_index = self._build_semantic_index(embeddings, all_chunks)
        else:
            self._bm25_index = None
            self._semantic_index = None

        self.stats = IndexStats(
            indexed_files=indexed_files,
            total_chunks=len(all_chunks),
            languages=language_counts,
        )
        return self.stats

    def find_related(self, file_path: str, line: int, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the chunk at the given file location.

        :param file_path: Absolute path to the file.
        :param line: Line number (1-indexed) used to identify the source chunk.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = next(
            (c for c in self.chunks if c.file_path == file_path and c.start_line <= line <= c.end_line),
            None,
        )
        if target is None or self._semantic_index is None:
            return []
        model = self._ensure_model()
        results = search_semantic(target.content, model, self._semantic_index, top_k + 1)
        return [r for r in results if r.chunk != target][:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | str = SearchMode.HYBRID,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param mode: Search strategy — ``"hybrid"`` (default), ``"semantic"``, or ``"bm25"``.
        :param alpha: Blend weight for hybrid mode; 1.0 = pure semantic, 0.0 = pure BM25.
        :return: Ranked list of :class:`SearchResult` objects, best match first.
        :raises ValueError: If ``mode`` is not a recognised search strategy.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks or bm25_index is None or semantic_index is None:
            return []

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k)

        model = self._ensure_model()
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, model, semantic_index, top_k)
        if mode == SearchMode.HYBRID:
            return search_hybrid(query, model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha)
        raise ValueError(f"Unknown search mode: {mode!r}")

    @staticmethod
    def _resolve_cache_config(
        model: Encoder | None,
        *,
        enable_caching: bool,
        cache_dir: str | Path | None,
        model_name: str | None,
    ) -> tuple[Path | None, str | None]:
        """Determine cache directory and namespace based on constructor args."""
        if not enable_caching:
            return None, None
        root = Path(cache_dir).expanduser() if cache_dir is not None else Path.home() / ".cache" / "semble"
        if model is None:
            return root, _DEFAULT_MODEL_NAME
        if model_name is not None:
            return root, model_name
        return None, None

    def _ensure_model(self) -> Encoder:
        """Return the current model, loading the default if none was provided.

        :return: The active encoder.
        """
        if self.model is None:
            self.model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
        return self.model

    def _embed_chunks(self, chunks: list[Chunk]) -> EmbeddingMatrix:
        """Embed chunks, consulting memory then disk before calling the model.

        Lookup order: in-memory cache → disk cache → encode. The model is loaded
        (or downloaded) only when there are genuine cache misses.

        :param chunks: Chunks to embed.
        :return: Matrix of embeddings, one row per chunk, in input order.
        """
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)

        cache = EmbeddingCache(self._embedding_cache, self.cache_dir, self.cache_namespace)

        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            if cache.get(chunk.content_hash) is None:
                miss_indices.append(i)
                miss_texts.append(chunk.content)

        if miss_indices:
            model = self._ensure_model()
            for i, embedding in zip(miss_indices, model.encode(miss_texts), strict=True):
                cache.put(chunks[i].content_hash, embedding)

        return np.array([self._embedding_cache[chunk.content_hash] for chunk in chunks], dtype=np.float32)

    def _build_bm25_index(self, chunks: list[Chunk]) -> bm25s.BM25:
        """Build a BM25 index over tokenized, path-enriched chunk text."""
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(self._enrich_for_bm25(chunk)) for chunk in chunks],
            show_progress=False,
        )
        return bm25_index

    def _build_semantic_index(self, embeddings: EmbeddingMatrix, chunks: list[Chunk]) -> Vicinity:
        """Build an ANNS index over chunk embeddings for semantic search."""
        return Vicinity.from_vectors_and_items(embeddings, chunks, metric=Metric.COSINE)

    def _enrich_for_bm25(self, chunk: Chunk) -> str:
        """Append file stem to BM25 content to boost path-based queries."""
        stem = Path(chunk.file_path).stem
        # Repeat the stem twice to up-weight file-path matches in BM25.
        return f"{chunk.content} {stem} {stem}"
