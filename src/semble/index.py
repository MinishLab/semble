from __future__ import annotations

import contextlib
from pathlib import Path

import bm25s
import numpy as np
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.cache import make_embedding_cache
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
        model_id: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize a SembleIndex.

        :param model: Embedding model to use. Defaults to "Pringled/potion-code-16M"
            (loaded lazily on first use).
        :param model_id: Stable identifier for the encoder (e.g. its HuggingFace hub ID).
            Required when cache_dir is set; used as the disk-cache namespace so
            embeddings from different models never mix. For the built-in default model
            pass model_id="Pringled/potion-code-16M". When using a custom model,
            also pass a matching model object — otherwise semantic/hybrid search will
            raise ValueError to prevent silent dimensionality mismatches.
        :param cache_dir: Directory for the disk embedding cache. When given, previously
            computed embeddings are reused across runs. Only embeddings are persisted;
            BM25 and the ANNS index are always rebuilt in-memory. ~ is expanded
            automatically. model_id is required when this is set.
        :raises ValueError: If cache_dir is given without model_id.
        """
        if cache_dir is not None and model_id is None:
            raise ValueError("model_id is required when cache_dir is provided")
        self.model: Encoder | None = model
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
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
        cache_dir: str | Path | None = None,
        model_id: str | None = None,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        Backend configuration (model, model_id, cache_dir) is forwarded to
        the constructor; source-selection arguments (extensions, ignore,
        include_docs) are forwarded to index.

        :param path: Root directory to index.
        :param model: Embedding model to use.
        :param extensions: File extensions to include.
        :param ignore: Directory names to skip.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :param cache_dir: Directory for the disk embedding cache.
        :param model_id: Stable identifier for the encoder used as the cache namespace.
        :return: An indexed SembleIndex.
        """
        instance = cls(model=model, model_id=model_id, cache_dir=cache_dir)
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
        :param mode: Search strategy — "hybrid" (default), "semantic", or "bm25".
        :param alpha: Blend weight for hybrid mode; 1.0 = pure semantic, 0.0 = pure BM25.
        :return: Ranked list of SearchResult objects, best match first.
        :raises ValueError: If mode is not a recognised search strategy.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks or bm25_index is None or semantic_index is None:
            return []

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k)

        # Semantic and hybrid both need an embedding model; load the default lazily.
        model = self._ensure_model()
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, model, semantic_index, top_k)
        if mode == SearchMode.HYBRID:
            return search_hybrid(query, model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha)
        raise ValueError(f"Unknown search mode: {mode!r}")

    def _ensure_model(self) -> Encoder:
        """Return the current model, loading the default if none was provided.

        :return: The active encoder.
        :raises ValueError: If the index was configured with a non-default model_id
            and no explicit model was supplied.  Lazy-loading the built-in default
            model would produce query vectors with a different dimensionality than the
            cached embeddings, causing silent shape mismatches in Vicinity.
        """
        if self.model is None:
            # Only safe to lazy-load the default when the embeddings in this index
            # were produced by (or are compatible with) the default model.  That is
            # true when no model_id was set (no disk cache) or when model_id matches
            # the default exactly.
            if self.model_id is not None and self.model_id != _DEFAULT_MODEL_NAME:
                raise ValueError(
                    f"This index was configured with model {self.model_id!r} but no model was "
                    f"supplied at construction time.  Pass the matching model explicitly to "
                    f"avoid embedding dimensionality mismatches."
                )
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

        cache = make_embedding_cache(self._embedding_cache, self.cache_dir, self.model_id)

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
