from __future__ import annotations

import contextlib
from pathlib import Path

import bm25s
import numpy as np
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.chunker import chunk_source
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.sources import language_for_path, resolve_extensions, walk_files
from semble.types import Chunk, EmbeddingMatrix, Encoder, IndexStats, SearchMode, SearchResult
from semble.utils import tokenize

_DEFAULT_MODEL_NAME = "Pringled/potion-code-16M"


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(self, model: Encoder | None = None) -> None:
        """Initialize a SembleIndex."""
        self.model = model
        self._chunks: list[Chunk] = []
        self._embedding_cache: dict[str, EmbeddingMatrix] = {}
        self._bm25_index: bm25s.BM25 | None = None
        self._semantic_index: Vicinity | None = None
        self._stats = IndexStats()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> SembleIndex:
        """Create a SembleIndex index from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to ``Pringled/potion-code-16M``.
        :param extensions: File extensions to include. Defaults to all code extensions.
        :param ignore: Directory names to skip. Defaults to the standard ignored directories.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :return: An indexed SembleIndex.
        """
        instance = cls(model=model)
        instance.index(path, extensions=extensions, ignore=ignore, include_docs=include_docs)
        return instance

    def index(
        self,
        path: str | Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> IndexStats:
        """Index a directory."""
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

        self._chunks = all_chunks

        if all_chunks:
            model = self._ensure_model()
            embeddings = self._embed_chunks(model, all_chunks)
            self._bm25_index = self._build_bm25_index(all_chunks)
            self._semantic_index = self._build_semantic_index(embeddings, all_chunks)
        else:
            self._bm25_index = None
            self._semantic_index = None

        self._stats = IndexStats(
            indexed_files=indexed_files,
            total_chunks=len(all_chunks),
            languages=language_counts,
        )
        return self._stats

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
        model, bm25_index, semantic_index = self.model, self._bm25_index, self._semantic_index
        if not self._chunks or model is None or bm25_index is None or semantic_index is None:
            return []

        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, model, semantic_index, top_k)
        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self._chunks, top_k)
        if mode == SearchMode.HYBRID:
            return search_hybrid(query, model, semantic_index, bm25_index, self._chunks, top_k, alpha=alpha)
        raise ValueError(f"Unknown search mode: {mode!r}")

    @property
    def chunks(self) -> list[Chunk]:
        """Return the indexed chunks from the last call to index."""
        return self._chunks

    @property
    def stats(self) -> IndexStats:
        """Return indexing statistics from the last call to index."""
        return self._stats

    def _ensure_model(self) -> Encoder:
        """Return the current model, loading the default if none was provided."""
        if self.model is None:
            model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
            self.model = model
            return model
        return self.model

    def _embed_chunks(self, model: Encoder, chunks: list[Chunk]) -> EmbeddingMatrix:
        """Embed chunks, reusing cached embeddings when available."""
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)
        uncached = [
            (i, chunk.content) for i, chunk in enumerate(chunks) if chunk.content_hash not in self._embedding_cache
        ]
        if uncached:
            indices, texts = zip(*uncached, strict=True)
            for i, embedding in zip(indices, model.encode(list(texts)), strict=True):
                self._embedding_cache[chunks[i].content_hash] = embedding
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
