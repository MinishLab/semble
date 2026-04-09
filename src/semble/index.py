from __future__ import annotations

import contextlib
import os
import tempfile
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


def _model_namespace(model_id: str) -> str:
    """Return a safe directory name for *model_id*.

    HuggingFace IDs contain ``/`` (e.g. ``Pringled/potion-code-16M``), so we
    replace every ``/`` with ``--`` to keep the namespace human-readable while
    staying safe as a single path segment.
    """
    return model_id.replace("/", "--")


def _embedding_cache_path(cache_dir: Path, model_ns: str, content_hash: str) -> Path:
    """Return the per-embedding file path inside *cache_dir*."""
    return cache_dir / model_ns / content_hash[:2] / f"{content_hash}.npy"


def _load_embedding(path: Path) -> EmbeddingMatrix | None:
    """Load a single embedding from *path*, returning ``None`` on any error."""
    try:
        return np.load(path, allow_pickle=False)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _save_embedding_atomic(path: Path, embedding: EmbeddingMatrix) -> None:
    """Write *embedding* to *path* atomically via a temp-file rename.

    Atomic rename ensures concurrent processes never read a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, embedding, allow_pickle=False)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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
        # Recorded during index() so _ensure_model() can validate lazy-load safety.
        self._model_id: str | None = None

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
        """Create a SembleIndex index from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to ``Pringled/potion-code-16M``.
        :param extensions: File extensions to include. Defaults to all code extensions.
        :param ignore: Directory names to skip. Defaults to the standard ignored directories.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :param cache_dir: Directory for the disk embedding cache. When given, previously
            computed embeddings are reused across runs. Only embeddings are cached; BM25
            and the ANNS index are always rebuilt in-memory. *model_id* is required when
            this is set. To cache with the built-in default model pass
            ``model_id="Pringled/potion-code-16M"``. When using a custom model, you must
            also pass a matching *model* object so that semantic/hybrid search can encode
            queries with the same dimensionality as the cached embeddings.
        :param model_id: Stable identifier for the encoder (e.g. its HuggingFace hub ID).
            Used as the cache namespace so embeddings from different models never mix.
        :return: An indexed SembleIndex.
        """
        instance = cls(model=model)
        instance.index(
            path,
            extensions=extensions,
            ignore=ignore,
            include_docs=include_docs,
            cache_dir=cache_dir,
            model_id=model_id,
        )
        return instance

    def index(
        self,
        path: str | Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
        cache_dir: str | Path | None = None,
        model_id: str | None = None,
    ) -> IndexStats:
        """Index a directory.

        :param path: Root directory to index.
        :param extensions: File extensions to include.
        :param ignore: Directory names to skip.
        :param include_docs: If True, also index documentation files.
        :param cache_dir: Directory for the disk embedding cache. *model_id* is required
            when this is set. Only embeddings are persisted; BM25 and the ANNS index are
            always rebuilt in-memory from cached embeddings. To cache with the built-in
            default model pass ``model_id="Pringled/potion-code-16M"``. For a custom
            model, also pass the matching *model* object so that semantic/hybrid search
            can encode queries correctly.
        :param model_id: Stable identifier for the encoder used as the cache namespace.
        :return: Statistics about the indexed files and chunks.
        :raises ValueError: If *cache_dir* is given without *model_id*.
        """
        if cache_dir is not None and model_id is None:
            raise ValueError("model_id is required when cache_dir is provided")

        path = Path(path).resolve()
        extensions = resolve_extensions(extensions, include_docs=include_docs)

        # Record which model namespace these embeddings belong to so that
        # _ensure_model() can validate lazy-load safety at search time.
        self._model_id = model_id

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
            _cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
            _model_ns = _model_namespace(model_id) if model_id is not None else None
            embeddings = self._embed_chunks(all_chunks, cache_dir=_cache_dir, model_ns=_model_ns)
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
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self._chunks or bm25_index is None:
            return []

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self._chunks, top_k)

        # Semantic and hybrid both need an embedding model; load the default lazily.
        if semantic_index is None:
            return []
        model = self._ensure_model()
        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, model, semantic_index, top_k)
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
        """Return the current model, loading the default if none was provided.

        :return: The active encoder.
        :raises ValueError: If the index was built from a non-default model's cached
            embeddings and no explicit model was supplied.  Lazy-loading the built-in
            default model would produce query vectors with a different dimensionality
            than the cached embeddings, causing silent shape mismatches in Vicinity.
        """
        if self.model is None:
            # Only safe to lazy-load the default when the embeddings in this index
            # were produced by (or are compatible with) the default model.  That is
            # true when no model_id was recorded (no disk cache used) or when the
            # recorded model_id matches the default model exactly.
            if self._model_id is not None and self._model_id != _DEFAULT_MODEL_NAME:
                raise ValueError(
                    f"This index was built with model {self._model_id!r} but no model was "
                    f"supplied at construction time.  Pass the matching model explicitly to "
                    f"avoid embedding dimensionality mismatches."
                )
            model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
            self.model = model
            return model
        return self.model

    def _embed_chunks(
        self,
        chunks: list[Chunk],
        cache_dir: Path | None = None,
        model_ns: str | None = None,
    ) -> EmbeddingMatrix:
        """Embed *chunks*, consulting memory then disk before calling the model.

        Lookup order: in-memory cache → disk cache → encode. The model is loaded
        (or downloaded) only when there are genuine cache misses.
        """
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)

        # Pass 1: identify misses after checking memory, then disk.
        disk_hits: dict[str, EmbeddingMatrix] = {}
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            h = chunk.content_hash
            if h in self._embedding_cache:
                continue
            if cache_dir is not None and model_ns is not None:
                emb = _load_embedding(_embedding_cache_path(cache_dir, model_ns, h))
                if emb is not None:
                    disk_hits[h] = emb
                    continue
            miss_indices.append(i)
            miss_texts.append(chunk.content)

        # Promote disk hits to the in-memory cache.
        self._embedding_cache.update(disk_hits)

        # Pass 2: encode only real misses (lazy-loads model when needed).
        if miss_indices:
            model = self._ensure_model()
            for i, embedding in zip(miss_indices, model.encode(miss_texts), strict=True):
                h = chunks[i].content_hash
                self._embedding_cache[h] = embedding
                if cache_dir is not None and model_ns is not None:
                    _save_embedding_atomic(_embedding_cache_path(cache_dir, model_ns, h), embedding)

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
