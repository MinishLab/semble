from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from model2vec import StaticModel

from semble.chunker import EXTENSION_MAP, chunk_source
from semble.search import (
    BM25Index,
    SemanticIndex,
    search_bm25,
    search_hybrid,
    search_semantic,
    search_symbol,
)
from semble.types import Chunk, IndexStats, SearchResult

_DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".yaml", ".yml", ".toml", ".json"})
ALL_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_MAP)
CODE_EXTENSIONS: frozenset[str] = ALL_EXTENSIONS - _DOC_EXTENSIONS

DEFAULT_IGNORE: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        ".tox",
        "dist",
        "build",
        ".eggs",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".semble",
    }
)


class SembleIndex:
    """Fast local code index with hybrid search.

    Usage::

        index = SembleIndex()
        stats = index.index_directory("./my-project")
        results = index.search("how does auth work?", top_k=5)
    """

    def __init__(self, model_name: str = "Pringled/potion-code-16M") -> None:
        self._model_name = model_name
        self._model: StaticModel | None = None
        self._chunks: list[Chunk] = []
        self._embedding_cache: dict[str, npt.NDArray[np.float32]] = {}
        self._bm25_index: BM25Index | None = None
        self._semantic_index: SemanticIndex | None = None
        self._file_lines: dict[str, list[str]] = {}
        self._stats = IndexStats()

    @property
    def model(self) -> StaticModel:
        """Lazily load the embedding model."""
        if self._model is None:
            self._model = StaticModel.from_pretrained(self._model_name)
        return self._model

    def index_directory(
        self,
        path: str | Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> IndexStats:
        """Index all code files under the given directory.

        :param path: Root directory to index.
        :param extensions: File extensions to include. Defaults to code-only.
        :param ignore: Directory/file names to skip. Defaults to common VCS/build dirs.
        :param include_docs: If True, also index docs (md, yaml, toml, json).
        :returns: Statistics about the indexed content.
        """
        path = Path(path).resolve()
        ignore = ignore or DEFAULT_IGNORE
        if extensions is None:
            extensions = ALL_EXTENSIONS if include_docs else CODE_EXTENSIONS

        t_start = time.perf_counter()

        all_chunks: list[Chunk] = []
        lang_counts: dict[str, int] = {}
        self._file_lines = {}

        for fp in self._walk_files(path, extensions, ignore):
            language = EXTENSION_MAP.get(fp.suffix.lower())
            with contextlib.suppress(OSError):
                source = fp.read_text(encoding="utf-8", errors="replace")
                self._file_lines[str(fp)] = source.splitlines()
                file_chunks = chunk_source(source, str(fp), language)
                all_chunks.extend(file_chunks)
                for c in file_chunks:
                    if c.language:
                        lang_counts[c.language] = lang_counts.get(c.language, 0) + 1

        t_emb_start = time.perf_counter()
        embeddings = self._embed_chunks(all_chunks)
        t_emb = time.perf_counter() - t_emb_start

        self._chunks = all_chunks

        if all_chunks:
            self._bm25_index = BM25Index([self._enrich_for_bm25(c) for c in all_chunks])
            self._semantic_index = SemanticIndex(all_chunks, embeddings)

        self._stats = IndexStats(
            total_files=len(self._file_lines),
            total_chunks=len(all_chunks),
            index_time_ms=(time.perf_counter() - t_start) * 1000,
            embedding_time_ms=t_emb * 1000,
            languages=lang_counts,
        )
        return self._stats

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Search the index.

        :param query: Natural language or code query.
        :param top_k: Number of results to return.
        :param mode: Search mode — one of "hybrid", "semantic", "bm25", "symbol".
        :param alpha: Semantic weight for hybrid mode (1-alpha goes to BM25). Default 0.5.
        :returns: List of search results, best first.
        :raises ValueError: If mode is not recognized.
        """
        if not self._chunks:
            return []

        if mode == "semantic":
            if self._semantic_index is None:
                return []
            return search_semantic(query, self.model, self._semantic_index, top_k)
        if mode == "bm25":
            if self._bm25_index is None:
                return []
            return search_bm25(query, self._bm25_index, self._chunks, top_k)
        if mode == "symbol":
            return search_symbol(query, self._file_lines, top_k)
        if mode == "hybrid":
            if self._semantic_index is None or self._bm25_index is None:
                return []
            return search_hybrid(
                query,
                self.model,
                self._semantic_index,
                self._bm25_index,
                self._chunks,
                top_k,
                alpha=alpha,
            )
        raise ValueError(
            f"Unknown search mode: {mode!r}. Choose from: hybrid, semantic, bm25, symbol"
        )

    def get_context(self, file_path: str, line: int, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically related to the chunk at the given location.

        :param file_path: Absolute path of the file being edited.
        :param line: Line number within the file.
        :param top_k: Number of related chunks to return.
        :returns: Related chunks from the same or other files, excluding the target chunk.
        """
        if self._semantic_index is None:
            return []
        target = next(
            (
                c
                for c in self._chunks
                if c.file_path == file_path and c.start_line <= line <= c.end_line
            ),
            None,
        )
        if target is None:
            return []
        results = search_semantic(target.content, self.model, self._semantic_index, top_k + 1)
        return [r for r in results if r.chunk.content_hash != target.content_hash][:top_k]

    @property
    def stats(self) -> IndexStats:
        """Return indexing statistics from the last call to index_directory."""
        return self._stats

    # -- Private helpers --

    def _walk_files(
        self,
        root: Path,
        extensions: frozenset[str],
        ignore: frozenset[str],
    ) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = sorted(d for d in dirnames if d not in ignore)
            for filename in sorted(filenames):
                p = Path(dirpath) / filename
                if p.suffix.lower() in extensions:
                    yield p

    def _embed_chunks(self, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)
        uncached = [
            (i, c.content)
            for i, c in enumerate(chunks)
            if c.content_hash not in self._embedding_cache
        ]
        if uncached:
            indices, texts = zip(*uncached, strict=True)
            new_embs = self.model.encode(list(texts))
            for idx, emb in zip(indices, new_embs, strict=True):
                self._embedding_cache[chunks[idx].content_hash] = emb
        return np.array([self._embedding_cache[c.content_hash] for c in chunks], dtype=np.float32)

    def _enrich_for_bm25(self, chunk: Chunk) -> str:
        """Append file stem to BM25 content to boost path-based queries."""
        stem = Path(chunk.file_path).stem
        return f"{chunk.content} {stem} {stem}"
