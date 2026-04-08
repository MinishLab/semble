import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import bm25s
import numpy as np
import numpy.typing as npt
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.chunker import EXTENSION_MAP, chunk_source
from semble.search import _tokenize, search_bm25, search_hybrid, search_semantic, search_symbol
from semble.types import Chunk, Encoder, FileLines, IndexStats, SearchMode, SearchResult

DEFAULT_MODEL_NAME = "Pringled/potion-code-16M"

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
        stats = index.index("./my-project")
        results = index.search("how does auth work?", top_k=5)
    """

    def __init__(self, model: Encoder | None = None) -> None:
        """Create an empty index with an optional encoder."""
        self.model = model
        self._chunks: list[Chunk] = []
        self._embedding_cache: dict[str, npt.NDArray[np.float32]] = {}
        self._bm25_index: bm25s.BM25 | None = None
        self._semantic_index: Vicinity | None = None
        self._file_lines: FileLines = {}
        self._stats = IndexStats()

    def index(
        self,
        path: str | Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> IndexStats:
        """Index all code files under the given directory."""
        path = Path(path).resolve()
        ignore = ignore or DEFAULT_IGNORE
        if extensions is None:
            extensions = ALL_EXTENSIONS if include_docs else CODE_EXTENSIONS

        all_chunks: list[Chunk] = []
        language_counts: dict[str, int] = {}
        self._file_lines = {}

        for file_path in self._walk_files(path, extensions, ignore):
            language = EXTENSION_MAP.get(file_path.suffix.lower())
            with contextlib.suppress(OSError):
                source = file_path.read_text(encoding="utf-8", errors="replace")
                self._file_lines[str(file_path)] = source.splitlines()
                file_chunks = chunk_source(source, str(file_path), language)
                all_chunks.extend(file_chunks)
                for chunk in file_chunks:
                    if chunk.language:
                        language_counts[chunk.language] = language_counts.get(chunk.language, 0) + 1

        embedding_start_time = time.perf_counter()
        embeddings = self._embed_chunks(all_chunks)
        embedding_time_seconds = time.perf_counter() - embedding_start_time

        self._chunks = all_chunks

        if all_chunks:
            self._bm25_index = bm25s.BM25()
            self._bm25_index.index(
                [_tokenize(self._enrich_for_bm25(chunk)) for chunk in all_chunks],
                show_progress=False,
            )
            self._semantic_index = Vicinity.from_vectors_and_items(
                embeddings,
                all_chunks,
                metric=Metric.COSINE,
            )

        self._stats = IndexStats(
            total_files=len(self._file_lines),
            total_chunks=len(all_chunks),
            embedding_time_ms=embedding_time_seconds * 1000,
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
        """Search the index."""
        if not self._chunks:
            return []

        try:
            mode = SearchMode(mode)
        except ValueError as exc:
            modes = ", ".join(str(value) for value in SearchMode)
            raise ValueError(f"Unknown search mode: {mode!r}. Choose from: {modes}") from exc

        if mode is SearchMode.SEMANTIC:
            if self._semantic_index is None or self.model is None:
                return []
            return search_semantic(query, self.model, self._semantic_index, top_k)
        if mode is SearchMode.BM25:
            if self._bm25_index is None:
                return []
            return search_bm25(query, self._bm25_index, self._chunks, top_k)
        if mode is SearchMode.SYMBOL:
            return search_symbol(query, self._file_lines, top_k)
        if self._semantic_index is None or self._bm25_index is None or self.model is None:
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

    @property
    def stats(self) -> IndexStats:
        """Return indexing statistics from the last call to index."""
        return self._stats

    def _walk_files(
        self,
        root: Path,
        extensions: frozenset[str],
        ignore: frozenset[str],
    ) -> Iterator[Path]:
        """Yield matching files while pruning ignored directories."""
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = sorted(d for d in dirnames if d not in ignore)
            for filename in sorted(filenames):
                file_path = Path(dirpath) / filename
                if file_path.suffix.lower() in extensions:
                    yield file_path

    def _embed_chunks(self, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
        """Embed chunks, reusing cached embeddings when available."""
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)
        model = self.model
        if model is None:
            model = cast(Encoder, StaticModel.from_pretrained(DEFAULT_MODEL_NAME))
            self.model = model
        uncached = [
            (index, chunk.content)
            for index, chunk in enumerate(chunks)
            if chunk.content_hash not in self._embedding_cache
        ]
        if uncached:
            indices, texts = zip(*uncached, strict=True)
            new_embeddings = model.encode(list(texts))
            for index, embedding in zip(indices, new_embeddings, strict=True):
                self._embedding_cache[chunks[index].content_hash] = embedding
        return np.array(
            [self._embedding_cache[chunk.content_hash] for chunk in chunks], dtype=np.float32
        )

    def _enrich_for_bm25(self, chunk: Chunk) -> str:
        """Append file stem to BM25 content to boost path-based queries."""
        stem = Path(chunk.file_path).stem
        return f"{chunk.content} {stem} {stem}"
