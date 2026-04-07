"""Core index engine: the main entry point for semble."""

from __future__ import annotations

import ast
import contextlib
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from model2vec import StaticModel

from semble._chunker import chunk_file
from semble._search import (
    BM25Index,
    SemanticIndex,
    dedup_results,
    search_bm25,
    search_hybrid_alpha,
    search_hybrid_rrf,
    search_semantic,
    search_symbol,
)
from semble._types import Chunk, IndexStats, SearchResult, SymbolKind

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".sh",
        ".sql",
    }
)
ALL_EXTENSIONS: frozenset[str] = CODE_EXTENSIONS | frozenset(
    {".md", ".yaml", ".yml", ".toml", ".json"}
)

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


def _extract_python_docstrings(path: Path) -> dict[str, str]:
    """Extract docstrings keyed by symbol name from a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return {}
    ds: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            docstring = ast.get_docstring(node)
            if docstring:
                ds[node.name] = docstring
    return ds


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
        self._docstrings: dict[str, dict[str, str]] = {}
        self._bm25_index: BM25Index | None = None
        self._semantic_index: SemanticIndex | None = None
        self._hash_to_chunk: dict[str, Chunk] = {}
        self._file_lines: dict[str, list[str]] = {}
        self._stats = IndexStats()

    @property
    def model(self) -> StaticModel:
        """Lazily load the embedding model."""
        if self._model is None:
            from model2vec import StaticModel

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
        files = list(self._walk_files(path, extensions, ignore))

        all_chunks: list[Chunk] = []
        lang_counts: dict[str, int] = {}

        self._file_lines = {}
        for fp in files:
            file_chunks = chunk_file(fp)
            all_chunks.extend(file_chunks)
            for c in file_chunks:
                if c.language:
                    lang_counts[c.language] = lang_counts.get(c.language, 0) + 1
            fp_str = str(fp)
            with contextlib.suppress(OSError):
                self._file_lines[fp_str] = fp.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            if fp.suffix == ".py":
                self._docstrings[fp_str] = _extract_python_docstrings(fp)

        t_emb = time.perf_counter()
        embeddings = self._embed_chunks(all_chunks)
        t_emb = time.perf_counter() - t_emb

        self._chunks = all_chunks
        self._hash_to_chunk = {c.content_hash: c for c in all_chunks}

        if all_chunks:
            enriched = [self._enrich_for_bm25(c) for c in all_chunks]
            enriched_chunks = [
                Chunk(
                    content=enriched[i],
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    symbol_name=c.symbol_name,
                    symbol_kind=c.symbol_kind,
                    language=c.language,
                    content_hash=c.content_hash,
                )
                for i, c in enumerate(all_chunks)
            ]
            self._bm25_index = BM25Index(enriched_chunks)
            self._semantic_index = SemanticIndex(all_chunks, embeddings)

        self._stats = IndexStats(
            total_files=len(files),
            total_chunks=len(all_chunks),
            total_symbols=sum(1 for c in all_chunks if c.symbol_kind != SymbolKind.CHUNK),
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
        dedup: bool = True,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Search the index.

        :param query: Natural language or code query.
        :param top_k: Number of results to return.
        :param mode: Search mode — one of "hybrid", "hybrid_rrf", "semantic", "bm25", "symbol".
        :param dedup: If True, remove near-duplicate results.
        :param alpha: Semantic weight for hybrid mode (1-alpha goes to BM25). Default 0.7.
        :returns: List of search results, best first.
        :raises ValueError: If mode is not recognized.
        """
        if not self._chunks:
            return []

        if mode == "semantic":
            if self._semantic_index is None:
                return []
            results = search_semantic(
                query,
                self.model,
                self._semantic_index,
                self._chunks,
                self._hash_to_chunk,
                top_k * 2,
            )
        elif mode == "bm25":
            if self._bm25_index is None:
                return []
            results = search_bm25(query, self._bm25_index, self._chunks, top_k * 2)
        elif mode == "symbol":
            results = search_symbol(query, self._file_lines, top_k)
        elif mode == "hybrid":
            if self._semantic_index is None or self._bm25_index is None:
                return []
            results = search_hybrid_alpha(
                query,
                self.model,
                self._semantic_index,
                self._bm25_index,
                self._chunks,
                self._hash_to_chunk,
                top_k * 2,
                alpha=alpha,
            )
        elif mode == "hybrid_rrf":
            if self._semantic_index is None or self._bm25_index is None:
                return []
            results = search_hybrid_rrf(
                query,
                self.model,
                self._semantic_index,
                self._bm25_index,
                self._chunks,
                self._hash_to_chunk,
                top_k * 2,
            )
        else:
            raise ValueError(
                f"Unknown search mode: {mode!r}. "
                "Choose from: hybrid, hybrid_rrf, semantic, bm25, symbol"
            )

        if dedup and len(results) > 1:
            results = dedup_results(results, self._embedding_cache)

        return results[:top_k]

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
        results = search_semantic(
            target.content,
            self.model,
            self._semantic_index,
            self._chunks,
            self._hash_to_chunk,
            top_k + 1,
        )
        return [r for r in results if r.chunk.content_hash != target.content_hash][:top_k]

    @property
    def stats(self) -> IndexStats:
        return self._stats

    # -- Private helpers --

    def _walk_files(
        self,
        root: Path,
        extensions: frozenset[str],
        ignore: frozenset[str],
    ) -> Generator[Path, None, None]:
        for item in sorted(root.rglob("*")):
            if any(part in ignore for part in item.parts):
                continue
            if item.is_file() and item.suffix.lower() in extensions:
                yield item

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
        """Append file stem and docstring to BM25 content for better recall."""
        parts = [chunk.content]
        stem = Path(chunk.file_path).stem
        parts.append(f" {stem} {stem}")
        ds = self._docstrings.get(chunk.file_path, {})
        if chunk.symbol_name and chunk.symbol_name in ds:
            parts.append(ds[chunk.symbol_name])
        return " ".join(parts)
