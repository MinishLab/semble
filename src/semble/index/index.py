from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from bm25s import BM25
from vicinity import Vicinity

from semble.index.create import create_index_from_path
from semble.index.dense import load_model
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.types import Chunk, Encoder, IndexStats, SearchMode, SearchResult


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: Encoder,
        bm25_index: BM25,
        semantic_index: Vicinity,
        chunks: list[Chunk],
        index_root: Path,
    ) -> None:
        """Configure the index.

        :param model: Embedding model to use.
        :param bm25_index: The bm25 index.
        :param semantic_index: The semantic index.
        :param chunks: The found chunks.
        :param index_root: The root of the index.
        """
        self.model: Encoder = model
        self.chunks: list[Chunk] = chunks
        self.stats = IndexStats()
        self._bm25_index: BM25 = bm25_index
        self._semantic_index: Vicinity = semantic_index
        self._index_root: Path = index_root

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> SembleIndex:
        """Create and index a SembleIndex from a directory.

        :param path: Root directory to index.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :return: An indexed SembleIndex.
        """
        model = model or load_model()
        bm25, vicinity, chunks, stats, index_root = cls.index(
            path, model=model, extensions=extensions, ignore=ignore, include_docs=include_docs
        )

        index = SembleIndex(model, bm25, vicinity, chunks, index_root)
        index.stats = stats

        return index

    @classmethod
    def from_git(
        cls,
        url: str,
        ref: str | None = None,
        model: Encoder | None = None,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> SembleIndex:
        """Clone a git repository and index it.

        :param url: URL of the git repository to clone (any git provider).
        :param ref: Branch or tag to check out. Defaults to the remote HEAD.
        :param model: Embedding model to use. Defaults to potion-code-16M.
        :param extensions: File extensions to include. Defaults to a standard set of code extensions.
        :param ignore: Directory names to skip. Defaults to common VCS and build dirs.
        :param include_docs: If True, also index documentation files (.md, .yaml, etc.).
        :return: An indexed SembleIndex. Chunk file paths are repo-relative (e.g. ``src/foo.py``).
        :raises RuntimeError: If git is not on PATH or the clone fails.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = ["git", "clone", "--depth", "1", *(["--branch", ref] if ref else []), url, tmp_dir]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
            except FileNotFoundError:
                raise RuntimeError("git is not installed or not on PATH") from None
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed for {url!r}:\n{result.stderr.strip()}")
            model = model or load_model()
            resolved_path = Path(tmp_dir).resolve()
            bm25, vicinity, chunks, stats, index_root = create_index_from_path(
                resolved_path,
                model=model,
                extensions=extensions,
                ignore=ignore,
                include_docs=include_docs,
                display_root=resolved_path,
            )

            index = SembleIndex(model, bm25, vicinity, chunks, index_root)
            index.stats = stats

            return index

    @classmethod
    def index(
        cls,
        path: str | Path,
        model: Encoder,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
    ) -> tuple[BM25, Vicinity, list[Chunk], IndexStats, Path]:
        """Index a directory using the backend configured at construction time.

        :param path: Root directory to index.
        :param model: The model used to index.
        :param extensions: File extensions to include.
        :param ignore: Directory names to skip.
        :param include_docs: If True, also index documentation files.
        :return: Statistics about the indexed files and chunks.
        """
        return create_index_from_path(
            Path(path).resolve(), model, extensions=extensions, ignore=ignore, include_docs=include_docs
        )

    def find_related(self, file_path: str, line: int, top_k: int = 5) -> list[SearchResult]:
        """Return chunks semantically similar to the chunk at the given file location.

        :param file_path: Path to the file, in the same format stored by the index.
            For indexes built with `from_path` this is an absolute path; for
            indexes built with `from_git` this is a repo-relative path
            (e.g. ``src/foo.py``).  Use `chunk.file_path` from a prior search result
            to guarantee the correct format.
        :param line: Line number (1-indexed) used to identify the source chunk.
        :param top_k: Number of similar chunks to return.
        :return: Ranked list of SearchResult objects, most similar first.
        """
        target = next(
            (c for c in self.chunks if c.file_path == file_path and c.start_line <= line <= c.end_line),
            None,
        )
        if target is None:
            return []
        results = search_semantic(target.content, self.model, self._semantic_index, top_k + 1)
        return [r for r in results if r.chunk != target][:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | str = SearchMode.HYBRID,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Search the index and return the top-k most relevant chunks.

        :param query: Natural-language or keyword query string.
        :param top_k: Maximum number of results to return.
        :param mode: Search strategy — "hybrid" (default), "semantic", or "bm25".
        :param alpha: Blend weight for hybrid score combination; 1.0 = full semantic
            weight, 0.0 = full BM25 weight. File-path penalties and diversity reranking
            are applied regardless. ``None`` auto-detects from query type.
        :return: Ranked list of :class:`SearchResult` objects, best match first.
        :raises ValueError: If `mode` is not a recognised search strategy.
        """
        bm25_index, semantic_index = self._bm25_index, self._semantic_index
        if not self.chunks:
            return []

        if mode == SearchMode.BM25:
            return search_bm25(query, bm25_index, self.chunks, top_k)

        if mode == SearchMode.SEMANTIC:
            return search_semantic(query, self.model, semantic_index, top_k)
        if mode == SearchMode.HYBRID:
            return search_hybrid(query, self.model, semantic_index, bm25_index, self.chunks, top_k, alpha=alpha)
        raise ValueError(f"Unknown search mode: {mode!r}")
