from __future__ import annotations

import contextlib
import subprocess
import tempfile
from pathlib import Path

import bm25s
import numpy as np
import numpy.typing as npt
from huggingface_hub import utils as hf_utils
from model2vec import StaticModel
from vicinity import Metric, Vicinity

from semble.chunker import chunk_source
from semble.file_walker import language_for_path, resolve_extensions, walk_files
from semble.search import search_bm25, search_hybrid, search_semantic
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, IndexStats, SearchMode, SearchResult

DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"


class SembleIndex:
    """Fast local code index with hybrid search."""

    def __init__(
        self,
        model: Encoder | None = None,
    ) -> None:
        """Configure the index.

        :param model: Embedding model to use. Defaults to potion-code-16M.
        """
        self.model: Encoder | None = model
        self.chunks: list[Chunk] = []
        self.stats = IndexStats()
        self._bm25_index: bm25s.BM25 | None = None
        self._semantic_index: Vicinity | None = None
        self._index_root: Path | None = None

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
        instance = cls(model=model)
        instance.index(path, extensions=extensions, ignore=ignore, include_docs=include_docs)
        return instance

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
            instance = cls(model=model)
            instance._index_path(
                Path(tmp_dir).resolve(),
                extensions=extensions,
                ignore=ignore,
                include_docs=include_docs,
                display_root=Path(tmp_dir).resolve(),
            )
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
        return self._index_path(Path(path).resolve(), extensions=extensions, ignore=ignore, include_docs=include_docs)

    def _index_path(
        self,
        path: Path,
        extensions: frozenset[str] | None = None,
        ignore: frozenset[str] | None = None,
        include_docs: bool = False,
        display_root: Path | None = None,
    ) -> IndexStats:
        """Index a resolved directory, optionally storing chunk paths relative to display_root.

        :param path: Resolved absolute path to index.
        :param extensions: File extensions to include.
        :param ignore: Directory names to skip.
        :param include_docs: If True, also index documentation files.
        :param display_root: If set, chunk file paths are stored relative to this root.
        :return: Statistics about the indexed files and chunks.
        """
        self._index_root = None if display_root is not None else path
        extensions = resolve_extensions(extensions, include_docs=include_docs)

        all_chunks: list[Chunk] = []
        language_counts: dict[str, int] = {}
        indexed_files = 0

        for file_path in walk_files(path, extensions, ignore):
            language = language_for_path(file_path)
            with contextlib.suppress(OSError):
                source = file_path.read_text(encoding="utf-8", errors="replace")
                indexed_files += 1
                chunk_path = str(file_path.relative_to(display_root)) if display_root else str(file_path)
                file_chunks = chunk_source(source, chunk_path, language)
                all_chunks.extend(file_chunks)
                for chunk in file_chunks:
                    if chunk.language:
                        language_counts[chunk.language] = language_counts.get(chunk.language, 0) + 1

        self.chunks = all_chunks

        if all_chunks:
            embeddings = self._embed_chunks(all_chunks)
            self._bm25_index = bm25s.BM25()
            self._bm25_index.index(
                [tokenize(self._enrich_for_bm25(chunk, self._index_root)) for chunk in all_chunks],
                show_progress=False,
            )
            self._semantic_index = Vicinity.from_vectors_and_items(embeddings, all_chunks, metric=Metric.COSINE)
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

    def _ensure_model(self) -> Encoder:
        """Return the current model, loading the default if none was provided."""
        if self.model is None:
            # Disable HF progress bars since the model is loaded silently in the background during indexing.
            hf_utils.disable_progress_bars()
            try:
                self.model = StaticModel.from_pretrained(DEFAULT_MODEL_NAME)
            finally:
                hf_utils.enable_progress_bars()
        return self.model

    def _embed_chunks(self, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
        """Embed chunks using the configured model."""
        if not chunks:
            return np.empty((0, 256), dtype=np.float32)
        model = self._ensure_model()
        return np.array(model.encode([c.content for c in chunks]), dtype=np.float32)

    def _enrich_for_bm25(self, chunk: Chunk, root: Path | None) -> str:
        """Append file path components to BM25 content to boost path-based queries.

        Uses a repo-relative path so that machine-specific directory components
        (usernames, workspace names, temp dirs) are never indexed as tokens.
        """
        path = Path(chunk.file_path)
        if root is not None:
            with contextlib.suppress(ValueError):
                path = path.relative_to(root)
        stem = path.stem
        # Collect directory names from the (now relative) path, skipping filesystem roots.
        dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
        dir_text = " ".join(dir_parts[-3:])  # Last 3 repo-relative directory components
        # Repeat the stem twice to up-weight file-path matches in BM25.
        return f"{chunk.content} {stem} {stem} {dir_text}"
