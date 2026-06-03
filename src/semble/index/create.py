from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
from pathlib import Path
from threading import Lock

from model2vec.model import StaticModel
from vicinity.backends.basic import BasicArgs

from semble.cache import build_git_cache_save_metadata
from semble.chunking import chunk_source
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import walk_files
from semble.index.files import FileStatus, detect_language, get_extensions, get_file_status, read_file_text
from semble.index.sparse import SparseIndex, TantivySparseIndex
from semble.types import Chunk, ContentType


@dataclass(frozen=True, slots=True)
class ChunkFileResult:
    chunks: list[Chunk]
    file_size: int | None
    file_hash: str | None


@dataclass(frozen=True, slots=True)
class CachedChunk:
    content: str
    start_line: int
    end_line: int
    language: str | None


class ChunkTemplateCache:
    """Thread-safe cache for chunks derived from exact duplicate file contents."""

    def __init__(self) -> None:
        """Create an empty template cache for one index build."""
        self._templates: dict[tuple[str | None, str], Future[list[CachedChunk]]] = {}
        self._lock = Lock()

    def chunks_for(self, source: str, file_path: str, language: str | None) -> tuple[list[Chunk], str]:
        """Return chunks for source and path, reusing templates for exact duplicate content."""
        file_hash = hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()
        cache_key = (language, file_hash)
        with self._lock:
            future = self._templates.get(cache_key)
            should_compute = future is None
            if future is None:
                future = Future()
                self._templates[cache_key] = future
        if should_compute:
            try:
                chunks = chunk_source(source, file_path, language)
                future.set_result(
                    [CachedChunk(chunk.content, chunk.start_line, chunk.end_line, chunk.language) for chunk in chunks]
                )
            except BaseException as exc:
                future.set_exception(exc)
                with self._lock:
                    if self._templates.get(cache_key) is future:
                        del self._templates[cache_key]
                raise
        cached_chunks = future.result()
        return [
            Chunk(cached.content, file_path, cached.start_line, cached.end_line, cached.language)
            for cached in cached_chunks
        ], file_hash


@dataclass(frozen=True, slots=True)
class IndexBuild:
    sparse_index: SparseIndex
    semantic_index: SelectableBasicBackend
    chunks: list[Chunk]
    file_sizes: dict[str, int]
    file_hashes: dict[str, str] = field(default_factory=dict)
    git_cache_metadata: tuple[dict[str, str], ...] | None = None
    tracked_paths: tuple[str, ...] | None = None


def _default_chunk_worker_count() -> int:
    configured = os.environ.get("SEMBLE_CHUNK_WORKERS")
    if configured:
        with contextlib.suppress(ValueError):
            return max(1, int(configured))
    return min(32, max(1, os.cpu_count() or 1))


_CHUNK_WORKER_COUNT = _default_chunk_worker_count()


def _chunk_file(
    file_path: Path,
    display_root: Path | None,
    template_cache: ChunkTemplateCache | None = None,
) -> ChunkFileResult:
    language = detect_language(file_path)
    with contextlib.suppress(OSError):
        file_status = get_file_status(file_path, None)
        if file_status != FileStatus.VALID:
            return ChunkFileResult([], None, None)
        source = read_file_text(file_path)
        chunk_path = file_path.relative_to(display_root) if display_root else file_path
        if template_cache is None:
            chunks = chunk_source(source, str(chunk_path), language)
            file_hash = hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()
        else:
            chunks, file_hash = template_cache.chunks_for(source, str(chunk_path), language)
        if not chunks:
            return ChunkFileResult([], None, None)
        return ChunkFileResult(chunks, len(source), file_hash)
    return ChunkFileResult([], None, None)


ChunkCollection = tuple[list[Chunk], dict[str, int], dict[str, str]]
GitMetadata = tuple[tuple[dict[str, str], ...], tuple[str, ...]]
GitMetadataJob = tuple[ThreadPoolExecutor, Future[tuple[list[dict[str, str]], list[str]] | None]]


def _collect_chunks_serial(
    files: Iterable[Path],
    display_root: Path | None,
    template_cache: ChunkTemplateCache | None,
) -> ChunkCollection:
    return _merge_chunk_file_results(_chunk_file(file_path, display_root, template_cache) for file_path in files)


def _collect_chunks_parallel(
    files: Iterable[Path],
    display_root: Path | None,
    template_cache: ChunkTemplateCache | None,
) -> ChunkCollection:
    with ThreadPoolExecutor(max_workers=_CHUNK_WORKER_COUNT) as executor:
        return _merge_chunk_file_results(executor.map(_chunk_file, files, repeat(display_root), repeat(template_cache)))


def _merge_chunk_file_results(results: Iterable[ChunkFileResult]) -> ChunkCollection:
    chunks: list[Chunk] = []
    file_sizes: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for result in results:
        chunks.extend(result.chunks)
        if result.file_size is not None and result.chunks:
            file_path = result.chunks[0].file_path
            file_sizes.setdefault(file_path, result.file_size)
            if result.file_hash is not None:
                file_hashes.setdefault(file_path, result.file_hash)
    return chunks, file_sizes, file_hashes


def _git_metadata_from_job(job: GitMetadataJob | None) -> GitMetadata | None:
    if job is None:
        return None
    executor, future = job
    try:
        metadata = future.result()
    finally:
        executor.shutdown(wait=True)
    if metadata is None:
        return None
    git_roots, tracked_paths = metadata
    return tuple(git_roots), tuple(tracked_paths)


def _start_git_metadata_job(path: Path, display_root: Path | None, file_paths: Sequence[str]) -> GitMetadataJob | None:
    if display_root is None:
        return None
    try:
        display_root = display_root.resolve()
        path.resolve().relative_to(display_root)
    except (OSError, ValueError):
        return None
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(build_git_cache_save_metadata, display_root, file_paths)
    return executor, future


def _collect_chunks(
    path: Path,
    extensions: Sequence[str],
    display_root: Path | None,
) -> ChunkCollection:
    files = walk_files(path, extensions)
    template_cache = ChunkTemplateCache()
    if _CHUNK_WORKER_COUNT <= 1:
        return _collect_chunks_serial(files, display_root, template_cache)
    return _collect_chunks_parallel(files, display_root, template_cache)


def create_index_build_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
) -> IndexBuild:
    """Create an index build from a resolved directory, reusing first-read file sizes."""
    normalized = (content,) if isinstance(content, ContentType) else content
    chunks, file_sizes, file_hashes = _collect_chunks(path, get_extensions(normalized), display_root)

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    git_metadata_job = _start_git_metadata_job(path, display_root, sorted(file_sizes))
    try:
        embeddings = embed_chunks(model, chunks)
        sparse_index = TantivySparseIndex.build_temporary(chunks)
        args = BasicArgs()
        semantic_index = SelectableBasicBackend(embeddings, args)
        git_metadata = _git_metadata_from_job(git_metadata_job)
        git_metadata_job = None
    finally:
        if git_metadata_job is not None:
            git_metadata_job[0].shutdown(wait=True)
    git_roots, tracked_paths = (None, None) if git_metadata is None else git_metadata
    return IndexBuild(sparse_index, semantic_index, chunks, file_sizes, file_hashes, git_roots, tracked_paths)


def create_index_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
) -> tuple[SparseIndex, SelectableBasicBackend, list[Chunk]]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :raises ValueError: if no items were found, no index can be created.
    :return: A sparse index, vicinity index and list of chunks
    """
    build = create_index_build_from_path(path, model, content, display_root)
    return build.sparse_index, build.semantic_index, build.chunks
