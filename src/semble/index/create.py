from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

from model2vec.model import StaticModel
from vicinity.backends.basic import BasicArgs

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


@dataclass(frozen=True, slots=True)
class IndexBuild:
    sparse_index: SparseIndex
    semantic_index: SelectableBasicBackend
    chunks: list[Chunk]
    file_sizes: dict[str, int]


def _default_chunk_worker_count() -> int:
    configured = os.environ.get("SEMBLE_CHUNK_WORKERS")
    if configured:
        with contextlib.suppress(ValueError):
            return max(1, int(configured))
    return min(32, max(1, os.cpu_count() or 1))


_CHUNK_WORKER_COUNT = _default_chunk_worker_count()


def _chunk_file(file_path: Path, display_root: Path | None) -> ChunkFileResult:
    language = detect_language(file_path)
    with contextlib.suppress(OSError):
        file_status = get_file_status(file_path, None)
        if file_status != FileStatus.VALID:
            return ChunkFileResult([], None)
        source = read_file_text(file_path)
        chunk_path = file_path.relative_to(display_root) if display_root else file_path
        chunks = chunk_source(source, str(chunk_path), language)
        return ChunkFileResult(chunks, len(source) if chunks else None)
    return ChunkFileResult([], None)


def _collect_chunks_serial(files: Iterable[Path], display_root: Path | None) -> tuple[list[Chunk], dict[str, int]]:
    return _merge_chunk_file_results(_chunk_file(file_path, display_root) for file_path in files)


def _collect_chunks_parallel(files: Iterable[Path], display_root: Path | None) -> tuple[list[Chunk], dict[str, int]]:
    with ThreadPoolExecutor(max_workers=_CHUNK_WORKER_COUNT) as executor:
        return _merge_chunk_file_results(executor.map(_chunk_file, files, repeat(display_root)))


def _merge_chunk_file_results(results: Iterable[ChunkFileResult]) -> tuple[list[Chunk], dict[str, int]]:
    chunks: list[Chunk] = []
    file_sizes: dict[str, int] = {}
    for result in results:
        chunks.extend(result.chunks)
        if result.file_size is not None and result.chunks:
            file_sizes.setdefault(result.chunks[0].file_path, result.file_size)
    return chunks, file_sizes


def _collect_chunks(
    path: Path,
    extensions: Sequence[str],
    display_root: Path | None,
) -> tuple[list[Chunk], dict[str, int]]:
    files = walk_files(path, extensions)
    if _CHUNK_WORKER_COUNT <= 1:
        return _collect_chunks_serial(files, display_root)
    return _collect_chunks_parallel(files, display_root)


def create_index_build_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
) -> IndexBuild:
    """Create an index build from a resolved directory, reusing first-read file sizes."""
    normalized = (content,) if isinstance(content, ContentType) else content
    chunks, file_sizes = _collect_chunks(path, get_extensions(normalized), display_root)

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    embeddings = embed_chunks(model, chunks)
    sparse_index = TantivySparseIndex.build_temporary(chunks)
    args = BasicArgs()
    semantic_index = SelectableBasicBackend(embeddings, args)
    return IndexBuild(sparse_index, semantic_index, chunks, file_sizes)


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
