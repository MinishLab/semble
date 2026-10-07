import contextlib
import itertools
import logging
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import threading
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from model2vec.model import StaticModel
from tqdm import tqdm
from vicinity.backends.basic import BasicArgs

from semble.chunking import chunk_source
from semble.index.bm25 import BM25
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import walk_files
from semble.index.files import (
    MAX_FILE_BYTES,
    FileStatus,
    detect_language,
    get_extensions,
    get_file_status,
    read_file_text,
)
from semble.index.sparse import enrich_for_bm25
from semble.index.types import FileManifestEntry, PreviousIndex, make_chunk_id
from semble.tokens import tokenize
from semble.types import Chunk, ContentType, EmbeddingMatrix

logger = logging.getLogger(__name__)

# Processes that chunk files when at least _MIN_FILES_FOR_PROCESSES need chunking; 0 chunks in the main process.
_WORKERS = int(os.environ.get("SEMBLE_INDEX_WORKERS", 4))
_MIN_FILES_FOR_PROCESSES = 200
# Files handed to the workers but not yet consumed by the main process.
_MAX_FILES_IN_FLIGHT = 256

# A file's chunks and their BM25 tokens.
_ChunkedFile = tuple[list[Chunk], list[list[str]]]


def _warn_skipped_large(skipped_large: list[str]) -> None:
    """Warn about files skipped for exceeding the maximum indexable file size."""
    if skipped_large:
        logger.warning(
            "Skipped %d file(s) exceeding the maximum file size of %d bytes "
            "(raise SEMBLE_MAX_FILE_BYTES to include them): %s%s",
            len(skipped_large),
            MAX_FILE_BYTES,
            ", ".join(skipped_large[:5]),
            " ..." if len(skipped_large) > 5 else "",
        )


def _chunk_file(file_path: Path, indexed_path: str) -> _ChunkedFile | None:
    """Chunk a file and tokenize its chunks for BM25, or return None if it can't be read."""
    try:
        source = read_file_text(file_path)
    except OSError:
        return None
    file_chunks = chunk_source(source, indexed_path, detect_language(file_path))
    return file_chunks, [tokenize(enrich_for_bm25(chunk)) for chunk in file_chunks]


def _init_worker() -> None:
    """Leave Ctrl-C to the parent, and exit this worker once the parent is gone, even if it never shut the pool down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    assert parent is not None

    def wait_and_exit() -> None:
        multiprocessing.connection.wait([parent.sentinel])
        os._exit(0)

    threading.Thread(target=wait_and_exit, daemon=True).start()


def _chunk_files(files: list[tuple[Path, str]]) -> Iterator[_ChunkedFile | None]:
    """Chunk files in order, in worker processes when there are enough of them."""
    if _WORKERS == 0 or len(files) < _MIN_FILES_FOR_PROCESSES:
        yield from itertools.starmap(_chunk_file, files)
    else:
        with ProcessPoolExecutor(
            max_workers=_WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
        ) as executor:
            submitted = (executor.submit(_chunk_file, *file) for file in files)
            pending = deque(itertools.islice(submitted, _MAX_FILES_IN_FLIGHT))
            while pending:
                yield pending.popleft().result()
                pending.extend(itertools.islice(submitted, 1))


def _stat_files(
    path: Path,
    extensions: Sequence[str],
    display_root: Path | None,
    previous_manifest: dict[str, FileManifestEntry],
    skipped_large: list[str],
) -> list[tuple[Path, str, int, FileManifestEntry | None]]:
    """Return each indexable file's path, indexed path, mtime and previous manifest entry."""
    files = []
    for file_path in walk_files(path, extensions):
        with contextlib.suppress(OSError):
            stat = file_path.stat()
            file_status = get_file_status(file_path, stat)
            if file_status is FileStatus.TOO_LARGE:
                skipped_large.append(str(file_path))
            if file_status != FileStatus.VALID:
                continue

            indexed_path = str(file_path.relative_to(display_root) if display_root else file_path)
            files.append((file_path, indexed_path, stat.st_mtime_ns, previous_manifest.get(indexed_path)))
    return files


def _reindex_file(
    bm25_index: BM25,
    indexed_path: str,
    file_tokens: list[list[str]],
    previous_entry: FileManifestEntry | None,
) -> None:
    """Replace a file's BM25 postings: remove its old slots (if any), then add its new ones."""
    if previous_entry is not None:
        for slot in range(previous_entry.count):
            bm25_index.remove_document(make_chunk_id(indexed_path, slot))
    for slot, tokens in enumerate(file_tokens):
        bm25_index.add_document(make_chunk_id(indexed_path, slot), tokens)


def _has_same_vector_layout(
    manifest: dict[str, FileManifestEntry], previous_manifest: dict[str, FileManifestEntry]
) -> bool:
    """Return whether both manifests use the same chunk ranges."""
    return len(manifest) == len(previous_manifest) and all(
        (previous_entry := previous_manifest.get(indexed_path)) is not None
        and entry.start == previous_entry.start
        and entry.count == previous_entry.count
        for indexed_path, entry in manifest.items()
    )


def create_index_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous: PreviousIndex | None = None,
    show_progress_bar: bool = False,
) -> tuple[BM25, SelectableBasicBackend, list[Chunk], dict[str, FileManifestEntry]]:
    """Create an index from a resolved directory, optionally reusing a previous index's unchanged files.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous: A previously built index to reuse unchanged files' chunks/embeddings/postings from.
    :param show_progress_bar: Show a progress bar on stderr while indexing.
    :raises ValueError: if no items were found, no index can be created.
    :return: A BM25 index, semantic index, list of chunks, and file manifest.
    """
    # PreviousIndex is consumed; mutate BM25 in place to avoid a copy.
    bm25_index = previous.bm25_index if previous is not None else BM25.empty()
    previous_manifest = previous.manifest if previous is not None else {}

    normalized = (content,) if isinstance(content, ContentType) else content
    resolved_extensions = get_extensions(normalized)

    chunks: list[Chunk] = []
    chunk_ids: list[str] = []
    vector_parts: list[EmbeddingMatrix] = []
    manifest: dict[str, FileManifestEntry] = {}
    embedding_parts: list[tuple[int, int, int]] = []

    skipped_large: list[str] = []

    files = _stat_files(path, resolved_extensions, display_root, previous_manifest, skipped_large)
    stale = [
        (file_path, indexed_path)
        for file_path, indexed_path, mtime_ns, previous_entry in files
        if previous_entry is None or previous_entry.mtime_ns != mtime_ns
    ]

    chunked = _chunk_files(stale)
    for _, indexed_path, mtime_ns, previous_entry in tqdm(
        files,
        desc="Indexing",
        unit="file",
        file=sys.stderr,
        leave=False,
        colour="green",
        miniters=1,  # tqdm's adaptive miniters stalls the bar when fast files are followed by slow ones
        disable=not show_progress_bar,
    ):
        if previous is not None and previous_entry is not None and previous_entry.mtime_ns == mtime_ns:
            file_chunks = previous.chunks[previous_entry.start : previous_entry.end]
            vector_parts.append(previous.vectors[previous_entry.start : previous_entry.end])
        else:
            result = next(chunked)
            if result is None:
                continue
            file_chunks, file_tokens = result
            _reindex_file(bm25_index, indexed_path, file_tokens, previous_entry)

            embedding_parts.append((len(vector_parts), len(chunks), len(file_chunks)))
            vector_parts.append(embed_chunks(model, file_chunks))

        start = len(chunks)
        chunks.extend(file_chunks)
        chunk_ids.extend(make_chunk_id(indexed_path, slot) for slot in range(len(file_chunks)))
        manifest[indexed_path] = FileManifestEntry(mtime_ns=mtime_ns, start=start, count=len(file_chunks))

    for indexed_path in previous_manifest.keys() - manifest.keys():
        _reindex_file(bm25_index, indexed_path, [], previous_manifest[indexed_path])

    _warn_skipped_large(skipped_large)

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    if previous is not None and _has_same_vector_layout(manifest, previous_manifest):
        embeddings = previous.vectors
        for vector_part, start, count in embedding_parts:
            embeddings[start : start + count] = vector_parts[vector_part]
    else:
        embeddings = np.vstack(vector_parts)
    bm25_index.set_doc_order(chunk_ids)
    semantic_index = SelectableBasicBackend(embeddings, BasicArgs())

    return bm25_index, semantic_index, chunks, manifest
