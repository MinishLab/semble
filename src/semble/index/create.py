import contextlib
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from model2vec.model import StaticModel
from vicinity.backends.basic import BasicArgs

from semble.chunking import chunk_source
from semble.index.bm25 import BM25
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import walk_files
from semble.index.files import (
    FileStatus,
    detect_language,
    get_extensions,
    get_file_status,
)
from semble.index.sparse import enrich_for_bm25
from semble.index.types import FileManifestEntry, PreviousIndex, make_chunk_id
from semble.tokens import tokenize
from semble.types import Chunk, ContentType, EmbeddingMatrix


def _reindex_file(
    bm25_index: BM25,
    indexed_path: str,
    file_chunks: list[Chunk],
    previous_entry: FileManifestEntry | None,
) -> None:
    """Replace a file's BM25 postings: remove its old slots (if any), then add its new ones."""
    if previous_entry is not None:
        for slot in range(previous_entry.count):
            bm25_index.remove_document(make_chunk_id(indexed_path, slot))
    for slot, chunk in enumerate(file_chunks):
        bm25_index.add_document(make_chunk_id(indexed_path, slot), tokenize(enrich_for_bm25(chunk)))


def _remove_deleted_files(
    bm25_index: BM25,
    manifest: dict[str, FileManifestEntry],
    previous_manifest: dict[str, FileManifestEntry],
) -> None:
    """Remove BM25 documents belonging to files absent from the new manifest."""
    for indexed_path, previous_entry in previous_manifest.items():
        if indexed_path not in manifest:
            for slot in range(previous_entry.count):
                bm25_index.remove_document(make_chunk_id(indexed_path, slot))


def _embed_changed_chunks(
    model: StaticModel,
    embedding_parts: list[tuple[int, int, int]],
    changed_chunks: list[Chunk],
    vector_parts: list[EmbeddingMatrix],
) -> None:
    """Embed changed chunks in one batch and restore their per-file rows."""
    if not changed_chunks:
        return
    changed_embeddings = embed_chunks(model, changed_chunks)
    offset = 0
    for index, _, count in embedding_parts:
        vector_parts[index] = changed_embeddings[offset : offset + count]
        offset += count


def create_index_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
    previous: PreviousIndex | None = None,
) -> tuple[BM25, SelectableBasicBackend, list[Chunk], dict[str, FileManifestEntry]]:
    """Create an index from a resolved directory, optionally reusing a previous index's unchanged files.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :param previous: A previously built index to reuse unchanged files' chunks/embeddings/postings from.
    :raises ValueError: if no items were found, no index can be created.
    :return: A BM25 index, semantic index, list of chunks, and file manifest.
    """
    # PreviousIndex is consumed; mutate BM25 in place to avoid a copy.
    bm25_index = previous.bm25_index if previous is not None else BM25()
    previous_manifest = previous.manifest if previous is not None else {}

    normalized = (content,) if isinstance(content, ContentType) else content
    resolved_extensions = get_extensions(normalized)

    chunks: list[Chunk] = []
    chunk_ids: list[str] = []
    vector_parts: list[EmbeddingMatrix] = []
    manifest: dict[str, FileManifestEntry] = {}
    embedding_parts: list[tuple[int, int, int]] = []
    changed_chunks: list[Chunk] = []

    for file_path in walk_files(path, resolved_extensions):
        language = detect_language(file_path)
        with contextlib.suppress(OSError):
            file_status = get_file_status(file_path, None)
            if file_status != FileStatus.VALID:
                continue

            indexed_path = str(file_path.relative_to(display_root) if display_root else file_path)
            raw_source = file_path.read_bytes()
            file_hash = hashlib.sha256(raw_source).hexdigest()
            previous_entry = previous_manifest.get(indexed_path)

            if previous is not None and previous_entry is not None and previous_entry.file_hash == file_hash:
                file_chunks = previous.chunks[previous_entry.start : previous_entry.start + previous_entry.count]
                vector_parts.append(
                    previous.vectors[previous_entry.start : previous_entry.start + previous_entry.count]
                )
            else:
                source = raw_source.decode("utf-8", errors="replace")
                file_chunks = chunk_source(source, indexed_path, language)
                _reindex_file(bm25_index, indexed_path, file_chunks, previous_entry)

                if file_chunks:
                    embedding_parts.append((len(vector_parts), len(chunks), len(file_chunks)))
                    vector_parts.append(np.empty((0, 0), dtype=np.float32))  # Replaced after batched embedding.
                    changed_chunks.extend(file_chunks)
                else:
                    vector_parts.append(np.empty((0, model.dim), dtype=np.float32))

            start = len(chunks)
            chunks.extend(file_chunks)
            chunk_ids.extend(make_chunk_id(indexed_path, slot) for slot in range(len(file_chunks)))
            manifest[indexed_path] = FileManifestEntry(file_hash=file_hash, start=start, count=len(file_chunks))

    if previous is not None:
        _remove_deleted_files(bm25_index, manifest, previous_manifest)

    if not chunks:
        raise ValueError(f"No supported files found under {path}.")

    if previous is None:
        embeddings = embed_chunks(model, chunks)
    else:
        _embed_changed_chunks(model, embedding_parts, changed_chunks, vector_parts)
        same_vector_layout = len(manifest) == len(previous_manifest) and all(
            (previous_entry := previous_manifest.get(indexed_path)) is not None
            and entry.start == previous_entry.start
            and entry.count == previous_entry.count
            for indexed_path, entry in manifest.items()
        )
        if same_vector_layout:
            embeddings = previous.vectors
            for vector_part, start, count in embedding_parts:
                embeddings[start : start + count] = vector_parts[vector_part]
        else:
            embeddings = np.vstack(vector_parts)
    bm25_index.set_doc_order(chunk_ids)
    semantic_index = SelectableBasicBackend(embeddings, BasicArgs())

    return bm25_index, semantic_index, chunks, manifest
