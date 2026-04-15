from __future__ import annotations

import contextlib
from pathlib import Path

import bm25s
from vicinity import Metric, Vicinity

from semble.chunker import chunk_source
from semble.file_walker import filter_extensions, language_for_path, walk_files
from semble.index.dense import embed_chunks
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, IndexStats


def create_index_from_path(
    path: Path,
    model: Encoder,
    extensions: frozenset[str] | None = None,
    ignore: frozenset[str] | None = None,
    include_docs: bool = False,
    display_root: Path | None = None,
) -> tuple[bm25s.BM25, Vicinity, list[Chunk], IndexStats, Path]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param extensions: File extensions to include.
    :param ignore: Directory names to skip.
    :param include_docs: If True, also index documentation files.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :raises ValueError: if no items were found, no index can be created.
    :return: Statistics about the indexed files and chunks.
    """
    index_root = display_root or path
    extensions = filter_extensions(extensions, include_docs=include_docs)

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

    chunks = all_chunks

    if all_chunks:
        embeddings = embed_chunks(model, all_chunks)
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(chunk, index_root)) for chunk in all_chunks],
            show_progress=False,
        )
        semantic_index = Vicinity.from_vectors_and_items(embeddings, all_chunks, metric=Metric.COSINE)
    else:
        raise ValueError("Unable to create index.")

    stats = IndexStats(
        indexed_files=indexed_files,
        total_chunks=len(all_chunks),
        languages=language_counts,
    )
    return bm25_index, semantic_index, chunks, stats, index_root
