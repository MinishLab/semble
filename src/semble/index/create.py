import contextlib
from pathlib import Path

import bm25s
from vicinity.backends.basic import BasicArgs

from semble.index.chunker import chunk_source
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import filter_extensions, language_for_path, walk_files
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk, Encoder

_MAX_FILE_BYTES = 1_000_000  # 1 MB max file size to read and index


def create_index_from_path(
    path: Path,
    model: Encoder,
    extensions: frozenset[str] | None = None,
    ignore: frozenset[str] | None = None,
    include_text_files: bool = False,
    display_root: Path | None = None,
) -> tuple[bm25s.BM25, SelectableBasicBackend, list[Chunk]]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param extensions: File extensions to include.
    :param ignore: Directory names to skip.
    :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
    :param display_root: If set, chunk file paths are stored relative to this root.
    :raises ValueError: if no items were found, no index can be created.
    :return: A bm25 index, vicinity index and list of chunks
    """
    extensions = filter_extensions(extensions, include_text_files=include_text_files)

    chunks: list[Chunk] = []

    for file_path in walk_files(path, extensions, ignore):
        language = language_for_path(file_path)
        with contextlib.suppress(OSError):
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
            chunk_path = file_path.relative_to(display_root) if display_root else file_path
            chunks.extend(chunk_source(source, str(chunk_path), language))

    if chunks:
        embeddings = embed_chunks(model, chunks)
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(chunk)) for chunk in chunks],
            show_progress=False,
        )
        args = BasicArgs()
        semantic_index = SelectableBasicBackend(embeddings, args)
    else:
        raise ValueError(f"No supported files found under {path}.")

    return bm25_index, semantic_index, chunks
