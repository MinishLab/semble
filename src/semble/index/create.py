import contextlib
from collections.abc import Sequence
from pathlib import Path

import bm25s
from model2vec.model import StaticModel
from vicinity.backends.basic import BasicArgs

from semble.chunking import chunk_source
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import walk_files
from semble.index.files import FileStatus, detect_language, get_extensions, get_file_status, read_file_text
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk, ContentType


def create_index_from_path(
    path: Path,
    model: StaticModel,
    content: ContentType | Sequence[ContentType] = (ContentType.CODE,),
    display_root: Path | None = None,
) -> tuple[bm25s.BM25, SelectableBasicBackend, list[Chunk]]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param content: Content types to index.
    :param display_root: If set, chunk file paths are stored relative to this root.
    :raises ValueError: if no items were found, no index can be created.
    :return: A bm25 index, vicinity index and list of chunks
    """
    chunks: list[Chunk] = []
    normalized = (content,) if isinstance(content, ContentType) else content
    resolved_extensions = get_extensions(normalized)
    for file_path in walk_files(path, resolved_extensions):
        language = detect_language(file_path)
        with contextlib.suppress(OSError):
            file_status = get_file_status(file_path, None)
            if file_status != FileStatus.VALID:
                continue
            source = read_file_text(file_path)
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
