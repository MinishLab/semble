import contextlib
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.types import Chunk

# Matches definition keywords followed by an optional namespace prefix and an identifier.
# Used to extract defined names for BM25 enrichment so that symbol queries can find
# chunks that define a name even when BM25 tokenisation would otherwise miss it.
_DEF_NAME_RE = re.compile(
    r"(?:^|(?<=\s))(?:class|def|func|function|fn|interface|struct|enum|trait|type|object"
    r"|module|defmodule|record|protocol|typedef)\s+"
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def selector_to_mask(selector: npt.NDArray[np.int_] | None) -> npt.NDArray[np.bool_] | None:
    """Convert a selector array to a boolean mask."""
    if selector is None:
        return None
    mask = np.zeros(len(selector), dtype=bool)
    mask[selector] = True
    return mask


def enrich_for_bm25(chunk: Chunk, root: Path | None) -> str:
    """Append file path components and defined symbol names to BM25 content.

    Uses a repo-relative path so that machine-specific directory components
    (usernames, workspace names, temp dirs) are never indexed as tokens.

    Also extracts names defined in the chunk (classes, functions, etc.) and
    appends them as extra tokens so BM25 can match queries that reference those
    names directly.
    """
    path = Path(chunk.file_path)
    if root is not None:
        with contextlib.suppress(ValueError):
            path = path.relative_to(root)
    stem = path.stem
    # Collect directory names from the (now relative) path, skipping filesystem roots.
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])  # Last 3 repo-relative directory components

    # Extract defined names (class/function/etc.) to boost symbol BM25 matching.
    def_names = " ".join(_DEF_NAME_RE.findall(chunk.content))

    # Repeat the stem twice to up-weight file-path matches in BM25.
    return f"{chunk.content} {stem} {stem} {dir_text} {def_names}"
