import contextlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.types import Chunk


def selector_to_mask(selector: npt.NDArray[np.int_] | None) -> npt.NDArray[np.bool_] | None:
    """Convert a selector array to a boolean mask."""
    if selector is None:
        return None
    mask = np.zeros(len(selector), dtype=bool)
    mask[selector] = True
    return mask


def enrich_for_bm25(chunk: Chunk, root: Path | None) -> str:
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
