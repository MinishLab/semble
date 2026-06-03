from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel
from vicinity.backends.basic import CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

from semble.types import Chunk
from semble.utils import resolve_model_name


@cache
def _load_cached(model_path: str) -> StaticModel:
    """Load a model and cache it, but only after the path resolves."""
    # Disable HF progress bars since the model is loaded silently in the background during indexing.
    disable_progress_bars()
    try:
        model = StaticModel.from_pretrained(model_path, force_download=False)
    finally:
        disable_progress_bars()

    return model


def load_model(model_path: str | None = None) -> tuple[StaticModel, str]:
    """Return the current model, loading the default if none was provided."""
    if model_path is None:
        model_path = resolve_model_name()
    model = _load_cached(model_path)
    return model, model_path


def _unique_contents(chunks: list[Chunk]) -> tuple[list[str], list[int]]:
    """Return unique chunk contents and per-chunk positions into that unique list."""
    content_positions: dict[str, int] = {}
    contents: list[str] = []
    positions: list[int] = []
    for chunk in chunks:
        position = content_positions.get(chunk.content)
        if position is None:
            position = len(contents)
            content_positions[chunk.content] = position
            contents.append(chunk.content)
        positions.append(position)
    return contents, positions


def _can_deduplicate_embeddings(model: StaticModel) -> bool:
    """Return whether exact text dedup preserves the model's full-batch output bitwise."""
    model_type = type(model)
    return model_type.__module__ == "model2vec.model" and model_type.__name__ == "StaticModel"


def embed_chunks(
    model: StaticModel,
    chunks: list[Chunk],
    *,
    use_multiprocessing: bool = True,
) -> npt.NDArray[np.float32]:
    """Embed chunks using the configured model."""
    if not chunks:
        return np.empty((0, model.dim), dtype=np.float32)
    if not _can_deduplicate_embeddings(model):
        return np.array(
            model.encode([c.content for c in chunks], use_multiprocessing=use_multiprocessing),
            dtype=np.float32,
        )
    contents, positions = _unique_contents(chunks)
    embeddings = np.array(
        model.encode(contents, use_multiprocessing=use_multiprocessing),
        dtype=np.float32,
    )
    if len(contents) == len(chunks):
        return embeddings
    return embeddings[np.array(positions, dtype=np.intp)]


class SelectableBasicBackend(CosineBasicBackend):
    def _selector_dist(self, x: npt.NDArray, selector: npt.NDArray[np.int_]) -> npt.NDArray:
        """Compute cosine distance."""
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors[selector].T)
        return 1 - sim

    def query(self, vectors: npt.NDArray, k: int, selector: npt.NDArray[np.int_] | None = None) -> QueryResult:
        """Batched distance query.

        :param vectors: The vectors to query.
        :param k: The number of nearest neighbors to return.
        :param selector: Optional array of chunk indices to filter results by.
        :return: A list of tuples with the indices and distances.
        :raises ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")

        out: QueryResult = []
        num_vectors = len(self.vectors)
        effective_k = min(k, num_vectors)
        if selector is not None:
            effective_k = min(effective_k, len(selector))

        # Batch the queries
        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            if selector is not None:
                distances = self._selector_dist(batch, selector)
            else:
                distances = self._dist(batch)

            # Efficiently get the k smallest distances
            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            # Extend the output with tuples of (indices, distances)
            if selector is not None:
                sorted_indices = selector[sorted_indices]
            out.extend(zip(sorted_indices, sorted_distances))

        return out

    def save(self, path: Path) -> None:
        """Save the selectable basic backend."""
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)

    @classmethod
    def load(cls, path: Path) -> "SelectableBasicBackend":
        """Load a selectable basic backend."""
        loaded = super().load(path)
        return SelectableBasicBackend(loaded.vectors, loaded.arguments)
