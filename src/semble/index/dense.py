import numpy as np
import numpy.typing as npt
from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel
from vicinity.backends.basic import CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

from semble.types import Chunk, Encoder

_DEFAULT_MODEL_NAME = "minishlab/potion-code-16M"


def load_model(model_path: str | None = None) -> Encoder:
    """Return the current model, loading the default if none was provided."""
    if model_path is None:
        model_path = _DEFAULT_MODEL_NAME
    # Disable HF progress bars since the model is loaded silently in the background during indexing.
    disable_progress_bars()
    try:
        model = StaticModel.from_pretrained(model_path)
    finally:
        disable_progress_bars()
    return model


def embed_chunks(model: Encoder, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Embed chunks using the configured model."""
    if not chunks:
        return np.empty((0, 256), dtype=np.float32)
    return np.array(model.encode([c.content for c in chunks]), dtype=np.float32)


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
