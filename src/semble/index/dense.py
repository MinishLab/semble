import numpy as np
import numpy.typing as npt
from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel

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
