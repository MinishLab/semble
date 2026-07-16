from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization


class BM25:
    """BM25 inverted index supporting incremental document updates."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._documents: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._total_doc_length = 0
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_order: list[str] = []
        self._positions: dict[str, int] = {}

    def add_document(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one document, rejecting duplicate IDs."""
        if chunk_id in self._documents:
            raise ValueError(f"chunk_id already indexed: {chunk_id}")
        counts = Counter(tokens)
        self._documents[chunk_id] = counts
        self._doc_lengths[chunk_id] = len(tokens)
        self._total_doc_length += len(tokens)
        for term, count in counts.items():
            self.postings.setdefault(term, {})[chunk_id] = count

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document's postings; no-op if chunk_id is not indexed."""
        counts = self._documents.pop(chunk_id, None)
        if counts is None:
            return
        self._total_doc_length -= self._doc_lengths.pop(chunk_id)
        for term in counts:
            docs = self.postings[term]
            docs.pop(chunk_id, None)
            if not docs:
                del self.postings[term]

    def set_doc_order(self, chunk_ids: list[str]) -> None:
        """Set the current global chunk-list order that get_scores' output is aligned to."""
        self.doc_order = chunk_ids
        self._positions = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Calculate BM25 scores for a tokenized query.

        :param tokens: Tokenized search query.
        :param weight_mask: Optional boolean mask aligned with doc_order.
        :return: Scores aligned with doc_order.
        """
        output_size = len(self.doc_order)
        corpus_size = len(self._documents)
        scores: npt.NDArray[np.float32] = np.zeros(output_size, dtype=np.float32)
        if not tokens or corpus_size == 0:
            return scores

        avgdl = self._total_doc_length / corpus_size
        for term, query_tf in Counter(tokens).items():
            docs = self.postings.get(term)
            if not docs:
                continue
            df = len(docs)
            idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
            for chunk_id, tf in docs.items():
                idx = self._positions.get(chunk_id)
                if idx is None:
                    continue
                dl = self._doc_lengths[chunk_id]
                tfc = tf / (_K1 * (1 - _B + _B * dl / avgdl) + tf)
                scores[idx] += query_tf * idf * tfc

        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    def save(self, path: Path) -> None:
        """Persist the index to path/index.json."""
        path.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self._documents, "doc_order": self.doc_order}
        (path / "index.json").write_bytes(orjson.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "BM25":
        """Load an index from path/index.json, reconstructing its postings."""
        data = orjson.loads((path / "index.json").read_bytes())
        index = cls()
        doc_order = data["doc_order"]
        documents = data["documents"]
        if len(doc_order) != len(set(doc_order)) or set(documents) != set(doc_order):
            raise ValueError("Persisted BM25 document state is inconsistent")
        index._documents = {chunk_id: Counter(counts) for chunk_id, counts in documents.items()}
        for chunk_id, counts in index._documents.items():
            for term, count in counts.items():
                index.postings.setdefault(term, {})[chunk_id] = count
        index._doc_lengths = {chunk_id: sum(counts.values()) for chunk_id, counts in index._documents.items()}
        index._total_doc_length = sum(index._doc_lengths.values())
        index.set_doc_order(doc_order)
        return index
