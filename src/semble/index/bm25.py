from __future__ import annotations

import math
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt
import orjson

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization


class BM25:
    """BM25 inverted index stored as term-sorted posting arrays, supporting incremental document updates."""

    def __init__(self) -> None:
        """Create an empty index."""
        self.doc_order: list[str] = []
        self._doc_index: dict[str, int] = {}  # chunk id -> position in doc_order
        self._pending: dict[str, Counter[str]] = {}  # added documents, applied by set_doc_order
        self._terms: dict[str, int] = {}  # term -> term id, in id order
        self._offsets: npt.NDArray[np.int64] = np.zeros(1, dtype=np.int64)  # term id -> slice of the posting arrays
        self._post_docs: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        self._post_tfs: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        self._doc_lengths: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)

    def add_document(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one document, rejecting duplicate IDs. Takes effect on the next set_doc_order."""
        if chunk_id in self._doc_index or chunk_id in self._pending:
            raise ValueError(f"chunk_id already indexed: {chunk_id}")
        self._pending[chunk_id] = Counter(tokens)

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document; no-op if chunk_id is not indexed. Takes effect on the next set_doc_order."""
        if self._doc_index.pop(chunk_id, None) is None:
            self._pending.pop(chunk_id, None)

    def set_doc_order(self, chunk_ids: list[str]) -> None:
        """Apply pending changes and order documents as in chunk_ids, which get_scores' output is aligned to."""
        position = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        if len(position) != len(chunk_ids) or position.keys() != self._doc_index.keys() | self._pending.keys():
            raise ValueError("Document order must list every indexed document exactly once")
        # Removed documents map to -1, so their postings are dropped.
        renumber = np.full(len(self._doc_lengths), -1, dtype=np.int64)
        renumber[list(self._doc_index.values())] = [position[chunk_id] for chunk_id in self._doc_index]
        alive = renumber >= 0
        entry_docs = renumber[self._post_docs]
        kept = entry_docs >= 0
        lengths = np.zeros(len(chunk_ids), dtype=np.int64)
        lengths[renumber[alive]] = self._doc_lengths[alive]

        # Packed arrays rather than lists keep peak memory low when a full build adds millions of postings.
        new_terms, new_tfs, terms_per_doc = array("q"), array("q"), array("q")
        for chunk_id, counts in self._pending.items():
            new_terms.extend(self._terms.setdefault(term, len(self._terms)) for term in counts)
            new_tfs.extend(counts.values())
            terms_per_doc.append(len(counts))
            lengths[position[chunk_id]] = counts.total()
        new_positions = np.array([position[chunk_id] for chunk_id in self._pending], dtype=np.int64)
        new_docs = np.repeat(new_positions, terms_per_doc)
        self._pending.clear()

        entry_terms = np.repeat(np.arange(len(self._offsets) - 1), np.diff(self._offsets))
        self._set_postings(
            chunk_ids,
            lengths,
            np.concatenate([entry_docs[kept], new_docs]),
            np.concatenate([entry_terms[kept], np.frombuffer(new_terms, dtype=np.int64)]),
            np.concatenate([self._post_tfs[kept], np.frombuffer(new_tfs, dtype=np.int64)]),
        )

    def get_scores(
        self, tokens: list[str], weight_mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.float32]:
        """Calculate BM25 scores for a tokenized query.

        :param tokens: Tokenized search query.
        :param weight_mask: Optional boolean mask aligned with doc_order.
        :return: Scores aligned with doc_order.
        """
        corpus_size = len(self._doc_lengths)
        scores: npt.NDArray[np.float32] = np.zeros(corpus_size, dtype=np.float32)
        if not tokens or corpus_size == 0:
            return scores

        avgdl = self._doc_lengths.mean()
        for term, query_tf in Counter(tokens).items():
            term_id = self._terms.get(term)
            if term_id is None:
                continue
            start, end = self._offsets[term_id], self._offsets[term_id + 1]
            docs, tf = self._post_docs[start:end], self._post_tfs[start:end]
            idf = math.log(1 + (corpus_size - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += query_tf * idf * tf / (_K1 * (1 - _B + _B * self._doc_lengths[docs] / avgdl) + tf)

        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    @classmethod
    def merge(cls, parts: list[tuple[str, BM25]]) -> BM25:
        """Combine indexes into one corpus, prefixing every chunk id with its part's label."""
        merged = cls()
        indexes = [part for _, part in parts]
        starts = np.cumsum([0] + [len(part.doc_order) for part in indexes])
        remaps = [
            np.array([merged._terms.setdefault(term, len(merged._terms)) for term in part._terms], dtype=np.int64)
            for part in indexes
        ]
        merged._set_postings(
            [f"{label}/{chunk_id}" for label, part in parts for chunk_id in part.doc_order],
            np.concatenate([part._doc_lengths for part in indexes]),
            np.concatenate([part._post_docs + start for part, start in zip(indexes, starts)]),
            np.concatenate([np.repeat(remap, np.diff(part._offsets)) for part, remap in zip(indexes, remaps)]),
            np.concatenate([part._post_tfs for part in indexes]),
        )
        return merged

    def save(self, path: Path) -> None:
        """Persist the index to path/index.json (doc order and terms) and path/postings.npz (posting arrays)."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "index.json").write_bytes(orjson.dumps({"doc_order": self.doc_order, "terms": list(self._terms)}))
        with (path / "postings.npz").open("wb") as f:
            np.savez(f, offsets=self._offsets, docs=self._post_docs, tfs=self._post_tfs, doc_lengths=self._doc_lengths)

    @classmethod
    def load(cls, path: Path) -> BM25:
        """Load an index from path/index.json and path/postings.npz."""
        data = orjson.loads((path / "index.json").read_bytes())
        with np.load(path / "postings.npz") as arrays:
            offsets, docs, tfs, doc_lengths = (arrays[key] for key in ("offsets", "docs", "tfs", "doc_lengths"))
        doc_order, terms = data["doc_order"], data["terms"]
        if (
            len(set(doc_order)) != len(doc_order)
            or len(doc_lengths) != len(doc_order)
            or len(offsets) != len(terms) + 1
            or not offsets[-1] == len(docs) == len(tfs)
            or (len(docs) and docs.max() >= len(doc_order))
        ):
            raise ValueError("Persisted BM25 document state is inconsistent")
        index = cls()
        index.doc_order = doc_order
        index._doc_index = {chunk_id: i for i, chunk_id in enumerate(doc_order)}
        index._terms = {term: i for i, term in enumerate(terms)}
        index._offsets, index._post_docs, index._post_tfs, index._doc_lengths = offsets, docs, tfs, doc_lengths
        return index

    def _set_postings(
        self,
        doc_order: list[str],
        doc_lengths: npt.NDArray[np.integer],
        docs: npt.NDArray[np.integer],
        terms: npt.NDArray[np.integer],
        tfs: npt.NDArray[np.integer],
    ) -> None:
        """Replace all documents and postings, given one (doc, term, tf) entry per posting in any order."""
        order = np.argsort(terms, kind="stable")
        self.doc_order = doc_order
        self._doc_index = {chunk_id: i for i, chunk_id in enumerate(doc_order)}
        self._offsets = np.concatenate([[0], np.cumsum(np.bincount(terms, minlength=len(self._terms)))])
        self._post_docs = docs[order].astype(np.int32)
        self._post_tfs = tfs[order].astype(np.int32)
        self._doc_lengths = doc_lengths.astype(np.int32)
