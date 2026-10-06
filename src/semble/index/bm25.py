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
    """BM25 inverted index supporting incremental document updates."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._pending: dict[str, Counter[str]] = {}  # added documents, applied by set_doc_order
        self._terms: dict[str, int] = {}  # term -> term id, assigned in insertion order
        # Postings sorted by term: term id t's postings are at [_offsets[t], _offsets[t + 1]) in the arrays below.
        self._offsets: npt.NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self._posting_docs: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        self._posting_tfs: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        self._doc_lengths: npt.NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        self.doc_order: list[str] = []
        self._positions: dict[str, int] = {}

    def add_document(self, chunk_id: str, tokens: list[str]) -> None:
        """Index one document, rejecting duplicate IDs."""
        if chunk_id in self._positions or chunk_id in self._pending:
            raise ValueError(f"chunk_id already indexed: {chunk_id}")
        self._pending[chunk_id] = Counter(tokens)

    def remove_document(self, chunk_id: str) -> None:
        """Remove a document's postings; no-op if chunk_id is not indexed."""
        if self._positions.pop(chunk_id, None) is None:
            self._pending.pop(chunk_id, None)

    def set_doc_order(self, chunk_ids: list[str]) -> None:
        """Apply added and removed documents, and set the chunk-list order that get_scores' output is aligned to."""
        target = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        if len(target) != len(chunk_ids) or target.keys() != self._positions.keys() | self._pending.keys():
            raise ValueError("Document order must list every indexed document exactly once")

        # Existing postings, renumbered to their position in chunk_ids. Removed documents get -1 and are dropped.
        renumber = np.full(len(self._doc_lengths), -1, dtype=np.int64)
        renumber[list(self._positions.values())] = [target[chunk_id] for chunk_id in self._positions]
        docs = renumber[self._posting_docs]
        kept = docs >= 0

        # Postings of added documents. Packed arrays keep memory low when a full build adds millions.
        new_docs, new_terms, new_tfs = array("q"), array("q"), array("q")
        for chunk_id, counts in self._pending.items():
            new_docs.extend([target[chunk_id]] * len(counts))
            new_terms.extend(self._terms.setdefault(term, len(self._terms)) for term in counts)
            new_tfs.extend(counts.values())
        self._pending.clear()

        self._set_postings(
            chunk_ids,
            np.concatenate([docs[kept], np.frombuffer(new_docs, dtype=np.int64)]),
            np.concatenate([self._posting_terms()[kept], np.frombuffer(new_terms, dtype=np.int64)]),
            np.concatenate([self._posting_tfs[kept], np.frombuffer(new_tfs, dtype=np.int64)]),
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
            docs, tf = self._posting_docs[start:end], self._posting_tfs[start:end]
            idf = math.log(1 + (corpus_size - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += query_tf * idf * tf / (_K1 * (1 - _B + _B * self._doc_lengths[docs] / avgdl) + tf)

        if weight_mask is not None:
            scores = scores * weight_mask
        return scores

    @classmethod
    def merge(cls, parts: list[tuple[str, BM25]]) -> BM25:
        """Combine indexes into one corpus, prefixing every chunk id with its part's label."""
        merged = cls()
        starts = np.cumsum([0] + [len(part.doc_order) for _, part in parts])
        new_term_ids = [
            np.array([merged._terms.setdefault(term, len(merged._terms)) for term in part._terms], dtype=np.int64)
            for _, part in parts
        ]
        merged._set_postings(
            [f"{label}/{chunk_id}" for label, part in parts for chunk_id in part.doc_order],
            np.concatenate([part._posting_docs + start for (_, part), start in zip(parts, starts)]),
            np.concatenate([ids[part._posting_terms()] for (_, part), ids in zip(parts, new_term_ids)]),
            np.concatenate([part._posting_tfs for _, part in parts]),
        )
        return merged

    def save(self, path: Path) -> None:
        """Persist the index to path/index.json and path/postings.npz."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "index.json").write_bytes(orjson.dumps({"doc_order": self.doc_order, "terms": list(self._terms)}))
        with (path / "postings.npz").open("wb") as f:
            np.savez(
                f, offsets=self._offsets, docs=self._posting_docs, tfs=self._posting_tfs, doc_lengths=self._doc_lengths
            )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        """Load an index from path/index.json and path/postings.npz."""
        data = orjson.loads((path / "index.json").read_bytes())
        with np.load(path / "postings.npz") as arrays:
            offsets, docs, tfs, doc_lengths = (arrays[key] for key in ("offsets", "docs", "tfs", "doc_lengths"))
        doc_order, terms = data["doc_order"], data["terms"]
        if (
            len(set(doc_order)) != len(doc_order)
            or len(doc_lengths) != len(doc_order)
            or len(offsets) != len(terms) + 1
            or offsets[-1] != len(docs)
            or len(tfs) != len(docs)
            or (docs.size and (docs.min() < 0 or docs.max() >= len(doc_order)))
        ):
            raise ValueError("Persisted BM25 document state is inconsistent")
        index = cls()
        index.doc_order = doc_order
        index._positions = {chunk_id: i for i, chunk_id in enumerate(doc_order)}
        index._terms = {term: i for i, term in enumerate(terms)}
        index._offsets, index._posting_docs, index._posting_tfs, index._doc_lengths = offsets, docs, tfs, doc_lengths
        return index

    def _posting_terms(self) -> npt.NDArray[np.int64]:
        """Return the term id of each posting."""
        return np.repeat(np.arange(len(self._offsets) - 1), np.diff(self._offsets))

    def _set_postings(
        self,
        doc_order: list[str],
        docs: npt.NDArray[np.integer],
        terms: npt.NDArray[np.integer],
        tfs: npt.NDArray[np.integer],
    ) -> None:
        """Replace all postings, given one (doc, term, tf) entry per posting in any order; terms must be in _terms."""
        order = np.argsort(terms, kind="stable")
        self.doc_order = doc_order
        self._positions = {chunk_id: i for i, chunk_id in enumerate(doc_order)}
        self._offsets = np.concatenate([[0], np.cumsum(np.bincount(terms, minlength=len(self._terms)))])
        self._posting_docs = docs[order].astype(np.int32)
        self._posting_tfs = tfs[order].astype(np.int32)
        self._doc_lengths = np.bincount(docs, weights=tfs, minlength=len(doc_order)).astype(np.int32)
