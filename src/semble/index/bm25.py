from __future__ import annotations

import math
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.index.npz import pack_strings, read_npz, unpack_strings

_K1 = 1.5  # Term-frequency saturation
_B = 0.75  # Document length normalization


class BM25:
    """BM25 inverted index supporting incremental document updates."""

    def __init__(
        self,
        doc_order: list[str],
        terms: list[str],
        offsets: npt.NDArray[np.int64],
        docs: npt.NDArray[np.int32],
        tfs: npt.NDArray[np.int32],
        doc_lengths: npt.NDArray[np.int32],
    ) -> None:
        """Create an index from postings sorted by term: term id t's postings are at [offsets[t], offsets[t + 1])."""
        self._set_state(doc_order, terms, offsets, docs, tfs, doc_lengths)

    @classmethod
    def empty(cls) -> BM25:
        """Create an empty index."""
        no_postings = np.zeros(0, dtype=np.int32)
        return cls([], [], np.zeros(1, dtype=np.int64), no_postings, no_postings, no_postings)

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
        moves = [(doc, target[chunk_id]) for chunk_id, doc in self._positions.items()]
        old_docs, new_docs = np.array(moves, dtype=np.int64).reshape(-1, 2).T
        renumber = np.full(len(self._doc_lengths), -1, dtype=np.int64)
        renumber[old_docs] = new_docs
        docs = renumber[self._posting_docs]
        kept = docs >= 0

        # Postings of added documents.
        added_docs, added_terms, added_tfs = array("q"), array("q"), array("q")
        for chunk_id, counts in self._pending.items():
            added_docs.extend([target[chunk_id]] * len(counts))
            added_terms.extend(self._terms.setdefault(term, len(self._terms)) for term in counts)
            added_tfs.extend(counts.values())

        self._set_state(
            *_sort_postings(
                chunk_ids,
                list(self._terms),
                np.concatenate([docs[kept], np.frombuffer(added_docs, dtype=np.int64)]),
                np.concatenate([self._posting_terms()[kept], np.frombuffer(added_terms, dtype=np.int64)]),
                np.concatenate([self._posting_tfs[kept], np.frombuffer(added_tfs, dtype=np.int64)]),
            )
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
        terms: dict[str, int] = {}
        new_term_ids = [
            np.array([terms.setdefault(term, len(terms)) for term in part._terms], dtype=np.int64) for _, part in parts
        ]
        starts = np.cumsum([0] + [len(part.doc_order) for _, part in parts])
        return cls(
            *_sort_postings(
                [f"{label}/{chunk_id}" for label, part in parts for chunk_id in part.doc_order],
                list(terms),
                np.concatenate([part._posting_docs + start for (_, part), start in zip(parts, starts)]),
                np.concatenate([ids[part._posting_terms()] for (_, part), ids in zip(parts, new_term_ids)]),
                np.concatenate([part._posting_tfs for _, part in parts]),
            )
        )

    def save(self, path: Path) -> None:
        """Persist the index to path/index.npz."""
        path.mkdir(parents=True, exist_ok=True)
        with (path / "index.npz").open("wb") as f:
            np.savez(
                f,
                doc_order=pack_strings(self.doc_order),
                terms=pack_strings(self._terms),
                offsets=self._offsets,
                docs=self._posting_docs,
                tfs=self._posting_tfs,
                doc_lengths=self._doc_lengths,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        """Load an index from path/index.npz."""
        arrays = read_npz(path / "index.npz")
        doc_order, terms = unpack_strings(arrays["doc_order"]), unpack_strings(arrays["terms"])
        offsets, docs, tfs, doc_lengths = (arrays[key] for key in ("offsets", "docs", "tfs", "doc_lengths"))
        _check_consistent(doc_order, terms, offsets, docs, tfs, doc_lengths)
        return cls(doc_order, terms, offsets, docs, tfs, doc_lengths)

    def _set_state(
        self,
        doc_order: list[str],
        terms: list[str],
        offsets: npt.NDArray[np.int64],
        docs: npt.NDArray[np.int32],
        tfs: npt.NDArray[np.int32],
        doc_lengths: npt.NDArray[np.int32],
    ) -> None:
        """Replace the whole index with the given postings and clear pending changes."""
        self.doc_order = doc_order
        self._positions = {chunk_id: i for i, chunk_id in enumerate(doc_order)}
        self._terms = {term: i for i, term in enumerate(terms)}  # term -> term id
        self._offsets = offsets
        self._posting_docs = docs
        self._posting_tfs = tfs
        self._doc_lengths = doc_lengths
        self._pending: dict[str, Counter[str]] = {}  # added documents, applied by set_doc_order

    def _posting_terms(self) -> npt.NDArray[np.int64]:
        """Return the term id of each posting."""
        return np.repeat(np.arange(len(self._offsets) - 1), np.diff(self._offsets))


def _sort_postings(
    doc_order: list[str],
    terms: list[str],
    docs: npt.NDArray[np.integer],
    term_ids: npt.NDArray[np.integer],
    tfs: npt.NDArray[np.integer],
) -> tuple[
    list[str], list[str], npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    """Build BM25 constructor arguments from one (doc, term id, tf) entry per posting, given in any order."""
    term_counts = np.bincount(term_ids, minlength=len(terms))
    used = term_counts > 0
    if not used.all():
        # Drop terms that no document contains any more, keeping the remaining ids in order.
        terms = [term for term, keep in zip(terms, used) if keep]
        term_ids = (np.cumsum(used) - 1)[term_ids]
        term_counts = term_counts[used]
    order = np.argsort(term_ids, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(term_counts)]).astype(np.int64)
    doc_lengths = np.bincount(docs, weights=tfs, minlength=len(doc_order)).astype(np.int32)
    return doc_order, terms, offsets, docs[order].astype(np.int32), tfs[order].astype(np.int32), doc_lengths


def _check_consistent(
    doc_order: list[str],
    terms: list[str],
    offsets: npt.NDArray[np.integer],
    docs: npt.NDArray[np.integer],
    tfs: npt.NDArray[np.integer],
    doc_lengths: npt.NDArray[np.integer],
) -> None:
    """Raise ValueError unless loaded postings and document order describe the same documents and terms."""
    if (
        len(set(doc_order)) != len(doc_order)
        or len(doc_lengths) != len(doc_order)
        or len(offsets) != len(terms) + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != len(docs)
        or len(tfs) != len(docs)
        or (docs.size and (docs.min() < 0 or docs.max() >= len(doc_order)))
    ):
        raise ValueError("Persisted BM25 document state is inconsistent")
