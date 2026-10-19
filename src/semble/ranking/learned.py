"""Learned reranking of the candidate pool: a small gradient-boosted tree ensemble, evaluated with numpy.

The model predicts a log-scale correction to each candidate's boosted score from signals the ranking stage
already computes (list ranks and scores, fusion, boosts, path penalty, file shape). It is trained offline by
``benchmarks/train_ranker.py`` and shipped as ``ranker.json``; no ML library is needed at query time.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import numpy.typing as npt

from semble.ranking.penalties import _file_path_penalty
from semble.types import Chunk

FEATURES: tuple[str, ...] = (
    "final_rank",
    "final_rel",
    "fused_rank",
    "fused_rel",
    "file_boost_rel",
    "stem_boost_rel",
    "definition_boost_rel",
    "path_penalty",
    "is_penalised_path",
    "semantic_rank",
    "semantic_score",
    "semantic_rel",
    "in_semantic",
    "bm25_rank",
    "bm25_rel",
    "in_bm25",
    "in_both",
    "file_chunks_in_pool",
    "file_best_final_rank",
    "is_file_best",
    "path_depth",
)
_MISSING_LIST_RANK = 100
_MISSING_FUSED_RANK = 200


@dataclass(frozen=True)
class RankingSignals:
    """Every intermediate score of one query's ranking stage, as produced by ``semble.search``."""

    alpha: float
    semantic: dict[Chunk, float]  # raw cosine similarity, in rank order
    bm25: dict[Chunk, float]  # raw BM25 score, in rank order
    fused: dict[Chunk, float]  # alpha-weighted RRF fusion
    file_boost: dict[Chunk, float]  # added by boost_multi_chunk_files
    stem_boost: dict[Chunk, float]  # added by the NL path-stem boost
    definition_boost: dict[Chunk, float]  # added by the symbol-definition boosts
    boosted: dict[Chunk, float]  # final score before path penalties and file saturation


def ranking_features(signals: RankingSignals) -> tuple[list[Chunk], npt.NDArray[np.float64]]:
    """Return the candidate pool (best boosted score first) and its feature matrix, columns as in FEATURES."""
    boosted = signals.boosted
    chunks = sorted(boosted, key=lambda c: -boosted[c])
    if not chunks:
        return [], np.empty((0, len(FEATURES)))
    semantic_rank = {c: r for r, c in enumerate(signals.semantic, 1)}
    bm25_rank = {c: r for r, c in enumerate(signals.bm25, 1)}
    semantic_top = next(iter(signals.semantic.values()), 1.0) or 1.0
    bm25_top = next(iter(signals.bm25.values()), 1.0) or 1.0
    fused_rank = {c: r for r, c in enumerate(sorted(signals.fused, key=lambda c: -signals.fused[c]), 1)}
    fused_max = max(signals.fused.values(), default=1.0) or 1.0
    boosted_max = boosted[chunks[0]] or 1.0
    file_ranks: dict[str, list[int]] = {}
    for rank, chunk in enumerate(chunks, 1):
        file_ranks.setdefault(chunk.file_path, []).append(rank)

    rows = []
    for rank, chunk in enumerate(chunks, 1):
        penalty = _file_path_penalty(chunk.file_path)
        ranks_in_file = file_ranks[chunk.file_path]
        rows.append(
            (
                rank,
                boosted[chunk] / boosted_max,
                fused_rank.get(chunk, _MISSING_FUSED_RANK),
                signals.fused.get(chunk, 0.0) / fused_max,
                signals.file_boost.get(chunk, 0.0) / fused_max,
                signals.stem_boost.get(chunk, 0.0) / fused_max,
                signals.definition_boost.get(chunk, 0.0) / fused_max,
                penalty,
                float(penalty < 1.0),
                semantic_rank.get(chunk, _MISSING_LIST_RANK),
                signals.semantic.get(chunk, 0.0),
                signals.semantic.get(chunk, 0.0) / semantic_top,
                float(chunk in semantic_rank),
                bm25_rank.get(chunk, _MISSING_LIST_RANK),
                signals.bm25.get(chunk, 0.0) / bm25_top,
                float(chunk in bm25_rank),
                float(chunk in semantic_rank and chunk in bm25_rank),
                len(ranks_in_file),
                ranks_in_file[0],
                float(ranks_in_file[0] == rank),
                len(Path(chunk.file_path).parts),
            )
        )
    return chunks, np.asarray(rows, dtype=np.float64)


class TreeEnsemble:
    """Sum of regression trees, evaluated for all rows and all trees at once.

    Trees are stored as padded node arrays: an internal node goes left when ``x[feature] <= threshold``;
    a leaf has ``feature == -1`` and contributes ``value``.
    """

    def __init__(
        self,
        features: tuple[str, ...],
        feature: npt.NDArray[np.int64],
        threshold: npt.NDArray[np.float64],
        left: npt.NDArray[np.int64],
        right: npt.NDArray[np.int64],
        value: npt.NDArray[np.float64],
    ) -> None:
        """Create an ensemble from (trees, nodes) arrays."""
        self.features = features
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value
        self.depth = self._max_depth()

    @classmethod
    def from_dict(cls, payload: dict) -> TreeEnsemble:
        """Load the JSON format written by benchmarks/train_ranker.py."""
        trees = payload["trees"]
        width = max(len(t["feature"]) for t in trees)

        def pad(key: str, fill: float, dtype: type) -> np.ndarray:
            return np.array([t[key] + [fill] * (width - len(t[key])) for t in trees], dtype=dtype)

        return cls(
            tuple(payload["features"]),
            pad("feature", -1, np.int64),
            pad("threshold", 0.0, np.float64),
            pad("left", 0, np.int64),
            pad("right", 0, np.int64),
            pad("value", 0.0, np.float64),
        )

    def _max_depth(self) -> int:
        """Longest root-to-leaf path over all trees (the number of steps predict needs)."""
        deepest = 0
        for t in range(len(self.feature)):
            stack = [(0, 0)]
            while stack:
                node, depth = stack.pop()
                if self.feature[t, node] < 0:
                    deepest = max(deepest, depth)
                else:
                    stack.extend(((int(self.left[t, node]), depth + 1), (int(self.right[t, node]), depth + 1)))
        return deepest

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the summed tree output for each row of x."""
        if not len(x):
            return np.zeros(0)
        n_trees = len(self.feature)
        trees = np.broadcast_to(np.arange(n_trees), (len(x), n_trees))
        rows = np.broadcast_to(np.arange(len(x))[:, None], (len(x), n_trees))
        node = np.zeros((len(x), n_trees), dtype=np.int64)
        for _ in range(self.depth):
            feature = self.feature[trees, node]
            internal = feature >= 0
            go_left = x[rows, np.maximum(feature, 0)] <= self.threshold[trees, node]
            child = np.where(go_left, self.left[trees, node], self.right[trees, node])
            node = np.where(internal, child, node)
        return self.value[trees, node].sum(axis=1)


@functools.cache
def load_ranker() -> TreeEnsemble | None:
    """Return the packaged ranker, or None when disabled (SEMBLE_LEARNED_RANKER=0) or not shipped."""
    if os.environ.get("SEMBLE_LEARNED_RANKER", "1") == "0":
        return None
    resource = resources.files("semble.ranking").joinpath("ranker.json")
    if not resource.is_file():
        return None
    ensemble = TreeEnsemble.from_dict(json.loads(resource.read_text()))
    if ensemble.features != FEATURES:
        raise ValueError("ranker.json was trained on a different feature set; retrain it.")
    return ensemble


def apply_learned_ranker(signals: RankingSignals, ranker: TreeEnsemble) -> dict[Chunk, float]:
    """Scale each candidate's boosted score by exp(model correction)."""
    chunks, x = ranking_features(signals)
    correction = ranker.predict(x)
    return {chunk: signals.boosted[chunk] * float(np.exp(c)) for chunk, c in zip(chunks, correction)}
