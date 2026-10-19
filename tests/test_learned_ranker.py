import numpy as np
import pytest

from semble.ranking import learned
from semble.ranking.learned import FEATURES, RankingSignals, TreeEnsemble, apply_learned_ranker, ranking_features
from tests.conftest import make_chunk

# Two trees over a 2-feature input: tree 0 splits on x0 <= 0.5, tree 1 is a single leaf.
_TOY = {
    "features": ["a", "b"],
    "trees": [
        {
            "feature": [0, -1, -1],
            "threshold": [0.5, 0.0, 0.0],
            "left": [1, 0, 0],
            "right": [2, 0, 0],
            "value": [0.0, -1.0, 2.0],
        },
        {"feature": [-1], "threshold": [0.0], "left": [0], "right": [0], "value": [0.25]},
    ],
}


def _signals() -> RankingSignals:
    a = make_chunk("def parse(): ...", "src/parser.py")
    b = make_chunk("def parse_all(): ...", "src/parser.py")
    c = make_chunk("def load(): ...", "tests/test_loader.py")
    return RankingSignals(
        alpha=0.5,
        semantic={a: 0.9, c: 0.4},
        bm25={b: 7.0, a: 5.0},
        fused={a: 0.02, b: 0.01, c: 0.005},
        file_boost={a: 0.004},
        stem_boost={},
        definition_boost={b: 0.03},
        boosted={a: 0.024, b: 0.04, c: 0.005},
    )


def test_tree_ensemble_predict() -> None:
    """Rows follow x <= threshold to the left; outputs sum over trees."""
    ensemble = TreeEnsemble.from_dict(_TOY)
    x = np.array([[0.2, 9.0], [0.5, 0.0], [0.7, 0.0]])
    assert ensemble.predict(x) == pytest.approx([-0.75, -0.75, 2.25])
    assert ensemble.predict(np.empty((0, 2))).shape == (0,)


def test_ranking_features() -> None:
    """The pool is ordered by boosted score and every column of FEATURES is filled."""
    signals = _signals()
    pool, x = ranking_features(signals)
    assert [c.content for c in pool] == ["def parse_all(): ...", "def parse(): ...", "def load(): ..."]
    assert x.shape == (3, len(FEATURES))
    row = dict(zip(FEATURES, x[0]))
    assert row["final_rank"] == 1 and row["final_rel"] == 1.0
    assert row["in_bm25"] == 1.0 and row["in_semantic"] == 0.0 and row["semantic_rank"] == 100
    assert row["file_chunks_in_pool"] == 2 and row["is_file_best"] == 1.0
    assert dict(zip(FEATURES, x[1]))["is_file_best"] == 0.0
    assert dict(zip(FEATURES, x[2]))["is_penalised_path"] == 1.0
    assert ranking_features(RankingSignals(0.5, {}, {}, {}, {}, {}, {}, {})) == ([], pytest.approx(np.empty((0, 21))))


def test_apply_learned_ranker_scales_boosted_scores() -> None:
    """Each candidate's boosted score is multiplied by exp(correction); the candidate set is unchanged."""
    signals = _signals()
    ensemble = TreeEnsemble.from_dict({**_TOY, "features": list(FEATURES)})
    scores = apply_learned_ranker(signals, ensemble)
    assert set(scores) == set(signals.boosted)
    pool, x = ranking_features(signals)
    for chunk, correction in zip(pool, ensemble.predict(x)):
        assert scores[chunk] == pytest.approx(signals.boosted[chunk] * np.exp(correction))


def test_packaged_ranker_matches_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped ranker.json matches FEATURES; SEMBLE_LEARNED_RANKER=0 disables it."""
    learned.load_ranker.cache_clear()
    ranker = learned.load_ranker()
    assert ranker is not None and ranker.features == FEATURES
    learned.load_ranker.cache_clear()
    monkeypatch.setenv("SEMBLE_LEARNED_RANKER", "0")
    assert learned.load_ranker() is None
    learned.load_ranker.cache_clear()
