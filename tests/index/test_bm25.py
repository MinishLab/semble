import math
from pathlib import Path

import numpy as np
import orjson
import pytest

from semble.index.bm25 import BM25


def _build(docs: dict[str, list[str]]) -> BM25:
    index = BM25()
    for chunk_id, tokens in docs.items():
        index.add_document(chunk_id, tokens)
    index.set_doc_order(list(docs))
    return index


def test_scoring_matches_lucene_formula() -> None:
    """BM25 scores use the Lucene term-frequency formula."""
    index = _build({"a": ["authenticate", "token"], "b": ["login", "password"]})
    scores = index.get_scores(["authenticate"])
    np.testing.assert_allclose(scores[0], math.log(1 + 1.5 / 1.5) / 2.5)
    assert scores[1] == 0


def test_removed_and_unordered_documents_stop_scoring() -> None:
    """Only documents retained in the current order contribute scores."""
    index = _build({"a": ["authenticate"], "b": ["login"]})
    index.remove_document("missing")
    index.set_doc_order(["b"])
    assert np.all(index.get_scores(["authenticate"]) == 0)

    index.remove_document("a")
    index.set_doc_order(["a", "b"])
    assert np.all(index.get_scores(["authenticate"]) == 0)


def test_duplicate_add_document_raises() -> None:
    """Re-adding an already-indexed chunk_id raises, catching caller bugs."""
    index = _build({"a": ["x"]})
    with pytest.raises(ValueError, match="already indexed"):
        index.add_document("a", ["y"])


@pytest.mark.parametrize(
    ("mask", "expected_nonzero"),
    [
        (None, [0, 1]),
        (np.array([True, False]), [0]),
    ],
)
def test_weight_mask_zeroes_masked_docs(mask: np.ndarray | None, expected_nonzero: list[int]) -> None:
    """weight_mask zeroes out scores for masked-out positions, by global chunk order."""
    index = _build({"a": ["shared"], "b": ["shared"]})
    scores = index.get_scores(["shared"], weight_mask=mask)
    nonzero = [i for i, s in enumerate(scores) if s > 0]
    assert nonzero == expected_nonzero


@pytest.mark.parametrize("query", [[], ["zzznonexistent"]])
def test_unmatched_queries_return_all_zero(query: list[str]) -> None:
    """Empty and unknown queries return an all-zero array sized to the corpus."""
    index = _build({"a": ["foo"], "b": ["bar"]})
    scores = index.get_scores(query)
    assert scores.shape == (2,)
    assert np.all(scores == 0)


def test_save_load_preserves_scores_and_doc_order(tmp_path: Path) -> None:
    """save/load roundtrips postings and doc_order, producing identical scores for a fixed query."""
    index = _build({"empty": [], "a": ["authenticate", "token"], "b": ["login", "password"]})
    index.save(tmp_path)

    loaded = BM25.load(tmp_path)
    assert loaded.doc_order == index.doc_order
    np.testing.assert_array_equal(loaded.get_scores(["authenticate"]), index.get_scores(["authenticate"]))


def test_load_rejects_inconsistent_document_order(tmp_path: Path) -> None:
    """Persisted document order must describe the same documents as the postings."""
    index = _build({"a": ["authenticate"]})
    index.save(tmp_path)
    index_path = tmp_path / "index.json"
    data = orjson.loads(index_path.read_bytes())
    data["doc_order"] = ["other"]
    index_path.write_bytes(orjson.dumps(data))

    with pytest.raises(ValueError, match="document state"):
        BM25.load(tmp_path)
