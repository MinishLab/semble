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


def test_removed_documents_stop_scoring() -> None:
    """Removed documents stop scoring, and the new order must list exactly the remaining documents."""
    index = _build({"a": ["authenticate"], "b": ["login"]})
    index.remove_document("missing")
    index.remove_document("a")
    with pytest.raises(ValueError, match="exactly once"):
        index.set_doc_order(["a", "b"])
    index.set_doc_order(["b"])
    assert np.all(index.get_scores(["authenticate"]) == 0)

    # Re-adding an id removed before set_doc_order replaces its postings, as incremental reindexing does.
    index.remove_document("b")
    index.add_document("b", ["authenticate"])
    index.set_doc_order(["b"])
    assert index.get_scores(["login"])[0] == 0
    assert index.get_scores(["authenticate"])[0] > 0


def test_merge_matches_single_corpus() -> None:
    """Merging two indexes scores identically to one index built over all documents with prefixed ids."""
    single = _build({"l/a": ["invoice", "client"], "l/b": ["invoice", "endpoint"], "r/c": ["config", "host"]})
    left = _build({"a": ["invoice", "client"], "b": ["invoice", "endpoint"]})
    right = _build({"c": ["config", "host"]})
    merged = BM25.merge([("l", left), ("r", right)])
    assert merged.doc_order == single.doc_order
    np.testing.assert_allclose(merged.get_scores(["invoice", "host"]), single.get_scores(["invoice", "host"]))


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


@pytest.mark.parametrize("corrupt", ["unknown_document", "negative_posting"])
def test_load_rejects_inconsistent_state(corrupt: str, tmp_path: Path) -> None:
    """Persisted document order and posting indices must describe the same documents."""
    index = _build({"a": ["authenticate"]})
    index.save(tmp_path)
    if corrupt == "unknown_document":
        data = orjson.loads((tmp_path / "index.json").read_bytes())
        data["doc_order"] = ["a", "other"]
        (tmp_path / "index.json").write_bytes(orjson.dumps(data))
    else:
        with np.load(tmp_path / "postings.npz") as arrays:
            postings = dict(arrays)
        postings["docs"] = np.array([-1], dtype=np.int32)
        np.savez(tmp_path / "postings.npz", **postings)

    with pytest.raises(ValueError, match="document state"):
        BM25.load(tmp_path)
