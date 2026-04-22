import pytest

from semble.ranking.boosting import apply_query_boost, boost_multi_chunk_files, resolve_alpha
from semble.ranking.penalties import rerank_topk
from tests.conftest import make_chunk


def test_rerank_topk_penalise_paths_false_respects_scores() -> None:
    """penalise_paths=False leaves score order intact, including __init__.py."""
    init_chunk = make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = make_chunk("def authenticate(token): ...", "src/semble/auth.py")
    ranked = rerank_topk({init_chunk: 2.0, impl_chunk: 1.0}, top_k=2, penalise_paths=False)
    assert ranked[0][0] == init_chunk


def test_rerank_topk_saturation_decay_preserves_order() -> None:
    """Chunks beyond the saturation threshold get decay but results stay score-ordered."""
    chunks = [make_chunk(f"def fn_{i}(): pass", "big_file.py") for i in range(5)]
    ranked = rerank_topk({c: float(5 - i) for i, c in enumerate(chunks)}, top_k=5)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_topk_empty_returns_empty_list() -> None:
    """rerank_topk with an empty scores dict returns an empty list."""
    assert rerank_topk({}, top_k=5) == []


@pytest.mark.parametrize(
    "penalised_path",
    [
        "src/semble/__init__.py",  # _REEXPORT_FILENAMES
        "tests/test_auth.py",  # _TEST_FILE_RE / _TEST_DIR_RE
        "src/compat/old_api.py",  # _COMPAT_DIR_RE
        "examples/demo.py",  # _EXAMPLES_DIR_RE
        "src/types/index.d.ts",  # _TYPE_DEFS_RE
    ],
)
def test_rerank_topk_demotes_penalised_paths(penalised_path: str) -> None:
    """Files matching each penalty pattern rank below an equal-scored regular file."""
    regular = make_chunk("def impl(): pass", "src/regular.py")
    penalised = make_chunk("def impl(): pass", penalised_path)
    ranked = rerank_topk({regular: 1.0, penalised: 1.0}, top_k=2)
    assert ranked[0][0] == regular


@pytest.mark.parametrize(
    ("query", "alpha_in", "expected"),
    [
        ("MyService", 0.7, 0.7),  # explicit value returned as-is
        ("MyService", None, 0.3),  # symbol query → _ALPHA_SYMBOL
        ("how does routing work", None, 0.5),  # NL query → _ALPHA_NL
    ],
)
def test_resolve_alpha(query: str, alpha_in: float | None, expected: float) -> None:
    """resolve_alpha returns explicit alpha or auto-detects from symbol/NL query type."""
    assert resolve_alpha(query, alpha_in) == pytest.approx(expected)


@pytest.mark.parametrize(
    "query",
    [
        "MyService",  # bare symbol query
        "how does MyService work",  # NL query with embedded symbol
    ],
)
def test_apply_query_boost_boosts_defining_chunk(query: str) -> None:
    """Symbol and NL-with-symbol queries both boost chunks that define the symbol."""
    defining = make_chunk("class MyService:\n    pass", "src/my_service.py")
    other = make_chunk("x = MyService()", "src/utils.py")
    scores: dict = {defining: 0.5, other: 0.4}

    boosted = apply_query_boost(scores, query, [defining, other])

    assert boosted[defining] > boosted[other]


@pytest.mark.parametrize(
    "query",
    [
        "MyService",
        "how does MyService work",
    ],
)
def test_apply_query_boost_scans_non_candidates(query: str) -> None:
    """Non-candidate chunks on stem-matched files get boosted when defining the symbol."""
    defining = make_chunk("class MyService:\n    pass", "src/myservice.py")
    candidate = make_chunk("x = 1", "src/other.py")
    scores: dict = {candidate: 0.5}

    boosted = apply_query_boost(scores, query, [defining, candidate])

    assert defining in boosted
    assert boosted[defining] > 0


@pytest.mark.parametrize(
    "query",
    [
        "UserService",  # bare symbol query
        "how does UserService work",  # NL with embedded symbol
    ],
)
def test_apply_query_boost_skips_non_matching_stem(query: str) -> None:
    """Non-candidate chunk with an unrelated stem is not boosted, regardless of query style."""
    defining = make_chunk("class UserService:\n    pass", "src/user_service.py")
    unrelated = make_chunk("x = 1", "src/totally_unrelated_name.py")
    scores: dict = {defining: 0.5}
    boosted = apply_query_boost(scores, query, [defining, unrelated])
    assert unrelated not in boosted


@pytest.mark.parametrize(
    ("query", "file_path"),
    [
        ("authenticate user session", "src/auth.py"),  # prefix / morphological match
        ("auth service", "src/auth_service.py"),  # every keyword exact-matches a stem part
    ],
)
def test_apply_query_boost_nl_stem_match_boosts(query: str, file_path: str) -> None:
    """NL query keywords matching file-stem parts boost the chunk above its baseline score."""
    chunk = make_chunk("def authenticate(): pass", file_path)
    scores: dict = {chunk: 0.5}
    boosted = apply_query_boost(scores, query, [chunk])
    assert boosted[chunk] > 0.5


def test_apply_query_boost_all_stopwords_is_noop() -> None:
    """NL query composed entirely of stopwords performs no stem boost."""
    chunk = make_chunk("def foo(): pass", "src/auth.py")
    scores: dict = {chunk: 0.5}
    boosted = apply_query_boost(scores, "the and or", [chunk])
    assert boosted[chunk] == pytest.approx(0.5)


def test_apply_query_boost_namespace_qualified_adds_full_name() -> None:
    """Namespace-qualified symbol adds both the leaf and the full query to the names set."""
    defining = make_chunk("class Base:\n    pass", "src/base.py")
    scores: dict = {defining: 0.5}
    boosted = apply_query_boost(scores, "Sinatra::Base", [defining])
    assert boosted[defining] > 0.5


def test_apply_query_boost_empty_scores_returns_empty() -> None:
    """Empty scores dict returns empty dict without error."""
    assert apply_query_boost({}, "SomeQuery", []) == {}


def test_boost_multi_chunk_files_promotes_top_chunk() -> None:
    """File with multiple chunks gets its top chunk boosted."""
    c1 = make_chunk("def a(): pass", "src/big.py")
    c2 = make_chunk("def b(): pass", "src/big.py")
    c3 = make_chunk("def c(): pass", "src/small.py")
    scores: dict = {c1: 1.0, c2: 0.8, c3: 1.0}
    boost_multi_chunk_files(scores)
    assert scores[c1] > 1.0


@pytest.mark.parametrize(
    "scores_in",
    [
        {},  # empty
        {"__zero__": 0.0},  # all-zero max (division-by-zero guard)
    ],
)
def test_boost_multi_chunk_files_noop_cases(scores_in: dict) -> None:
    """Empty dict and all-zero scores are both no-ops."""
    if "__zero__" in scores_in:
        chunk = make_chunk("x = 1", "src/foo.py")
        scores: dict = {chunk: 0.0}
    else:
        scores = {}
    snapshot = dict(scores)
    boost_multi_chunk_files(scores)
    assert scores == snapshot
