import pytest

from semble.ranking.boosting import apply_query_boost, boost_multi_chunk_files, resolve_alpha
from semble.ranking.penalties import rerank_topk
from tests.conftest import make_chunk


def test_rerank_topk() -> None:
    """rerank_topk: empty → []; penalise_paths=False respects raw scores; saturation decay keeps order."""
    assert rerank_topk({}, top_k=5) == []

    init_chunk = make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = make_chunk("def authenticate(token): ...", "src/semble/auth.py")
    ranked = rerank_topk({init_chunk: 2.0, impl_chunk: 1.0}, top_k=2, penalise_paths=False)
    assert ranked[0][0] == init_chunk

    saturated = [make_chunk(f"def fn_{i}(): pass", "big_file.py") for i in range(5)]
    ranked_sat = rerank_topk({c: float(5 - i) for i, c in enumerate(saturated)}, top_k=5)
    scores = [s for _, s in ranked_sat]
    assert scores == sorted(scores, reverse=True)


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


def test_apply_query_boost_edge_cases() -> None:
    """apply_query_boost: stopwords → noop; namespace-qualified → boosts leaf; empty scores → {}."""
    chunk = make_chunk("def foo(): pass", "src/auth.py")
    assert apply_query_boost({chunk: 0.5}, "the and or", [chunk])[chunk] == pytest.approx(0.5)

    defining = make_chunk("class Base:\n    pass", "src/base.py")
    assert apply_query_boost({defining: 0.5}, "Sinatra::Base", [defining])[defining] > 0.5

    assert apply_query_boost({}, "SomeQuery", []) == {}


def test_boost_multi_chunk_files() -> None:
    """boost_multi_chunk_files: no-op on empty / all-zero; promotes top chunk of a multi-chunk file."""
    empty: dict = {}
    boost_multi_chunk_files(empty)
    assert empty == {}

    zero_chunk = make_chunk("x = 1", "src/foo.py")
    all_zero: dict = {zero_chunk: 0.0}
    boost_multi_chunk_files(all_zero)
    assert all_zero[zero_chunk] == 0.0

    c1 = make_chunk("def a(): pass", "src/big.py")
    c2 = make_chunk("def b(): pass", "src/big.py")
    c3 = make_chunk("def c(): pass", "src/small.py")
    scores: dict = {c1: 1.0, c2: 0.8, c3: 1.0}
    boost_multi_chunk_files(scores)
    assert scores[c1] > 1.0
