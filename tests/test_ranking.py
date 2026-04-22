import pytest

from semble.ranking.boosting import (
    _boost_symbol_definitions,
    _chunk_defines_symbol,
    _count_keyword_matches,
    _extract_symbol_name,
    _is_symbol_query,
    _stem_matches,
    apply_query_boost,
    boost_multi_chunk_files,
    resolve_alpha,
)
from semble.ranking.penalties import _file_path_penalty, rerank_topk
from tests.conftest import make_chunk


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("HTTPAdapter", True),
        ("field_validator", True),
        ("URL", True),
        ("getUser", True),
        ("Sinatra::Base", True),
        ("_private", True),
        ("__init__", True),
        ("session", False),
        ("response", False),
        ("how does routing work", False),
    ],
)
def test_is_symbol_query(query: str, expected: bool) -> None:
    """Identifiers with uppercase/underscore/separator are symbols; plain lowercase words are not."""
    assert _is_symbol_query(query) is expected


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("src/auth.py", 1.0),
        ("src/semble/__init__.py", 0.5),
        ("tests/test_auth.py", 0.3),
        ("src/compat/old_api.py", 0.3),
        ("examples/demo.py", 0.3),
        ("src/types/index.d.ts", 0.7),
    ],
)
def test_file_path_penalty(file_path: str, expected: float) -> None:
    """Path penalties are applied correctly per file type."""
    assert _file_path_penalty(file_path) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("content", "symbol", "expected"),
    [
        ("class UserService:\n    pass", "UserService", True),
        ("def authenticate(token):\n    return token", "authenticate", True),
        ("struct Config {\n    host: String,\n}", "Config", True),
        ("CREATE TABLE users (\n  id INT\n);", "users", True),
        ("x = UserService()\n", "UserService", False),
        ("return Config(host='localhost')", "Config", False),
    ],
)
def test_chunk_defines_symbol(content: str, symbol: str, expected: bool) -> None:
    """Definition keyword + symbol name matches; bare usage does not."""
    assert _chunk_defines_symbol(make_chunk(content), symbol) is expected


def test_rerank_topk_init_demoted_by_default() -> None:
    """__init__.py is demoted below an equal-scored regular file."""
    init_chunk = make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = make_chunk("def authenticate(token): ...", "src/semble/auth.py")
    ranked = rerank_topk({init_chunk: 1.0, impl_chunk: 1.0}, top_k=2)
    assert ranked[0][0] == impl_chunk


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
    assert len(ranked) == 5
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Client", "Client"),
        ("Sinatra::Base", "Base"),
        ("a\\b", "b"),
        ("a->b", "b"),
        ("a.b.c", "c"),
    ],
)
def test_extract_symbol_name(query: str, expected: str) -> None:
    """Extracts the final identifier from namespace-qualified or plain symbols."""
    assert _extract_symbol_name(query) == expected


@pytest.mark.parametrize(
    ("stem", "name", "expected"),
    [
        ("auth", "auth", True),
        ("auth_service", "authservice", True),  # snake_case normalisation
        ("users", "user", True),  # plural stripping
        ("auth_services", "authservice", True),  # both transformations
        ("utils", "config", False),
    ],
)
def test_stem_matches(stem: str, name: str, expected: bool) -> None:
    """Stem matching handles exact, snake_case-normalised, and plural forms."""
    assert _stem_matches(stem, name) is expected


@pytest.mark.parametrize(
    ("keywords", "parts", "expected"),
    [
        ({"auth", "service"}, {"auth", "service", "utils"}, 2),  # exact
        ({"dep"}, {"dependencies"}, 1),  # prefix (3+ chars)
        ({"foo"}, {"bar", "baz"}, 0),  # no match
        ({"ab"}, {"abcdef"}, 0),  # prefix too short
    ],
)
def test_count_keyword_matches(keywords: set[str], parts: set[str], expected: int) -> None:
    """Keyword matching handles exact, prefix, and short-prefix cases."""
    assert _count_keyword_matches(keywords, parts) == expected


def test_apply_query_boost_embedded_symbol_boosts_defining_chunk() -> None:
    """NL query with CamelCase symbol boosts a chunk that defines it."""
    defining = make_chunk("class UserService:\n    pass", "src/user_service.py")
    other = make_chunk("x = UserService()", "src/utils.py")
    scores: dict = {defining: 0.5, other: 0.4}

    boosted = apply_query_boost(scores, "how does UserService work", [defining, other])

    assert boosted[defining] > boosted[other]


def test_apply_query_boost_embedded_symbol_scans_non_candidates() -> None:
    """Non-candidate chunks on stem-matched files get boosted when defining the symbol."""
    defining = make_chunk("class StateManager:\n    pass", "src/state.py")
    candidate = make_chunk("x = 1", "src/other.py")
    scores: dict = {candidate: 0.5}

    boosted = apply_query_boost(scores, "how does StateManager work", [defining, candidate])

    assert defining in boosted
    assert boosted[defining] > 0


def test_apply_query_boost_nl_query_boosts_stem_match() -> None:
    """NL query keywords matching file stems boost those chunks."""
    auth_chunk = make_chunk("def authenticate(): pass", "src/auth.py")
    other_chunk = make_chunk("def foo(): pass", "src/utils.py")
    scores: dict = {auth_chunk: 0.5, other_chunk: 0.5}

    boosted = apply_query_boost(scores, "authenticate user session", [auth_chunk, other_chunk])

    assert boosted[auth_chunk] > boosted[other_chunk]


def test_apply_query_boost_empty_scores_returns_empty() -> None:
    """Empty scores dict returns empty dict without error."""
    assert apply_query_boost({}, "SomeQuery", []) == {}


def test_boost_multi_chunk_files_empty() -> None:
    """Empty dict is a no-op."""
    scores: dict = {}
    boost_multi_chunk_files(scores)
    assert scores == {}


def test_boost_multi_chunk_files_zero_max() -> None:
    """All-zero scores is a no-op (avoids division by zero)."""
    chunk = make_chunk("x = 1", "src/foo.py")
    scores: dict = {chunk: 0.0}
    boost_multi_chunk_files(scores)
    assert scores[chunk] == 0.0


def test_boost_multi_chunk_files_promotes_top_chunk() -> None:
    """File with multiple chunks gets its top chunk boosted."""
    c1 = make_chunk("def a(): pass", "src/big.py")
    c2 = make_chunk("def b(): pass", "src/big.py")
    c3 = make_chunk("def c(): pass", "src/small.py")
    scores: dict = {c1: 1.0, c2: 0.8, c3: 1.0}
    boost_multi_chunk_files(scores)
    assert scores[c1] > 1.0


@pytest.mark.parametrize(
    ("query", "alpha_in", "expected"),
    [
        ("MyService", 0.7, 0.7),  # explicit value returned as-is
        ("MyService", None, 0.3),  # symbol query → _ALPHA_SYMBOL
        ("how does routing work", None, 0.5),  # NL query → _ALPHA_NL
    ],
)
def test_resolve_alpha(query: str, alpha_in: float | None, expected: float) -> None:
    """resolve_alpha returns explicit alpha or auto-detects from query type."""
    assert resolve_alpha(query, alpha_in) == pytest.approx(expected)


def test_apply_query_boost_symbol_query_boosts_defining_chunk() -> None:
    """Symbol query boosts chunks that define the queried symbol."""
    defining = make_chunk("class MyService:\n    pass", "src/my_service.py")
    other = make_chunk("x = MyService()", "src/utils.py")
    scores: dict = {defining: 0.5, other: 0.4}

    boosted = apply_query_boost(scores, "MyService", [defining, other])

    assert boosted[defining] > boosted[other]


def test_apply_query_boost_symbol_scans_non_candidates() -> None:
    """Symbol query scans non-candidate chunks on stem-matched files."""
    defining = make_chunk("class MyService:\n    pass", "src/myservice.py")
    candidate = make_chunk("x = 1", "src/other.py")
    scores: dict = {candidate: 0.5}

    boosted = apply_query_boost(scores, "MyService", [defining, candidate])

    assert defining in boosted
    assert boosted[defining] > 0


def test_boost_embedded_symbols_non_matching_stem_skipped() -> None:
    """Non-candidate chunk with an unrelated stem is skipped in _boost_embedded_symbols."""
    defining = make_chunk("class UserService:\n    pass", "src/user_service.py")
    unrelated = make_chunk("x = 1", "src/totally_unrelated_name.py")
    scores: dict = {defining: 0.5}

    boosted = apply_query_boost(scores, "how does UserService work", [defining, unrelated])

    # The unrelated chunk should NOT be boosted (stem doesn't match UserService).
    assert unrelated not in boosted


def test_boost_stem_matches_all_stopwords_skips_boost() -> None:
    """NL query composed entirely of stopwords performs no stem boost."""
    chunk = make_chunk("def foo(): pass", "src/auth.py")
    scores: dict = {chunk: 0.5}

    boosted = apply_query_boost(scores, "the and or", [chunk])

    # Score should be unchanged because all words are stopwords.
    assert boosted[chunk] == pytest.approx(0.5)


def test_rerank_topk_empty_returns_empty_list() -> None:
    """rerank_topk with an empty scores dict returns an empty list."""
    assert rerank_topk({}, top_k=5) == []


def test_boost_symbol_definitions_empty_symbol_is_noop() -> None:
    """_boost_symbol_definitions with a whitespace-only query is a no-op."""
    chunk = make_chunk("class Foo:\n    pass", "src/foo.py")
    scores: dict = {chunk: 0.5}
    _boost_symbol_definitions(scores, "   ", 1.0, [chunk])
    assert scores[chunk] == 0.5


def test_apply_query_boost_namespace_qualified_adds_full_name() -> None:
    """Namespace-qualified symbol adds both the leaf and the full query to the names set."""
    defining = make_chunk("class Base:\n    pass", "src/base.py")
    scores: dict = {defining: 0.5}

    boosted = apply_query_boost(scores, "Sinatra::Base", [defining])

    assert boosted[defining] > 0.5


def test_scan_non_candidates_non_matching_stem_skipped() -> None:
    """Non-candidate chunks with a non-matching stem are skipped by _scan_non_candidates."""
    defining = make_chunk("class MyService:\n    pass", "src/myservice.py")
    unrelated = make_chunk("x = 1", "src/unrelated.py")
    # Only defining is a candidate; unrelated is NOT in scores and has a non-matching stem.
    scores: dict = {defining: 0.5}

    boosted = apply_query_boost(scores, "MyService", [defining, unrelated])

    assert unrelated not in boosted
