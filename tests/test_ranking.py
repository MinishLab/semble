import pytest

from semble.ranking.boosting import (
    _chunk_defines_symbol,
    _count_keyword_matches,
    _extract_symbol_name,
    _is_symbol_query,
    _stem_matches,
    apply_query_boost,
    boost_multi_chunk_files,
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


# ---------------------------------------------------------------------------
# _extract_symbol_name
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _stem_matches
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _count_keyword_matches – prefix matching
# ---------------------------------------------------------------------------


def test_count_keyword_matches_exact() -> None:
    """All keywords match exactly."""
    assert _count_keyword_matches({"auth", "service"}, {"auth", "service", "utils"}) == 2


def test_count_keyword_matches_prefix() -> None:
    """Shorter keyword matches longer path-part as a prefix (min 3 chars)."""
    # "dep" is a 3-char prefix of "dependencies"
    assert _count_keyword_matches({"dep"}, {"dependencies"}) == 1


def test_count_keyword_matches_no_match() -> None:
    """No match returns zero."""
    assert _count_keyword_matches({"foo"}, {"bar", "baz"}) == 0


def test_count_keyword_matches_short_prefix_not_counted() -> None:
    """Prefix shorter than 3 chars is not counted."""
    assert _count_keyword_matches({"ab"}, {"abcdef"}) == 0


# ---------------------------------------------------------------------------
# apply_query_boost – NL query with embedded CamelCase symbol
# ---------------------------------------------------------------------------


def test_apply_query_boost_embedded_symbol_boosts_defining_chunk() -> None:
    """NL query with CamelCase symbol boosts a chunk that defines it."""
    defining = make_chunk("class UserService:\n    pass", "src/user_service.py")
    other = make_chunk("x = UserService()", "src/utils.py")
    scores: dict = {defining: 0.5, other: 0.4}

    boosted = apply_query_boost(scores, "how does UserService work", [defining, other])

    assert boosted[defining] > boosted[other]


def test_apply_query_boost_embedded_symbol_scans_non_candidates() -> None:
    """Non-candidate chunks on stem-matched files get boosted when defining the symbol."""
    # This chunk is NOT in the initial scores dict (non-candidate)
    defining = make_chunk("class StateManager:\n    pass", "src/state.py")
    candidate = make_chunk("x = 1", "src/other.py")
    scores: dict = {candidate: 0.5}

    boosted = apply_query_boost(scores, "how does StateManager work", [defining, candidate])

    # defining chunk should have been added to boosted results
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
    result = apply_query_boost({}, "SomeQuery", [])
    assert result == {}


# ---------------------------------------------------------------------------
# boost_multi_chunk_files
# ---------------------------------------------------------------------------


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
    # big.py top chunk (c1) should get a coherence boost
    assert scores[c1] > 1.0
