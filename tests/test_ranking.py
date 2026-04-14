"""Unit tests for ranking heuristics: symbol detection, path penalties, definition boosting."""

import pytest

from semble.ranking.boosting import _chunk_defines_symbol, _is_symbol_query
from semble.ranking.penalties import _file_path_penalty, _is_test_file, rerank_topk
from semble.types import Chunk


def _make_chunk(content: str, file_path: str = "src/module.py") -> Chunk:
    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        language="python",
        content_hash=content[:16],
    )


# ---------------------------------------------------------------------------
# _is_symbol_query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "HTTPAdapter",
        "TCPConnector",
        "AFError",
        "OAuth2PasswordBearer",
        "field_validator",
        "render_template",
        "URL",
        "IMessagePackFormatter",
        "TRPCError",
        "HttpClient",
        "getUser",
        "Sinatra::Base",
        "self->field",
        "_private",
        "__init__",
    ],
)
def test_is_symbol_query_true(query: str) -> None:
    """Identifiers with uppercase, underscores, or namespace separators are symbols."""
    assert _is_symbol_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "session",
        "response",
        "distill",
        "quantize",
        "authentication",
        "how does routing work",
    ],
)
def test_is_symbol_query_false(query: str) -> None:
    """Purely lowercase words and NL phrases are not symbols."""
    assert _is_symbol_query(query) is False


# ---------------------------------------------------------------------------
# _file_path_penalty
# ---------------------------------------------------------------------------


def test_file_path_penalty_normal_file() -> None:
    """Normal source files have no penalty (multiplier = 1.0)."""
    assert _file_path_penalty("src/auth.py", is_test=False) == pytest.approx(1.0)


def test_file_path_penalty_init_file() -> None:
    """__init__.py receives the moderate penalty."""
    penalty = _file_path_penalty("src/semble/__init__.py", is_test=False)
    assert penalty < 1.0
    assert penalty == pytest.approx(0.5)


def test_file_path_penalty_test_file() -> None:
    """Test files receive the strong penalty."""
    penalty = _file_path_penalty("tests/test_auth.py", is_test=True)
    assert penalty == pytest.approx(0.3)


def test_file_path_penalty_compat_dir() -> None:
    """Files under compat/ receive the strong penalty."""
    penalty = _file_path_penalty("src/compat/old_api.py", is_test=False)
    assert penalty == pytest.approx(0.3)


def test_file_path_penalty_examples_dir() -> None:
    """Files under examples/ receive the strong penalty."""
    penalty = _file_path_penalty("examples/demo.py", is_test=False)
    assert penalty == pytest.approx(0.3)


def test_file_path_penalty_type_defs() -> None:
    """TypeScript .d.ts declaration files receive the mild penalty."""
    penalty = _file_path_penalty("src/types/index.d.ts", is_test=False)
    assert penalty == pytest.approx(0.7)


def test_is_test_file_python() -> None:
    """Standard Python test file naming conventions are detected."""
    assert _is_test_file("tests/test_auth.py") is True
    assert _is_test_file("auth_test.py") is True
    assert _is_test_file("src/auth.py") is False


# ---------------------------------------------------------------------------
# _chunk_defines_symbol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "symbol"),
    [
        ("class UserService:\n    pass", "UserService"),
        ("def authenticate(token):\n    return token", "authenticate"),
        ("interface ILogger {\n  log(msg: string): void;\n}", "ILogger"),
        ("struct Config {\n    host: String,\n}", "Config"),
        ("CREATE TABLE users (\n  id INT\n);", "users"),
        ("CREATE TABLE Users (\n  id INT\n);", "Users"),  # SQL case-insensitive
    ],
)
def test_chunk_defines_symbol_true(content: str, symbol: str) -> None:
    """Chunks containing a definition keyword followed by the symbol name match."""
    chunk = _make_chunk(content)
    assert _chunk_defines_symbol(chunk, symbol) is True


@pytest.mark.parametrize(
    ("content", "symbol"),
    [
        ("x = UserService()\n", "UserService"),  # instantiation, not definition
        ("# authenticate is a helper\n", "authenticate"),  # comment only
        ("return Config(host='localhost')", "Config"),  # call, not definition
    ],
)
def test_chunk_defines_symbol_false(content: str, symbol: str) -> None:
    """Chunks that only use (but do not define) a symbol do not match."""
    chunk = _make_chunk(content)
    assert _chunk_defines_symbol(chunk, symbol) is False


# ---------------------------------------------------------------------------
# rerank_topk — __init__.py demotion and penalise_paths=False
# ---------------------------------------------------------------------------


def test_rerank_topk_init_demoted_by_default() -> None:
    """__init__.py is demoted below an equally-scored regular file when penalise_paths=True."""
    init_chunk = _make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = _make_chunk("def authenticate(token): ...", "src/semble/auth.py")

    scores = {init_chunk: 1.0, impl_chunk: 1.0}
    ranked = rerank_topk(scores, top_k=2)

    assert ranked[0][0] == impl_chunk, "__init__.py should be demoted below equal-scored impl file"
    assert ranked[1][0] == init_chunk


def test_rerank_topk_init_not_demoted_when_penalise_paths_false() -> None:
    """__init__.py keeps its score rank when penalise_paths=False (e.g. pure-semantic query)."""
    init_chunk = _make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = _make_chunk("def authenticate(token): ...", "src/semble/auth.py")

    # Give __init__.py a higher score to confirm it wins when penalties are off.
    scores = {init_chunk: 2.0, impl_chunk: 1.0}
    ranked = rerank_topk(scores, top_k=2, penalise_paths=False)

    assert ranked[0][0] == init_chunk, "__init__.py should rank first when it has highest score and penalties are off"


def test_rerank_topk_saturation_decay() -> None:
    """More than 2 chunks from the same file get saturation decay applied."""
    chunks = [_make_chunk(f"def fn_{i}(): pass", "big_file.py") for i in range(5)]
    # Give them descending scores so they naturally all want to be selected.
    scores = {c: float(5 - i) for i, c in enumerate(chunks)}
    ranked = rerank_topk(scores, top_k=5)

    # All 5 are returned (no cutoff), but the 3rd+ get decay applied.
    assert len(ranked) == 5
    # Scores should still be in descending order.
    result_scores = [s for _, s in ranked]
    assert result_scores == sorted(result_scores, reverse=True)
