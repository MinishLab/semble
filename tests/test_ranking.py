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
    ("file_path", "is_test", "expected"),
    [
        ("src/auth.py", False, 1.0),
        ("src/semble/__init__.py", False, 0.5),
        ("tests/test_auth.py", True, 0.3),
        ("src/compat/old_api.py", False, 0.3),
        ("examples/demo.py", False, 0.3),
        ("src/types/index.d.ts", False, 0.7),
    ],
)
def test_file_path_penalty(file_path: str, is_test: bool, expected: float) -> None:
    """Path penalties are applied correctly per file type."""
    assert _file_path_penalty(file_path, is_test=is_test) == pytest.approx(expected)


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
    assert _chunk_defines_symbol(_make_chunk(content), symbol) is expected


def test_rerank_topk_init_demoted_by_default() -> None:
    """__init__.py is demoted below an equal-scored regular file."""
    init_chunk = _make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = _make_chunk("def authenticate(token): ...", "src/semble/auth.py")
    ranked = rerank_topk({init_chunk: 1.0, impl_chunk: 1.0}, top_k=2)
    assert ranked[0][0] == impl_chunk


def test_rerank_topk_penalise_paths_false_respects_scores() -> None:
    """penalise_paths=False leaves score order intact, including __init__.py."""
    init_chunk = _make_chunk("from .auth import authenticate", "src/semble/__init__.py")
    impl_chunk = _make_chunk("def authenticate(token): ...", "src/semble/auth.py")
    ranked = rerank_topk({init_chunk: 2.0, impl_chunk: 1.0}, top_k=2, penalise_paths=False)
    assert ranked[0][0] == init_chunk


def test_rerank_topk_saturation_decay_preserves_order() -> None:
    """Chunks beyond the saturation threshold get decay but results stay score-ordered."""
    chunks = [_make_chunk(f"def fn_{i}(): pass", "big_file.py") for i in range(5)]
    ranked = rerank_topk({c: float(5 - i) for i, c in enumerate(chunks)}, top_k=5)
    assert len(ranked) == 5
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
