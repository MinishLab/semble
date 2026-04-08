"""Tests for semble.search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from semble.search import (
    BM25Index,
    SemanticIndex,
    search_bm25,
    search_hybrid,
    search_semantic,
    search_symbol,
)
from semble.types import Chunk


def _make_chunk(content: str, file_path: str = "file.py") -> Chunk:
    import hashlib

    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        language="python",
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        _make_chunk("def authenticate(token):\n    return token == 'secret'", "auth.py"),
        _make_chunk("def login(username, password):\n    pass", "auth.py"),
        _make_chunk("class UserService:\n    pass", "users.py"),
        _make_chunk("def format_date(dt):\n    return str(dt)", "utils.py"),
    ]


@pytest.fixture
def embeddings(chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(chunks), 256)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)  # type: ignore[no-any-return]


@pytest.fixture
def bm25(chunks: list[Chunk]) -> BM25Index:
    return BM25Index([c.content for c in chunks])


@pytest.fixture
def semantic(chunks: list[Chunk], embeddings: npt.NDArray[np.float32]) -> SemanticIndex:
    return SemanticIndex(chunks, embeddings)


# BM25 tests


def test_bm25_returns_results(bm25: BM25Index, chunks: list[Chunk]) -> None:
    results = search_bm25("authenticate token", bm25, chunks, top_k=3)
    assert len(results) > 0


def test_bm25_source_label(bm25: BM25Index, chunks: list[Chunk]) -> None:
    results = search_bm25("authenticate", bm25, chunks, top_k=3)
    assert all(r.source == "bm25" for r in results)


def test_bm25_relevant_result_first(bm25: BM25Index, chunks: list[Chunk]) -> None:
    results = search_bm25("authenticate token", bm25, chunks, top_k=4)
    assert "authenticate" in results[0].chunk.content


def test_bm25_no_results_for_garbage(bm25: BM25Index, chunks: list[Chunk]) -> None:
    results = search_bm25("zzzznonexistentterm", bm25, chunks, top_k=3)
    assert results == []


# Semantic tests


def test_semantic_returns_results(semantic: SemanticIndex, mock_model: Any) -> None:
    results = search_semantic("login", mock_model, semantic, top_k=3)
    assert len(results) > 0


def test_semantic_source_label(semantic: SemanticIndex, mock_model: Any) -> None:
    results = search_semantic("query", mock_model, semantic, top_k=4)
    assert all(r.source == "semantic" for r in results)


def test_semantic_scores_between_0_and_1(semantic: SemanticIndex, mock_model: Any) -> None:
    results = search_semantic("query", mock_model, semantic, top_k=4)
    for r in results:
        assert -1.0 <= r.score <= 1.0


# Symbol tests


def _file_lines(tmp_path: Path, name: str, content: str) -> dict[str, list[str]]:
    f = tmp_path / name
    f.write_text(content)
    return {str(f): content.splitlines()}


def test_symbol_finds_function_by_name(tmp_path: Path) -> None:
    fl = _file_lines(
        tmp_path, "auth.py", "def authenticate(token):\n    return token == 'secret'\n"
    )
    results = search_symbol("authenticate", fl, top_k=5)
    assert len(results) == 1
    assert "authenticate" in results[0].chunk.content


def test_symbol_finds_class(tmp_path: Path) -> None:
    fl = _file_lines(tmp_path, "users.py", "class UserService:\n    pass\n")
    results = search_symbol("UserService", fl, top_k=5)
    assert len(results) == 1
    assert "UserService" in results[0].chunk.content


def test_symbol_definitions_ranked_above_usages(tmp_path: Path) -> None:
    fl = _file_lines(
        tmp_path, "auth.py", "# uses authenticate here\ndef authenticate(token):\n    pass\n"
    )
    results = search_symbol("authenticate", fl, top_k=5)
    assert any("def authenticate" in r.chunk.content for r in results)


def test_symbol_source_label(tmp_path: Path) -> None:
    fl = _file_lines(tmp_path, "auth.py", "def login(u, p):\n    pass\n")
    results = search_symbol("login", fl, top_k=3)
    assert all(r.source == "symbol" for r in results)


def test_symbol_no_results_for_nonsense(tmp_path: Path) -> None:
    fl = _file_lines(tmp_path, "auth.py", "def authenticate(token):\n    pass\n")
    results = search_symbol("zzznomatch", fl, top_k=5)
    assert results == []


def test_symbol_one_result_per_file(tmp_path: Path) -> None:
    fl = _file_lines(tmp_path, "mod.py", "foo = 1\nfoo = 2\nfoo = 3\n")
    results = search_symbol("foo", fl, top_k=10)
    assert len(results) == 1


# Hybrid tests


def test_hybrid_returns_results(
    chunks: list[Chunk], semantic: SemanticIndex, bm25: BM25Index, mock_model: Any
) -> None:
    results = search_hybrid("authenticate token", mock_model, semantic, bm25, chunks, top_k=3)
    assert len(results) > 0


def test_hybrid_source_label(
    chunks: list[Chunk], semantic: SemanticIndex, bm25: BM25Index, mock_model: Any
) -> None:
    results = search_hybrid("login", mock_model, semantic, bm25, chunks, top_k=4)
    assert all(r.source == "hybrid" for r in results)
