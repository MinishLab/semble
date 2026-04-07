"""Tests for semble._search."""

from __future__ import annotations

import numpy as np
import pytest

from semble._search import (
    BM25Index,
    SemanticIndex,
    dedup_results,
    search_bm25,
    search_hybrid,
    search_semantic,
    search_symbol,
)
from semble._types import Chunk, SearchResult, SymbolKind


def _make_chunk(content: str, file_path: str = "file.py", symbol_name: str | None = None) -> Chunk:
    import hashlib

    return Chunk(
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=content.count("\n") + 1,
        symbol_name=symbol_name,
        symbol_kind=SymbolKind.FUNCTION if symbol_name else SymbolKind.CHUNK,
        language="python",
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
    )


@pytest.fixture
def chunks():
    return [
        _make_chunk(
            "def authenticate(token):\n    return token == 'secret'", "auth.py", "authenticate"
        ),
        _make_chunk("def login(username, password):\n    pass", "auth.py", "login"),
        _make_chunk("class UserService:\n    pass", "users.py", "UserService"),
        _make_chunk("def format_date(dt):\n    return str(dt)", "utils.py", "format_date"),
    ]


@pytest.fixture
def embeddings(chunks):
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(chunks), 256)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


@pytest.fixture
def bm25(chunks):
    return BM25Index(chunks)


@pytest.fixture
def semantic(chunks, embeddings):
    return SemanticIndex(chunks, embeddings)


@pytest.fixture
def hash_to_chunk(chunks):
    return {c.content_hash: c for c in chunks}


# BM25 tests


def test_bm25_returns_results(bm25, chunks) -> None:
    results = search_bm25("authenticate token", bm25, chunks, top_k=3)
    assert len(results) > 0


def test_bm25_source_label(bm25, chunks) -> None:
    results = search_bm25("authenticate", bm25, chunks, top_k=3)
    assert all(r.source == "bm25" for r in results)


def test_bm25_relevant_result_first(bm25, chunks) -> None:
    results = search_bm25("authenticate token", bm25, chunks, top_k=4)
    assert results[0].chunk.symbol_name == "authenticate"


def test_bm25_no_results_for_garbage(bm25, chunks) -> None:
    results = search_bm25("zzzznonexistentterm", bm25, chunks, top_k=3)
    assert results == []


# Semantic tests


def test_semantic_returns_results(chunks, embeddings, semantic, hash_to_chunk, mock_model) -> None:
    results = search_semantic("login", mock_model, semantic, chunks, hash_to_chunk, top_k=3)
    assert len(results) > 0


def test_semantic_source_label(chunks, embeddings, semantic, hash_to_chunk, mock_model) -> None:
    results = search_semantic("query", mock_model, semantic, chunks, hash_to_chunk, top_k=4)
    assert all(r.source == "semantic" for r in results)


def test_semantic_scores_between_0_and_1(
    chunks, embeddings, semantic, hash_to_chunk, mock_model
) -> None:
    results = search_semantic("query", mock_model, semantic, chunks, hash_to_chunk, top_k=4)
    for r in results:
        assert -1.0 <= r.score <= 1.0


# Symbol tests


def test_symbol_finds_function_by_name(chunks) -> None:
    results = search_symbol("authenticate", chunks, top_k=5)
    assert any(r.chunk.symbol_name == "authenticate" for r in results)


def test_symbol_finds_class(chunks) -> None:
    results = search_symbol("UserService", chunks, top_k=5)
    assert any(r.chunk.symbol_name == "UserService" for r in results)


def test_symbol_subword_match(chunks) -> None:
    # "user" should match "UserService" via subword splitting
    results = search_symbol("user service", chunks, top_k=5)
    assert any(r.chunk.symbol_name == "UserService" for r in results)


def test_symbol_source_label(chunks) -> None:
    results = search_symbol("login", chunks, top_k=3)
    assert all(r.source == "symbol" for r in results)


def test_symbol_no_results_for_nonsense(chunks) -> None:
    results = search_symbol("zzznomatch", chunks, top_k=5)
    assert results == []


# Hybrid tests


def test_hybrid_returns_results(
    chunks, embeddings, semantic, bm25, hash_to_chunk, mock_model
) -> None:
    results = search_hybrid(
        "authenticate token", mock_model, semantic, bm25, chunks, hash_to_chunk, top_k=3
    )
    assert len(results) > 0


def test_hybrid_source_label(chunks, embeddings, semantic, bm25, hash_to_chunk, mock_model) -> None:
    results = search_hybrid("login", mock_model, semantic, bm25, chunks, hash_to_chunk, top_k=4)
    assert all(r.source == "hybrid" for r in results)


# Dedup tests


def test_dedup_removes_exact_hash_duplicates(chunks, embeddings) -> None:
    cache = {c.content_hash: embeddings[i] for i, c in enumerate(chunks)}
    r = SearchResult(chunk=chunks[0], score=1.0, source="semantic")
    results = [r, r]  # exact duplicate
    deduped = dedup_results(results, cache)
    assert len(deduped) == 1


def test_dedup_keeps_distinct_results(chunks, embeddings) -> None:
    cache = {c.content_hash: embeddings[i] for i, c in enumerate(chunks)}
    results = [
        SearchResult(chunk=c, score=1.0 - i * 0.1, source="semantic") for i, c in enumerate(chunks)
    ]
    deduped = dedup_results(results, cache)
    # With random embeddings, all should be distinct
    assert len(deduped) == len(chunks)


def test_dedup_single_result_unchanged(chunks, embeddings) -> None:
    cache = {chunks[0].content_hash: embeddings[0]}
    results = [SearchResult(chunk=chunks[0], score=1.0, source="semantic")]
    assert dedup_results(results, cache) == results
