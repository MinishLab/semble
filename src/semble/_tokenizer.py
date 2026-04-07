"""Tokenizers for BM25 and symbol search."""

from __future__ import annotations

import re

QUERY_STOPS: frozenset[str] = frozenset(
    {
        "self",
        "def",
        "class",
        "return",
        "import",
        "from",
        "if",
        "else",
        "elif",
        "for",
        "in",
        "is",
        "not",
        "and",
        "or",
        "none",
        "true",
        "false",
        "try",
        "except",
        "raise",
        "with",
        "as",
        "pass",
        "the",
        "a",
        "an",
        "of",
        "to",
        "this",
        "that",
        "it",
        "how",
        "does",
        "do",
        "what",
        "where",
        "when",
        "which",
        "who",
        "are",
        "was",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
    }
)


def tokenize_simple(text: str) -> list[str]:
    """Simple identifier split. Best for standalone BM25."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def split_identifier(tok: str) -> list[str]:
    """Split camelCase/PascalCase into 2+ char parts."""
    return [
        p.lower()
        for p in re.findall(r"[A-Z]?[a-z]{2,}|[A-Z]{2,}(?=[A-Z][a-z]|\d|\b)", tok)
        if len(p) >= 2
    ]


def tokenize_subword(text: str) -> list[str]:
    """Original tokens + subword splits. Best for hybrid fusion."""
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    tokens = [t.lower() for t in raw]
    extras: list[str] = []
    for tok in raw:
        if "_" in tok:
            extras.extend(p.lower() for p in tok.split("_") if len(p) >= 2)
        extras.extend(split_identifier(tok))
    return tokens + extras
