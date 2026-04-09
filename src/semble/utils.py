import re

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing."""
    return _TOKEN_RE.findall(text.lower())
