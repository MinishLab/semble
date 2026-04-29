from semble.ranking.boosting import is_symbol_query

_ALPHA_SYMBOL = 0.3  # lean BM25 for exact keyword matching
_ALPHA_NL = 0.5  # balanced semantic + BM25


def resolve_alpha(query: str, alpha: float | None) -> float:
    """Return the blending weight for semantic scores, auto-detecting from query type."""
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if is_symbol_query(query) else _ALPHA_NL
