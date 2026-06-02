import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Split on camelCase/PascalCase boundaries:
#   "HandlerStack" -> ["Handler", "Stack"]
#   "getHTTPResponse" -> ["get", "HTTP", "Response"]
#   "XMLParser" -> ["XML", "Parser"]
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


@lru_cache(maxsize=65_536)
def _split_identifier_cached(token: str) -> tuple[str, ...]:
    lower = token.lower()
    parts: list[str] = []

    if "_" in token:
        # snake_case splitting
        parts = [p for p in lower.split("_") if p]
    else:
        # camelCase / PascalCase splitting
        parts = [m.lower() for m in _CAMEL_RE.findall(token)]

    if len(parts) >= 2:
        return (lower, *parts)
    return (lower,)


def split_identifier(token: str) -> list[str]:
    """Split a single identifier into sub-tokens via camelCase/snake_case.

    Returns the original token (lowered) plus any sub-tokens.
    E.g. "HandlerStack" -> ["handlerstack", "handler", "stack"]
         "my_func" -> ["my_func", "my", "func"]
         "simple" -> ["simple"]
    """
    return list(_split_identifier_cached(token))


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing.

    Compound identifiers (camelCase, PascalCase, snake_case) are expanded
    into sub-tokens so that partial matches work. The original compound
    token is preserved for exact-match boosting.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for tok in raw_tokens:
        result.extend(split_identifier(tok))
    return result
