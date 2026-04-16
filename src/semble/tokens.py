import re

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Split on camelCase/PascalCase boundaries:
#   "HandlerStack" -> ["Handler", "Stack"]
#   "getHTTPResponse" -> ["get", "HTTP", "Response"]
#   "XMLParser" -> ["XML", "Parser"]
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _split_identifier(token: str) -> list[str]:
    """Split a single identifier into sub-tokens via camelCase/snake_case.

    Returns the original token (lowered) plus any sub-tokens.
    E.g. "HandlerStack" -> ["handlerstack", "handler", "stack"]
         "my_func" -> ["my_func", "my", "func"]
         "simple" -> ["simple"]
    """
    lower = token.lower()
    parts: list[str] = []

    if "_" in token:
        # snake_case splitting
        parts = [p for p in lower.split("_") if p]
    else:
        # camelCase / PascalCase splitting
        parts = [m.lower() for m in _CAMEL_RE.findall(token)]

    if len(parts) >= 2:
        return [lower, *parts]
    return [lower]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing.

    Compound identifiers (camelCase, PascalCase, snake_case) are expanded
    into sub-tokens so that partial matches work. The original compound
    token is preserved for exact-match boosting.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for tok in raw_tokens:
        result.extend(_split_identifier(tok))
    return result
