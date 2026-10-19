import re

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Split on camelCase/PascalCase boundaries:
#   "HandlerStack" -> ["Handler", "Stack"]
#   "getHTTPResponse" -> ["get", "HTTP", "Response"]
#   "XMLParser" -> ["XML", "Parser"]
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Plural "-es" after a sibilant: "classes" -> "class", "matches" -> "match".
_ES_AFTER_SIBILANT_RE = re.compile(r"(?:s|x|z|ch|sh)es$")


def split_identifier(token: str) -> list[str]:
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


def stem(token: str) -> str:
    """Strip common English inflections so "handlers"/"handler" and "parsed"/"parse" share a BM25 term.

    Deliberately light (plurals, -ing, -ed) and applied to queries and documents alike,
    so over-stemming only merges terms on both sides. Tokens of 3 characters or fewer,
    and tokens with digits or underscores, are left as they are.
    E.g. "handlers" -> "handler", "classes" -> "class", "dependencies" -> "dependency"
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if _ES_AFTER_SIBILANT_RE.search(token):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing.

    Compound identifiers (camelCase, PascalCase, snake_case) are expanded
    into sub-tokens so that partial matches work. The original compound
    token is preserved for exact-match boosting. Every token is passed
    through :func:`stem`, so inflected forms share a term.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for tok in raw_tokens:
        result.extend(stem(part) for part in split_identifier(tok))
    return result
