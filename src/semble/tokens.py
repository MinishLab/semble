import re

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

# Split on camelCase/PascalCase boundaries:
#   "HandlerStack" -> ["Handler", "Stack"]
#   "getHTTPResponse" -> ["get", "HTTP", "Response"]
#   "XMLParser" -> ["XML", "Parser"]
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Suffix rules for lightweight stemming: (suffix, replacement).
# Applied to sub-tokens of length > 4 to generate a stem variant that is
# added *alongside* the original — never replacing it — so exact matches
# still work while morphological variants (colors↔color, utility↔util,
# serialization↔serial) can also match.
# Ordered longest-first so that "tion" fires before "on", etc.
_STEM_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("tion", ""),
    ("ity", ""),
    ("ing", ""),
    ("ness", ""),
    ("ment", ""),
    ("ies", "y"),
    ("ers", ""),
    ("ed", ""),
    ("er", ""),
    ("es", ""),
    ("s", ""),
)
# Minimum length of the resulting stem (avoids over-stemming short words).
_MIN_STEM_LEN = 3


def _stem(word: str) -> str | None:
    """Return a stemmed variant of *word*, or None if no suffix applies.

    Only fires when the word is long enough that stripping the suffix still
    leaves a meaningful stem (>= ``_MIN_STEM_LEN`` chars).
    """
    if len(word) <= 4:
        return None
    for suffix, replacement in _STEM_SUFFIXES:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)] + replacement
            if stem != word and len(stem) >= _MIN_STEM_LEN:
                return stem
    return None


def _split_identifier(token: str) -> list[str]:
    """Split a single identifier into sub-tokens via camelCase/snake_case.

    Returns the original token (lowered) plus any sub-tokens, plus stemmed
    variants of each sub-token so that morphological variants match
    (e.g. ``colors`` also emits ``color``, ``utility`` also emits ``util``).

    E.g. "HandlerStack" -> ["handlerstack", "handler", "stack"]
         "my_func" -> ["my_func", "my", "func"]
         "colors"  -> ["colors", "color"]
         "simple"  -> ["simple"]
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
        base = [lower, *parts]
    else:
        base = [lower]

    # Append stemmed variants (deduplicated, not replacing originals).
    seen = set(base)
    stems: list[str] = []
    for p in base:
        s = _stem(p)
        if s is not None and s not in seen:
            stems.append(s)
            seen.add(s)

    return base + stems


def tokenize(text: str) -> list[str]:
    """Split text into lowercase identifier-like tokens for BM25 indexing.

    Compound identifiers (camelCase, PascalCase, snake_case) are expanded
    into sub-tokens so that partial matches work. The original compound
    token is preserved for exact-match boosting. Stemmed variants are added
    so that morphological variants (plurals, gerunds, nominalizations) match.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    result: list[str] = []
    for tok in raw_tokens:
        result.extend(_split_identifier(tok))
    return result
