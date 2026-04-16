import re
from pathlib import Path

from semble.tokens import _split_identifier
from semble.types import Chunk

# Matches queries that look like symbol lookups. A query is treated as a symbol if it:
#   - is namespace-qualified (e.g. Sinatra::Base, self->field, a.b)
#   - starts with an underscore (e.g. _private, __init__)
#   - contains an uppercase letter or underscore (e.g. HTTPAdapter, field_validator, getUser)
#   - starts with an uppercase letter (e.g. URL, Base)
# Purely lowercase single words (e.g. "session", "response") are NOT matched —
# those are natural language queries that should use semantic search.
_SYMBOL_QUERY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"  # namespace-qualified
    r"|_[A-Za-z0-9_]*"  # leading underscore (_private, __init__)
    r"|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*"  # contains uppercase or underscore after pos 0
    r"|[A-Z][A-Za-z0-9]*"  # starts with uppercase (URL, Base, HTTPAdapter)
    r")$"
)

# Alpha values for query-adaptive blending.
_ALPHA_SYMBOL = 0.3  # Symbol queries: lean BM25 for exact keyword matching
_ALPHA_NL = 0.5  # Natural language queries: balanced semantic + BM25

# Definition keywords used across common languages.
# Case-sensitive: most language keywords are lowercase by convention, and applying
# IGNORECASE globally causes false positive boosts (e.g. "Module" in Python docs,
# "Class" method calls in Ruby).
_DEFINITION_KEYWORDS = (
    "class",
    "module",
    "def",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "func",
    "function",
    "object",
    "abstract class",
    "data class",
    "fn",
    "fun",  # Kotlin
    "package",
    "namespace",
    "protocol",  # Swift
    "record",  # C# 9+, Java 16+
    "typedef",  # C/C++/Dart: `typedef struct Foo Foo`
)

# SQL DDL keywords matched case-insensitively (SQL is commonly written in either
# all-caps or all-lowercase; mixing with IGNORECASE avoids duplicating entries).
_SQL_DEFINITION_KEYWORDS = (
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE PROCEDURE",
    "CREATE FUNCTION",
)

# Precompiled alternation bodies — the fixed part of each pattern.
# Only symbol_name changes per call; re.escape(symbol_name) is substituted
# into the suffix at call time.
_KEYWORD_PREFIX = r"(?:^|(?<=\s))(?:"
_DEFINITION_KEYWORD_BODY = "|".join(re.escape(keyword) for keyword in _DEFINITION_KEYWORDS)
_SQL_KEYWORD_BODY = "|".join(re.escape(keyword) for keyword in _SQL_DEFINITION_KEYWORDS)

# Additive boost multiplier for chunks that define a queried symbol.
_DEFINITION_BOOST_MULTIPLIER = 2.0

# Additive boost multiplier for NL queries when file stems match query words.
_STEM_BOOST_MULTIPLIER = 1.0

# Common English stopwords excluded from file-stem matching for NL queries.
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how if in is it not of on or the to was"
    " what when where which who why with".split()
)


def resolve_alpha(query: str, alpha: float | None) -> float:
    """Return the blending weight for semantic scores, auto-detecting from query type."""
    if alpha is not None:
        return alpha
    return _ALPHA_SYMBOL if _is_symbol_query(query) else _ALPHA_NL


def apply_query_boost(
    combined_scores: dict[Chunk, float],
    query: str,
    all_chunks: list[Chunk],
) -> dict[Chunk, float]:
    """Apply query-type-specific boosts to candidate scores.

    Dispatches to symbol-definition boosting or NL file-stem boosting
    based on query type.

    :param combined_scores: Existing combined scores for candidate chunks.
    :param query: The raw query string.
    :param all_chunks: The full chunk list (used for non-candidate definition scanning).
    :return: Updated scores dict with boosts applied.
    """
    if not combined_scores:
        return combined_scores

    max_score = max(combined_scores.values())
    boosted = dict(combined_scores)

    if _is_symbol_query(query):
        _boost_symbol_definitions(boosted, query, max_score, all_chunks)
    else:
        _boost_stem_matches(boosted, query, max_score)

    return boosted


def _is_symbol_query(query: str) -> bool:
    """Return True if the query looks like a bare symbol or namespace-qualified identifier."""
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def _extract_symbol_name(query: str) -> str:
    """Extract the final identifier from a possibly namespace-qualified query.

    Examples: "Sinatra::Base" → "Base", "Client" → "Client".
    """
    for separator in ("::", "\\", "->", "."):
        if separator in query:
            return query.rsplit(separator, 1)[-1]
    return query.strip()


def _chunk_defines_symbol(chunk: Chunk, symbol_name: str) -> bool:
    """Check whether a chunk contains a definition of *symbol_name*.

    Two passes: case-sensitive for general keywords (to avoid false positives
    from e.g. `Module.new` in Ruby or `Class` in docstrings), then
    case-insensitive for SQL DDL keywords where mixed-case is common.
    """
    escaped_symbol = re.escape(symbol_name)
    suffix = r")\s+" + escaped_symbol + r"(?:\s|[<({:\[;]|$)"
    if re.compile(_KEYWORD_PREFIX + _DEFINITION_KEYWORD_BODY + suffix, re.MULTILINE).search(chunk.content) is not None:
        return True
    return (
        re.compile(_KEYWORD_PREFIX + _SQL_KEYWORD_BODY + suffix, re.MULTILINE | re.IGNORECASE).search(chunk.content)
        is not None
    )


def _file_stem_matches_symbol(chunk: Chunk, symbol_name: str) -> bool:
    """Return True if the chunk's file stem matches the symbol name (case-insensitive, snake_case/PascalCase-aware)."""
    stem = Path(chunk.file_path).stem.lower()
    return stem == symbol_name.lower() or stem.replace("_", "") == symbol_name.lower()


def _definition_tier(chunk: Chunk, names: set[str], boost_unit: float) -> float:
    """Return the boost amount for a chunk that defines one of *names*.

    Tier 1.5 x boost_unit if the file stem also matches (strong signal).
    Tier 1.0 x boost_unit for definition keyword match alone.
    Returns 0.0 if the chunk does not define any of *names*.
    """
    if not any(_chunk_defines_symbol(chunk, name) for name in names):
        return 0.0
    has_stem = any(_file_stem_matches_symbol(chunk, name) for name in names)
    return boost_unit * (1.5 if has_stem else 1.0)


def _boost_symbol_definitions(
    boosted: dict[Chunk, float],
    query: str,
    max_score: float,
    all_chunks: list[Chunk],
) -> None:
    """Boost chunks that define the queried symbol (in-place).

    Scans both candidates and non-candidates whose file stem matches the
    symbol.  Non-candidate scanning is needed for large repos where the
    definition file may not rank in the top-N candidates despite BM25 stem
    enrichment.

    Definition tiers (see `_definition_tier`):
      - 1.5x boost_unit: definition keyword + file-stem match
      - 1.0x boost_unit: definition keyword only
    """
    symbol_name = _extract_symbol_name(query)
    if not symbol_name:
        return

    names = {symbol_name}
    if symbol_name != query.strip():
        names.add(query.strip())

    boost_unit = max_score * _DEFINITION_BOOST_MULTIPLIER

    for chunk in list(boosted):
        tier = _definition_tier(chunk, names, boost_unit)
        if tier:
            boosted[chunk] += tier

    # Scan non-candidate chunks whose file stem matches the symbol.
    # In large repos the definition file may not rank in the top-N candidates
    # despite BM25 stem enrichment; scanning by stem ensures it is found.
    symbol_lower = symbol_name.lower()
    for chunk in all_chunks:
        if chunk in boosted:
            continue
        stem = Path(chunk.file_path).stem.lower()
        if stem != symbol_lower and stem.replace("_", "") != symbol_lower:
            continue
        tier = _definition_tier(chunk, names, boost_unit)
        if tier:
            boosted[chunk] = tier


def _fuzzy_keyword_overlap(keywords: set[str], parts: set[str]) -> int:
    """Count query keywords that match path parts, allowing prefix overlap (min 3 chars)."""
    exact = keywords & parts
    if len(exact) == len(keywords):
        return len(exact)
    n_matches = len(exact)
    for keyword in keywords - exact:
        for part in parts:
            shorter, longer = (keyword, part) if len(keyword) <= len(part) else (part, keyword)
            if len(shorter) >= 3 and longer.startswith(shorter):
                n_matches += 1
                break
    return n_matches


def _boost_stem_matches(
    boosted: dict[Chunk, float],
    query: str,
    max_score: float,
) -> None:
    """Boost chunks whose file paths match NL query keywords (in-place).

    Uses prefix matching for morphological variants (e.g. "dependency" matches
    "dependencies").  Matches file stems and the immediate parent directory name.
    """
    keywords = {
        word.lower()
        for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
    if not keywords:
        return

    boost = max_score * _STEM_BOOST_MULTIPLIER
    path_cache: dict[str, set[str]] = {}
    for chunk in list(boosted):
        if chunk.file_path not in path_cache:
            path = Path(chunk.file_path)
            parts: set[str] = set(_split_identifier(path.stem))
            if path.parent.name and path.parent.name not in (".", "/", ".."):
                parts.update(_split_identifier(path.parent.name))
            path_cache[chunk.file_path] = parts
        n_matches = _fuzzy_keyword_overlap(keywords, path_cache[chunk.file_path])
        if n_matches > 0:
            match_ratio = n_matches / len(keywords)
            if match_ratio >= 0.10:
                boosted[chunk] += boost * match_ratio
