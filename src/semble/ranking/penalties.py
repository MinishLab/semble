import re
from pathlib import Path

from semble.types import Chunk

# Patterns that identify test files across common languages.
# Grouped by language for readability; combined into a single compiled regex.
_TEST_FILE_RE = re.compile(
    r"(?:^|/)"
    r"(?:"
    # Python
    r"test_[^/]*\.py"  # test_foo.py
    r"|[^/]*_test\.py"  # foo_test.py
    # Go
    r"|[^/]*_test\.go"  # foo_test.go
    # Java
    r"|[^/]*Tests?\.java"  # FooTest.java / FooTests.java
    # PHP
    r"|[^/]*Test\.php"  # FooTest.php
    # Ruby
    r"|[^/]*_spec\.rb"  # foo_spec.rb
    r"|[^/]*_test\.rb"  # foo_test.rb
    # JavaScript / TypeScript
    r"|[^/]*\.test\.[jt]sx?"  # foo.test.js/ts/jsx/tsx
    r"|[^/]*\.spec\.[jt]sx?"  # foo.spec.js/ts/jsx/tsx
    # Kotlin
    r"|[^/]*Tests?\.kt"  # FooTest.kt / FooTests.kt
    r"|[^/]*Spec\.kt"  # FooSpec.kt (Kotest)
    # Swift
    r"|[^/]*Tests?\.swift"  # FooTests.swift (XCTest)
    r"|[^/]*Spec\.swift"  # FooSpec.swift (Quick)
    # C#
    r"|[^/]*Tests?\.cs"  # FooTest.cs / FooTests.cs
    # C / C++
    r"|test_[^/]*\.cpp"  # test_foo.cpp (Google Test)
    r"|[^/]*_test\.cpp"  # foo_test.cpp (Google Test)
    r"|test_[^/]*\.c"  # test_foo.c
    r"|[^/]*_test\.c"  # foo_test.c
    # Scala
    r"|[^/]*Spec\.scala"  # FooSpec.scala (ScalaTest)
    r"|[^/]*Suite\.scala"  # FooSuite.scala (MUnit)
    r"|[^/]*Test\.scala"  # FooTest.scala
    # Dart
    r"|[^/]*_test\.dart"  # foo_test.dart
    r"|test_[^/]*\.dart"  # test_foo.dart
    # Lua
    r"|[^/]*_spec\.lua"  # foo_spec.lua (busted)
    r"|[^/]*_test\.lua"  # foo_test.lua
    r"|test_[^/]*\.lua"  # test_foo.lua (luaunit)
    # Shared helper patterns (all languages)
    r"|test_helpers?[^/]*\.\w+"  # test_helpers.go, test_helper.rb, etc.
    r")$"
)

# Directories whose contents are almost always test/spec code.
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")

# Regex matching path components that suggest a compatibility/legacy layer.
_COMPAT_DIR_RE = re.compile(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)")

# Regex matching path components that are examples or documentation code.
_EXAMPLES_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)")

# Regex matching TypeScript declaration files (stubs, not implementations).
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")

_STRONG_PENALTY = 0.3  # test files, compat shims, example/doc code
_MODERATE_PENALTY = 0.5  # __init__.py re-exports
_MILD_PENALTY = 0.7  # .d.ts declaration stubs (still carry useful type info)

# Maximum chunks from the same file before a saturation penalty is applied.
_FILE_SATURATION_THRESHOLD = 2

# Multiplicative penalty per extra chunk from the same file beyond the threshold.
_FILE_SATURATION_DECAY = 0.5


def rerank_topk(
    scores: dict[Chunk, float],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Select top-k results with file-path penalties and file-saturation decay.

    File-path penalties are applied first.  Candidates are then processed in
    descending penalised-score order with saturation decay applied greedily.
    Because decay only reduces scores and candidates are sorted by penalised
    score descending, we can stop early once the remaining scores cannot beat
    the current top-k floor.
    """
    if not scores:
        return []

    # Apply file-path penalties.
    penalty_cache: dict[str, float] = {}
    penalised: dict[Chunk, float] = {}
    for chunk, score in scores.items():
        if chunk.file_path not in penalty_cache:
            is_test = _is_test_file(chunk.file_path)
            penalty_cache[chunk.file_path] = _file_path_penalty(chunk.file_path, is_test=is_test)
        penalised[chunk] = score * penalty_cache[chunk.file_path]

    # Sort by penalised score (highest first) — single sort.
    ranked = sorted(penalised, key=lambda c: -penalised[c])

    # Greedy pass with file-saturation decay and early-exit.
    # Candidates are already sorted by pen_score descending, so pen_score is
    # an upper bound on any future eff_score (decay only reduces scores).
    # Once we have top_k items, any candidate whose pen_score cannot beat the
    # current k-th best effective score can be skipped — and so can every
    # subsequent candidate, so we break.
    # min_selected tracks the minimum effective score among the top_k collected
    # so far; it is recomputed after each addition to stay accurate.
    file_selected: dict[str, int] = {}
    selected: list[tuple[float, Chunk]] = []
    min_selected = float("+inf")

    for chunk in ranked:
        pen_score = penalised[chunk]

        if len(selected) >= top_k and pen_score <= min_selected:
            break

        already_selected = file_selected.get(chunk.file_path, 0)
        eff_score = pen_score
        if already_selected >= _FILE_SATURATION_THRESHOLD:
            excess = already_selected - _FILE_SATURATION_THRESHOLD + 1
            eff_score *= _FILE_SATURATION_DECAY**excess

        selected.append((eff_score, chunk))
        file_selected[chunk.file_path] = already_selected + 1

        if len(selected) >= top_k:
            min_selected = min(s for s, _ in selected)

    selected.sort(key=lambda t: -t[0])
    return [(chunk, score) for score, chunk in selected[:top_k]]


def _is_test_file(file_path: str) -> bool:
    """Return True if the file path matches common test-file naming conventions or lives in a test directory."""
    normalised = file_path.replace("\\", "/")
    return _TEST_FILE_RE.search(normalised) is not None or _TEST_DIR_RE.search(normalised) is not None


def _is_init_file(file_path: str) -> bool:
    """Return True if the file is a Python ``__init__.py``.

    These files typically re-export a module's public API but rarely contain
    the implementation.  ``index.js``/``index.ts`` and ``mod.rs`` are NOT
    penalised because they frequently hold primary implementation code
    (e.g. Express's ``index.js`` IS ``createApplication``).
    """
    return Path(file_path).name == "__init__.py"


def _file_path_penalty(file_path: str, *, is_test: bool) -> float:
    """Compute a multiplicative penalty for a file based on its path.

    Penalties are combined multiplicatively so that a file matching multiple
    patterns (e.g. a test helper in a compat directory) receives all applicable
    discounts.
    """
    normalised = file_path.replace("\\", "/")
    penalty = 1.0
    if is_test:
        penalty *= _STRONG_PENALTY
    if _is_init_file(file_path):
        penalty *= _MODERATE_PENALTY
    if _COMPAT_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _EXAMPLES_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _TYPE_DEFS_RE.search(normalised):
        penalty *= _MILD_PENALTY
    return penalty
