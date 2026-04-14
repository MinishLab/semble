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

# Test/spec directories.
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")

# Compat/legacy path components.
_COMPAT_DIR_RE = re.compile(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)")

# Examples/docs path components.
_EXAMPLES_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)")

# TypeScript declaration files (.d.ts stubs).
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
    *,
    penalise_paths: bool = True,
) -> list[tuple[Chunk, float]]:
    """Select top-k results with optional file-path penalties and file-saturation decay.

    When `penalise_paths` is True, path penalties are applied before sorting.
    Saturation decay is applied greedily during the greedy pass; because decay
    only reduces scores and candidates are pre-sorted descending, early exit is
    safe once the remaining scores cannot beat the current k-th best.

    :param scores: Combined scores for candidate chunks.
    :param top_k: Maximum number of results to return.
    :param penalise_paths: Apply file-path penalties (test files, __init__.py, compat dirs,
        etc.). Set to False for pure-semantic queries where these priors do not apply.
    :return: Sorted list of (chunk, score) pairs, highest score first.
    """
    if not scores:
        return []

    # Apply file-path penalties.
    penalty_cache: dict[str, float] = {}
    penalised: dict[Chunk, float] = {}
    for chunk, score in scores.items():
        if penalise_paths:
            if chunk.file_path not in penalty_cache:
                penalty_cache[chunk.file_path] = _file_path_penalty(chunk.file_path)
            penalised[chunk] = score * penalty_cache[chunk.file_path]
        else:
            penalised[chunk] = score

    # Sort by penalised score (highest first) — single sort.
    ranked = sorted(penalised, key=lambda c: -penalised[c])

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


def _file_path_penalty(file_path: str) -> float:
    """Return a combined multiplicative penalty for all applicable path patterns."""
    normalised = file_path.replace("\\", "/")
    penalty = 1.0
    if _TEST_FILE_RE.search(normalised) is not None or _TEST_DIR_RE.search(normalised) is not None:
        penalty *= _STRONG_PENALTY
    if Path(file_path).name == "__init__.py":
        penalty *= _MODERATE_PENALTY
    if _COMPAT_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _EXAMPLES_DIR_RE.search(normalised):
        penalty *= _STRONG_PENALTY
    if _TYPE_DEFS_RE.search(normalised):
        penalty *= _MILD_PENALTY
    return penalty
