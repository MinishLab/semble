from pathlib import Path

import pytest

from semble.index.files import (
    _CODE_LANGUAGES,
    _CONFIG_LANGUAGES,
    _DATA_LANGUAGES,
    _DOC_LANGUAGES,
    detect_language,
    detect_language_from_shebang,
    get_extensions,
    get_shebang_languages,
)
from semble.types import ContentType


def test_detect_language() -> None:
    """Test the detect_language function."""
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("b.js")) == "javascript"
    assert detect_language(Path("c.txt")) is None


def test_language_sets_are_consistent() -> None:
    """Code, doc, config, and data language sets are mutually disjoint."""
    sets = {"code": _CODE_LANGUAGES, "docs": _DOC_LANGUAGES, "config": _CONFIG_LANGUAGES, "data": _DATA_LANGUAGES}
    for a, set_a in sets.items():
        for b, set_b in sets.items():
            if a < b:
                assert set_a.isdisjoint(set_b), f"{a} and {b} overlap: {set_a & set_b}"


@pytest.mark.parametrize(
    ("types", "includes", "excludes"),
    [
        ([ContentType.CODE], [".py"], [".md", ".csv", ".toml"]),
        ([ContentType.DOCS], [".md"], [".py", ".csv", ".toml"]),
        ([ContentType.CONFIG], [".toml"], [".py", ".md", ".csv"]),
        ([ContentType.CODE, ContentType.DOCS], [".py", ".md"], [".csv", ".toml"]),
        (list(ContentType), [".py", ".md", ".toml"], []),
    ],
)
def test_get_extensions(types: list[ContentType], includes: list[str], excludes: list[str]) -> None:
    """get_extensions returns the right extensions for each combination of content types."""
    exts = set(get_extensions(types))
    for ext in includes:
        assert ext in exts
    for ext in excludes:
        assert ext not in exts


def test_all_excludes_data_extensions() -> None:
    """--content all does not include data file extensions (csv, json, tsv, psv)."""
    all_exts = set(get_extensions(list(ContentType)))
    for ext in (".csv", ".tsv", ".psv", ".json", ".json5"):
        assert ext not in all_exts, f"{ext} should not be indexed by 'all'"


@pytest.mark.parametrize(
    ("first_line", "expected"),
    [
        # env -S launcher wrapper: the real interpreter is the last mapped token.
        ("#!/usr/bin/env -S uv run --no-project --quiet python3", "python"),
        ("#!/usr/bin/env python3.12", "python"),
        ("#!/usr/bin/env bash", "bash"),
        ("#!/bin/sh", "bash"),
        ("#!/usr/bin/perl -w", "perl"),
        ("#!/usr/bin/env node", "javascript"),
        ("#!/usr/bin/env -S deno run --allow-net", "typescript"),
        ("#!/usr/bin/env Rscript", "r"),
        ("# not a shebang", None),
        ("", None),
        # A shebang only counts on the first line.
        ("plain text\n#!/bin/bash", None),
    ],
)
def test_detect_language_from_shebang(first_line: str, expected: str | None) -> None:
    """Shebang lines resolve to the interpreter language, ignoring env/launcher wrappers and flags."""
    assert detect_language_from_shebang(first_line) == expected


def test_get_shebang_languages() -> None:
    """Shebang languages are gated by content type and limited to interpreter-backed languages."""
    code = get_shebang_languages([ContentType.CODE])
    assert {"python", "bash"} <= code
    assert "markdown" not in code  # a docs language, never shebang-reachable
    assert get_shebang_languages([ContentType.DOCS]) == frozenset()
