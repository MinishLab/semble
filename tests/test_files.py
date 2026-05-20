from pathlib import Path

import pytest

from semble.index.files import _CODE_LANGUAGES, _DOC_LANGUAGES, _NON_CODE_LANGUAGES, detect_language, get_extensions
from semble.types import ContentType


def test_detect_language() -> None:
    """Test the detect_language function."""
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("b.js")) == "javascript"
    assert detect_language(Path("c.txt")) is None


def test_language_sets_are_consistent() -> None:
    """Code, doc, and non-code language sets satisfy their mutual invariants."""
    assert _CODE_LANGUAGES.isdisjoint(_DOC_LANGUAGES)
    assert _CODE_LANGUAGES.isdisjoint(_NON_CODE_LANGUAGES)
    assert _DOC_LANGUAGES <= _NON_CODE_LANGUAGES


@pytest.mark.parametrize(
    ("content", "includes", "excludes"),
    [
        (frozenset({ContentType.CODE}), [".py"], [".md"]),
        (frozenset({ContentType.DOCS}), [".md"], [".py"]),
        (frozenset({ContentType.ALL}), [".py", ".md"], []),
    ],
)
def test_get_extensions(content: frozenset[ContentType], includes: list[str], excludes: list[str]) -> None:
    """get_extensions returns the right extensions for each content type."""
    exts = set(get_extensions(content, None))
    for ext in includes:
        assert ext in exts
    for ext in excludes:
        assert ext not in exts


def test_get_extensions_code_and_docs() -> None:
    """Code + docs is the union of each individual set."""
    code = set(get_extensions(frozenset({ContentType.CODE}), None))
    docs = set(get_extensions(frozenset({ContentType.DOCS}), None))
    combined = set(get_extensions(frozenset({ContentType.CODE, ContentType.DOCS}), None))
    assert combined == code | docs


def test_get_extensions_additional() -> None:
    """Extra extensions are appended and existing ones are not duplicated."""
    base = get_extensions(frozenset({ContentType.ALL}), None)
    with_extra = get_extensions(frozenset({ContentType.ALL}), [".kjs"])
    assert set(with_extra) == set(base) | {".kjs"}

    base_code = get_extensions(frozenset({ContentType.CODE}), None)
    with_existing = get_extensions(frozenset({ContentType.CODE}), [".py"])
    assert set(with_existing) == set(base_code)
