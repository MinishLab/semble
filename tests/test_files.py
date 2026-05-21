from pathlib import Path

import pytest

from semble.index.files import _CODE_LANGUAGES, _CONFIG_LANGUAGES, _DOC_LANGUAGES, detect_language, get_extensions
from semble.types import ContentType


def test_detect_language() -> None:
    """Test the detect_language function."""
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("b.js")) == "javascript"
    assert detect_language(Path("c.txt")) is None


def test_language_sets_are_consistent() -> None:
    """Code, doc, and config language sets are mutually disjoint."""
    assert _CODE_LANGUAGES.isdisjoint(_DOC_LANGUAGES)
    assert _CODE_LANGUAGES.isdisjoint(_CONFIG_LANGUAGES)
    assert _DOC_LANGUAGES.isdisjoint(_CONFIG_LANGUAGES)


@pytest.mark.parametrize(
    ("content", "includes", "excludes"),
    [
        (ContentType.CODE, [".py"], [".md"]),
        (ContentType.DOCS, [".md"], [".py"]),
        (ContentType.ALL, [".py", ".md"], []),
    ],
)
def test_get_extensions(content: ContentType, includes: list[str], excludes: list[str]) -> None:
    """get_extensions returns the right extensions for each content type."""
    exts = set(get_extensions(content, None))
    for ext in includes:
        assert ext in exts
    for ext in excludes:
        assert ext not in exts


def test_get_extensions_additional() -> None:
    """Extra extensions are appended and existing ones are not duplicated."""
    base = get_extensions(ContentType.ALL, None)
    with_extra = get_extensions(ContentType.ALL, [".kjs"])
    assert set(with_extra) == set(base) | {".kjs"}

    base_code = get_extensions(ContentType.CODE, None)
    with_existing = get_extensions(ContentType.CODE, [".py"])
    assert set(with_existing) == set(base_code)
