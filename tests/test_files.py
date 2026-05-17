from pathlib import Path

from semble.index.files import _DOC_LANGUAGES, _EXTENSION_TO_LANGUAGE, detect_language, get_extensions


def test_detect_language() -> None:
    """Test the detect_language function."""
    assert detect_language(Path("a.py")) == "python"
    assert detect_language(Path("b.js")) == "javascript"
    assert detect_language(Path("c.txt")) is None


def test_get_extensions() -> None:
    """Test the get_extensions function."""
    all_extensions = get_extensions(True, None)
    without_doc_extensions = get_extensions(False, None)

    doc_extensions = set(all_extensions) - set(without_doc_extensions)

    for extension in doc_extensions:
        assert _EXTENSION_TO_LANGUAGE[extension] in _DOC_LANGUAGES
    for extension in without_doc_extensions:
        assert _EXTENSION_TO_LANGUAGE[extension] not in _DOC_LANGUAGES


def test_get_extensions_additional() -> None:
    """Test the get_extensions function."""
    all_extensions = get_extensions(True, None)
    all_extensions_extra = get_extensions(True, [".kjs"])

    assert set(all_extensions_extra) == set(all_extensions) | {".kjs"}

    all_extensions = get_extensions(False, None)
    all_extensions_extra = get_extensions(False, [".kjs"])

    assert set(all_extensions_extra) == set(all_extensions) | {".kjs"}

    all_extensions = get_extensions(False, None)
    all_extensions_extra = get_extensions(False, [".py"])

    assert set(all_extensions_extra) == set(all_extensions)
