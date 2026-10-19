import pytest

from semble.tokens import stem, tokenize


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("handlers", "handler"),
        ("classes", "class"),
        ("matches", "match"),
        ("dependencies", "dependency"),
        ("parsed", "pars"),
        ("parsing", "pars"),
        ("status", "status"),  # -us is not a plural
        ("process", "process"),  # -ss is not a plural
        ("analysis", "analysis"),  # -is is not a plural
        ("ids", "ids"),  # 3 characters or fewer: unchanged
        ("http2", "http2"),  # digits: unchanged
        ("my_func", "my_func"),  # underscores: unchanged
    ],
)
def test_stem(token: str, expected: str) -> None:
    """The stemmer strips plurals, -ing and -ed, and leaves short or non-alphabetic tokens alone."""
    assert stem(token) == expected


def test_tokenize_stems_query_and_document_alike() -> None:
    """An inflected query word and its base form in code produce the same BM25 term."""
    assert "handler" in tokenize("registered handlers")
    assert "handler" in tokenize("class EventHandler:")
    assert tokenize("RequestHandlers") == ["requesthandler", "request", "handler"]
