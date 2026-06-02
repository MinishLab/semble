from semble import tokens as tokens_module


def test_tokenize_reuses_identifier_splits_for_repeated_tokens() -> None:
    """Repeated identifiers in large files should not repeat camel/snake split work."""
    token = "RepeatedIdentifierForSparseCache"
    real_camel = tokens_module._CAMEL_RE
    tokens_module._split_identifier_cached.cache_clear()
    camel_calls: list[str] = []

    class CountingCamel:
        def findall(self, value: str) -> list[str]:
            camel_calls.append(value)
            return real_camel.findall(value)

    try:
        tokens_module._CAMEL_RE = CountingCamel()  # type: ignore[assignment]
        assert tokens_module.tokenize(" ".join([token, token, token])) == [
            "repeatedidentifierforsparsecache",
            "repeated",
            "identifier",
            "for",
            "sparse",
            "cache",
            "repeatedidentifierforsparsecache",
            "repeated",
            "identifier",
            "for",
            "sparse",
            "cache",
            "repeatedidentifierforsparsecache",
            "repeated",
            "identifier",
            "for",
            "sparse",
            "cache",
        ]
    finally:
        tokens_module._CAMEL_RE = real_camel  # type: ignore[assignment]
        tokens_module._split_identifier_cached.cache_clear()

    assert camel_calls == [token]
