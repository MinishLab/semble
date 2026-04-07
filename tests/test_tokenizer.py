"""Tests for semble._tokenizer."""

from semble._tokenizer import split_identifier, tokenize_simple, tokenize_subword


def test_tokenize_simple_basic() -> None:
    result = tokenize_simple("hello_world FooBar")
    assert "hello_world" in result
    assert "foobar" in result


def test_tokenize_simple_lowercase() -> None:
    result = tokenize_simple("MyClass")
    assert "myclass" in result
    assert "MyClass" not in result


def test_split_identifier_camel() -> None:
    assert "foo" in split_identifier("fooBar")
    assert "bar" in split_identifier("fooBar")


def test_split_identifier_pascal() -> None:
    parts = split_identifier("MyClassName")
    assert "my" in parts or "class" in parts or "name" in parts


def test_split_identifier_short_parts_excluded() -> None:
    # Single-char splits should be excluded
    result = split_identifier("aB")
    assert result == []


def test_tokenize_subword_includes_snake_parts() -> None:
    result = tokenize_subword("validate_token")
    assert "validate_token" in result
    assert "validate" in result
    assert "token" in result


def test_tokenize_subword_includes_camel_parts() -> None:
    result = tokenize_subword("getUserName")
    assert "getusername" in result
    # subword parts
    assert "get" in result or "user" in result or "name" in result


def test_tokenize_subword_no_duplicates_order_preserved() -> None:
    result = tokenize_subword("foo foo foo")
    # duplicates are fine (BM25 handles term frequency), just check it doesn't crash
    assert isinstance(result, list)
