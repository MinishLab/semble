from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semble.cache import (
    _get_valid_user_cache_dir,
    _linux_cache_dir,
    _windows_cache_dir,
    clear_cache,
    find_index_from_cache_folder,
    get_validated_cache,
    resolve_cache_folder,
    save_index_to_cache,
)
from semble.types import ContentType


def test_find_index_from_cache_folder_local_path(tmp_path: Path) -> None:
    """Local paths are normalised before hashing, result ends with /index."""
    result = find_index_from_cache_folder(str(tmp_path))
    assert result.name == "index"
    assert result == find_index_from_cache_folder(str(tmp_path))


def test_find_index_from_cache_folder_git_url() -> None:
    """Git URLs are hashed as-is (not expanded via Path.resolve)."""
    url = "https://github.com/org/repo.git"
    result = find_index_from_cache_folder(url)
    assert result.name == "index"
    assert result != find_index_from_cache_folder("https://github.com/org/other.git")


@pytest.mark.parametrize(
    ("env", "expected_base"),
    [
        ({"LOCALAPPDATA": "C:\\Local", "APPDATA": "C:\\Roaming"}, "C:\\Local"),
        ({"APPDATA": "C:\\Roaming"}, "C:\\Roaming"),
    ],
)
def test_windows_cache_dir_env(env: dict[str, str], expected_base: str) -> None:
    """_windows_cache_dir prefers LOCALAPPDATA, falls back to APPDATA."""
    with patch.dict("os.environ", env, clear=True):
        assert _windows_cache_dir("semble") == Path(expected_base) / "semble" / "Cache"


def test_linux_cache_dir_with_xdg() -> None:
    """_linux_cache_dir uses XDG_CACHE_HOME when set."""
    with patch.dict("os.environ", {"XDG_CACHE_HOME": "/xdg"}, clear=True):
        assert _linux_cache_dir("semble") == Path("/xdg") / "semble"


@pytest.mark.parametrize(
    ("fn", "expected_rel"),
    [
        (_windows_cache_dir, Path("AppData") / "Local" / "semble" / "Cache"),
        (_linux_cache_dir, Path(".cache") / "semble"),
    ],
)
def test_cache_dir_no_env(fn: object, expected_rel: Path) -> None:
    """Both helpers fall back to a home-relative path when no env vars are set."""
    home = Path("/fake/home")
    with patch.dict("os.environ", {}, clear=True):
        with patch("pathlib.Path.home", return_value=home):
            assert fn("semble") == home / expected_rel  # type: ignore[operator]


def test_save_index_to_cache(tmp_path: Path) -> None:
    """A freshly built index is saved under its cache key."""
    index = MagicMock(loaded_from_disk=False)
    with patch("semble.cache.find_index_from_cache_folder", return_value=tmp_path / "index"):
        save_index_to_cache(index, "repo")
    index.save.assert_called_once_with(tmp_path / "index")


@pytest.mark.parametrize(
    ("platform", "mock_target", "expected"),
    [
        ("win32", "semble.cache._windows_cache_dir", Path("/win")),
        ("linux", "semble.cache._linux_cache_dir", Path("/linux")),
        ("darwin", "semble.cache._macos_cache_dir", Path("/macos")),
    ],
)
def test_resolve_cache_folder(platform: str, mock_target: str, expected: Path) -> None:
    """resolve_cache_folder calls the correct platform helper."""
    with (
        patch.object(sys, "platform", platform),
        patch.dict("os.environ", {}, clear=True),
        patch(mock_target, return_value=expected) as mock_fn,
        patch("pathlib.Path.mkdir"),
    ):
        result = resolve_cache_folder()
    mock_fn.assert_called_once_with("semble")
    assert result == expected


def test_get_valid_user_cache_dir_relative_path() -> None:
    """_get_valid_user_cache_dir returns None when SEMBLE_CACHE_LOCATION is a relative path."""
    with patch.dict("os.environ", {"SEMBLE_CACHE_LOCATION": "relative/path"}):
        with patch("semble.cache.logger") as mock_logger:
            assert _get_valid_user_cache_dir() is None
        mock_logger.warning.assert_called_once()


def test_resolve_cache_folder_semble_cache_location(tmp_path: Path) -> None:
    """SEMBLE_CACHE_LOCATION takes precedence over all platform-specific helpers."""
    custom = tmp_path / "custom_cache"
    with patch.dict("os.environ", {"SEMBLE_CACHE_LOCATION": str(custom)}):
        result = resolve_cache_folder()
    assert result == custom
    assert custom.exists()


def test_clear_cache(tmp_path: Path) -> None:
    """clear_cache removes the index directory when it exists and is a no-op otherwise."""
    index_path = tmp_path / "index"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        clear_cache("/some/path")  # no-op: path doesn't exist yet
    index_path.mkdir()
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        clear_cache("/some/path")
    assert not index_path.exists()


def _write_metadata(
    path: Path,
    model_path: str,
    content_type: list[str],
    write_time: float,
    file_paths: list[str] | None = None,
    chunk_size: int | None = None,
) -> None:
    from semble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS

    path.mkdir(parents=True, exist_ok=True)
    (path / "chunks.json").write_text("[]")
    (path / "bm25_index").write_text("")
    (path / "semantic_index").write_text("")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "model_path": model_path,
                "content_type": content_type,
                "time": write_time,
                "file_paths": file_paths if file_paths is not None else [],
                "chunk_size": chunk_size if chunk_size is not None else _DESIRED_CHUNK_LENGTH_CHARS,
            }
        )
    )


def test_get_validated_cache_invalid_index(tmp_path: Path) -> None:
    """Returns None when the index directory is missing or incomplete."""
    with patch("semble.cache.find_index_from_cache_folder", return_value=tmp_path / "missing"):
        assert get_validated_cache("/path", None, [ContentType.CODE]) is None

    index_path = tmp_path / "index"
    index_path.mkdir()
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        assert get_validated_cache("/path", None, [ContentType.CODE]) is None


@pytest.mark.parametrize(
    ("stored_model", "stored_content", "req_model", "req_content"),
    [
        ("other/model", ["code"], "my/model", [ContentType.CODE]),  # model mismatch
        ("my/model", ["docs"], "my/model", [ContentType.CODE]),  # content mismatch
        ("my/model", ["unknown_type"], "my/model", [ContentType.CODE]),  # invalid content value
    ],
)
def test_get_validated_cache_metadata_mismatch(
    stored_model: str,
    stored_content: list[str],
    req_model: str,
    req_content: list[ContentType],
    tmp_path: Path,
) -> None:
    """Returns None when stored model or content type doesn't match the request."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, stored_model, stored_content, 0.0)
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        assert get_validated_cache("/path", req_model, req_content) is None


def test_get_validated_cache_reads_utf8_metadata_with_non_ascii_file_paths(tmp_path: Path) -> None:
    """Cache metadata is always UTF-8, even when the system default encoding is not."""
    from semble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS

    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)
    (index_path / "chunks.json").write_text("[]")
    (index_path / "bm25_index").write_text("")
    (index_path / "semantic_index").write_text("")

    non_ascii_path = "docs\\测试检查清单.md"
    with pytest.raises(UnicodeDecodeError):
        non_ascii_path.encode("utf-8").decode("cp936")

    (index_path / "metadata.json").write_text(
        json.dumps(
            {
                "model_path": "my/model",
                "content_type": ["docs"],
                "time": 0.0,
                "file_paths": [non_ascii_path],
                "chunk_size": _DESIRED_CHUNK_LENGTH_CHARS,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    real_open = builtins.open

    def open_with_cp936_default(file: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(file) == index_path / "metadata.json" and "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "cp936"
        return real_open(file, mode, *args, **kwargs)

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("builtins.open", side_effect=open_with_cp936_default):
            result = get_validated_cache("https://github.com/org/repo.git", "my/model", [ContentType.DOCS])
    assert result == index_path


def test_get_validated_cache_chunk_size_mismatch_returns_none(tmp_path: Path) -> None:
    """Cache built with a different chunk_size is not reused."""
    from semble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS

    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), chunk_size=_DESIRED_CHUNK_LENGTH_CHARS + 100)
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        assert get_validated_cache("/path", "my/model", [ContentType.CODE]) is None


def test_get_validated_cache_missing_chunk_size_returns_none(tmp_path: Path) -> None:
    """Old cache metadata without chunk_size field is not reused (transparent rebuild)."""
    index_path = tmp_path / "index"
    # Write metadata as old semble would — no chunk_size field
    index_path.mkdir(parents=True, exist_ok=True)
    (index_path / "chunks.json").write_text("[]")
    (index_path / "bm25_index").write_text("")
    (index_path / "semantic_index").write_text("")
    import json as _json

    (index_path / "metadata.json").write_text(
        _json.dumps(
            {
                "model_path": "my/model",
                "content_type": ["code"],
                "time": float("inf"),
                "file_paths": [],
            }
        )
    )
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        assert get_validated_cache("/path", "my/model", [ContentType.CODE]) is None


def test_get_validated_cache_legacy_metadata_returns_none(tmp_path: Path) -> None:
    """Old cache metadata missing content_type returns None instead of crashing."""
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True)
    (index_path / "chunks.json").write_text("[]")
    (index_path / "bm25_index").write_text("")
    (index_path / "semantic_index").write_text("")
    (index_path / "metadata.json").write_text(json.dumps({"model_path": "my/model", "time": 0.0}))
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        assert get_validated_cache("/path", "my/model", [ContentType.CODE]) is None


def test_get_validated_cache_resolves_default_model(tmp_path: Path) -> None:
    """When model_path is None, resolve_model_name() is used for comparison."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "default/model", ["code"], 0.0)
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.resolve_model_name", return_value="other/model"):
            assert get_validated_cache("/path", None, [ContentType.CODE]) is None


def test_get_validated_cache_git_url_returns_immediately(tmp_path: Path) -> None:
    """Git URL paths skip file-mtime checks and return the index path directly."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0)
    url = "https://github.com/org/repo.git"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(url, "my/model", [ContentType.CODE])
    assert result == index_path


@pytest.mark.parametrize(
    ("write_time", "walk_result", "write", "expected"),
    [
        (0.0, "stale", True, None),  # file newer than index → stale
        (float("inf"), [], True, "index"),  # no newer files → valid
        (float("inf"), "stale", False, None),  # no index, returns None
    ],
)
def test_get_validated_cache_mtime(
    write_time: float, walk_result: str | list, write: bool, expected: str | None, tmp_path: Path
) -> None:
    """Returns None when a tracked file is newer than the index; the path otherwise."""
    index_path = tmp_path / "index"
    stale_file = tmp_path / "src.py"
    stale_file.write_text("x = 1" if write else "")
    files = [stale_file] if walk_result == "stale" else walk_result
    # Include the file in stored manifest so manifest check passes and mtime check fires.
    stored_files = ["src.py"] if walk_result == "stale" else []
    _write_metadata(index_path, "my/model", ["code"], write_time, file_paths=stored_files)

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.get_extensions", return_value={".py"}):
            with patch("semble.cache.walk_files", return_value=files):
                result = get_validated_cache(str(tmp_path), "my/model", [ContentType.CODE])
    assert result == (index_path if expected == "index" else None)


@pytest.mark.parametrize(
    ("stored_files", "current_files"),
    [
        (["deleted.py"], []),  # file deleted since indexing
        ([], ["new.py"]),  # new file added since indexing
    ],
)
def test_get_validated_cache_manifest_mismatch(
    stored_files: list[str], current_files: list[str], tmp_path: Path
) -> None:
    """Returns None when the current file set differs from the stored manifest."""
    index_path = tmp_path / "index"
    walk_return = []
    for f in current_files:
        p = tmp_path / f
        # Make sure file is not empty
        p.write_text("a")
        walk_return.append(p)
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=stored_files)
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", return_value=walk_return):
            result = get_validated_cache(str(tmp_path), "my/model", [ContentType.CODE])
    assert result is None
