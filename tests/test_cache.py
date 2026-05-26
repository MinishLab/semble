from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from semble.cache import (
    _linux_cache_dir,
    _windows_cache_dir,
    clear_cache,
    find_index_from_cache_folder,
    get_validated_cache,
    resolve_cache_folder,
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


def test_resolve_cache_folder_windows() -> None:
    """resolve_cache_folder calls _windows_cache_dir on win32."""
    with patch.object(sys, "platform", "win32"):
        with patch("semble.cache._windows_cache_dir", return_value=Path("/win")) as mock_win:
            with patch("semble.cache._linux_cache_dir", return_value=Path("/linux")) as mock_linux:
                with patch("pathlib.Path.mkdir"):
                    result = resolve_cache_folder()
    mock_win.assert_called_once_with("semble")
    mock_linux.assert_called_once_with("semble")
    assert result == Path("/linux")


def test_resolve_cache_folder_linux() -> None:
    """resolve_cache_folder calls _linux_cache_dir on non-darwin platforms."""
    with patch.object(sys, "platform", "linux"):
        with patch("semble.cache._linux_cache_dir", return_value=Path("/linux")) as mock_linux:
            with patch("pathlib.Path.mkdir"):
                result = resolve_cache_folder()
    mock_linux.assert_called_once_with("semble")
    assert result == Path("/linux")


def test_clear_cache(tmp_path: Path) -> None:
    """clear_cache removes the index directory when it exists and is a no-op otherwise."""
    index_path = tmp_path / "index"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        clear_cache("/some/path")  # no-op: path doesn't exist yet
    index_path.mkdir()
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        clear_cache("/some/path")
    assert not index_path.exists()


def _write_metadata(path: Path, model_path: str, content_type: list[str], write_time: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "chunks.json").write_text("[]")
    (path / "bm25_index").write_text("")
    (path / "semantic_index").write_text("")
    (path / "metadata.json").write_text(
        json.dumps({"model_path": model_path, "content_type": content_type, "time": write_time})
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
    ("write_time", "walk_result", "expected"),
    [
        (0.0, "stale", None),  # file newer than index → stale
        (float("inf"), [], "index"),  # no newer files → valid
    ],
)
def test_get_validated_cache_mtime(
    write_time: float, walk_result: str | list, expected: str | None, tmp_path: Path
) -> None:
    """Returns None when a tracked file is newer than the index; the path otherwise."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], write_time)
    stale_file = tmp_path / "src.py"
    stale_file.write_text("x = 1")
    files = [stale_file] if walk_result == "stale" else walk_result

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.get_extensions", return_value={".py"}):
            with patch("semble.cache.walk_files", return_value=files):
                result = get_validated_cache(str(tmp_path), "my/model", [ContentType.CODE])
    assert result == (index_path if expected == "index" else None)
