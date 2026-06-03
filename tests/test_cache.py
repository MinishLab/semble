from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import semble.cache as cache_module
from semble.cache import (
    _get_valid_user_cache_dir,
    _git_cache_heads_match,
    _git_cache_is_current,
    _linux_cache_dir,
    _windows_cache_dir,
    clear_cache,
    find_index_from_cache_folder,
    get_validated_cache,
    refresh_git_cache_metadata,
    resolve_cache_folder,
    save_index_to_cache,
)
from semble.types import ContentType

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "t@t.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "t@t.com",
}


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


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)


def _git_commit_all(path: Path, message: str = "commit") -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, env=_GIT_ENV)
    subprocess.run(["git", "-C", str(path), "commit", "-m", message], check=True, capture_output=True, env=_GIT_ENV)


def _git_head(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True)
    return result.stdout.decode().strip()


def _write_metadata(
    path: Path,
    model_path: str,
    content_type: list[str],
    write_time: float,
    file_paths: list[str] | None = None,
    git_roots: list[dict[str, str]] | None = None,
    git_roots_version: int | None = cache_module.GIT_CACHE_ROOTS_VERSION,
) -> None:
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
                **({"git_roots": git_roots} if git_roots is not None else {}),
                **(
                    {"git_roots_version": git_roots_version}
                    if git_roots is not None and git_roots_version is not None
                    else {}
                ),
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


def test_get_validated_cache_accepts_lmdb_chunk_store(tmp_path: Path) -> None:
    """New cache entries are valid when chunk payloads live in LMDB instead of chunks.json."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0)
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["chunk_ids"] = []
    metadata_path.write_text(json.dumps(metadata))
    (index_path / "chunks.json").unlink()
    (index_path / "chunks.lmdb").mkdir()

    url = "https://github.com/org/repo.git"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(url, "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_rejects_lmdb_store_id_mismatch(tmp_path: Path) -> None:
    """Metadata-bound LMDB stores should reject incomplete replacement saves."""
    from semble.index.chunk_store import LmdbChunkStore

    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0)
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["chunk_ids"] = []
    metadata["chunk_store_id"] = "expected-store"
    metadata_path.write_text(json.dumps(metadata))
    (index_path / "chunks.json").unlink()
    store = LmdbChunkStore.open(index_path / "chunks.lmdb")
    try:
        store.write_store_id("other-store")
    finally:
        store.close()

    url = "https://github.com/org/repo.git"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(url, "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_rejects_lmdb_without_chunk_ids(tmp_path: Path) -> None:
    """LMDB-only cache without chunk_ids cannot restore chunk order and is invalid."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0)
    (index_path / "chunks.json").unlink()
    (index_path / "chunks.lmdb").mkdir()

    url = "https://github.com/org/repo.git"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(url, "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_rejects_chunk_ids_without_lmdb(tmp_path: Path) -> None:
    """Metadata with chunk_ids requires the LMDB payload store, even if chunks.json remains."""
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0)
    metadata_path = index_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["chunk_ids"] = [1]
    metadata_path.write_text(json.dumps(metadata))

    url = "https://github.com/org/repo.git"
    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(url, "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_clean_git_repo_skips_full_walk(tmp_path: Path) -> None:
    """Clean git worktrees should validate hot cache without walking the filesystem tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch(
            "os.walk",
            side_effect=AssertionError("git cache validation should not os.walk"),
        ):
            with patch(
                "semble.cache.walk_files",
                side_effect=AssertionError("git cache validation should not full-walk"),
            ):
                result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_rejects_legacy_git_roots_version_with_stale_file_set(tmp_path: Path) -> None:
    """Legacy git-root metadata should not trust stored paths as the full file inventory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('stored')\n")
    (repo / "new.py").write_text("print('missed by old inventory')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
        git_roots_version=1,
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch(
            "semble.cache.walk_files",
            side_effect=AssertionError("git cache validation should stay git-backed"),
        ):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_git_metadata_ignores_sembleignored_untracked_files(tmp_path: Path) -> None:
    """Git metadata hot validation should not compare raw git files ignored by Semble rules."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    (repo / ".sembleignore").write_text("ignored.py\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    (repo / "ignored.py").write_text("print('ignored')\n")
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_git_metadata_trusts_clean_tracked_files_without_mtime(tmp_path: Path) -> None:
    """Git metadata hot validation should trust git status for clean tracked files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        0.0,
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_git_metadata_skips_source_root_discovery(tmp_path: Path) -> None:
    """Git metadata should provide hot validation source roots without scanning stored paths."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch(
            "semble.cache._git_cache_source_roots",
            side_effect=AssertionError("git metadata should avoid source root discovery"),
        ):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_git_metadata_avoids_full_untracked_status(tmp_path: Path) -> None:
    """Git metadata hot validation should use ls-files, not full untracked status scans."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache._run_git", wraps=cache_module._run_git) as run_git:
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path
    assert not any("--untracked-files=all" in call.args for call in run_git.mock_calls)


def test_get_validated_cache_git_metadata_rejects_head_change_without_full_walk(tmp_path: Path) -> None:
    """Git metadata hot validation should reject clean worktrees when HEAD changed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo, "initial")
    old_head = _git_head(repo)
    (repo / "new.py").write_text("print('new tracked')\n")
    _git_commit_all(repo, "new file")
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": old_head}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_cached_untracked_ignore_file_skips_full_walk(tmp_path: Path) -> None:
    """Cached untracked ignore rules should rely on mtime instead of permanent status invalidation."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    (repo / ".sembleignore").write_text("old-ignore\n")
    write_time = (repo / ".sembleignore").stat().st_mtime + 1.0
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], write_time, file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_cached_untracked_file_skips_full_walk(tmp_path: Path) -> None:
    """Cached untracked files should not keep invalidating hot cache forever."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    (repo / "scratch.py").write_text("print('cached untracked')\n")
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=["scratch.py", "src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_cached_dirty_tracked_file_skips_full_walk(tmp_path: Path) -> None:
    """Cached dirty tracked files should not keep invalidating hot cache forever."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src.py"
    source.write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    source.write_text("print('cached dirty tracked')\n")
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        source.stat().st_mtime + 1.0,
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_newer_dirty_tracked_file_returns_none_without_full_walk(tmp_path: Path) -> None:
    """Dirty tracked files newer than metadata should invalidate hot cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "src.py"
    source.write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    source.write_text("print('newer dirty tracked')\n")
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        0.0,
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_git_cache_head_validation_checks_source_roots_concurrently(tmp_path: Path) -> None:
    """Multi-root HEAD validation should not serialize independent git root checks."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    ready = threading.Event()
    lock = threading.Lock()
    entered = 0

    def fake_git_head(source_root: Path) -> str:
        nonlocal entered
        with lock:
            entered += 1
            if entered == 2:
                ready.set()
        if not ready.wait(0.2):
            return "timeout"
        return f"head-{source_root.name}"

    with patch("semble.cache._git_head", side_effect=fake_git_head):
        result = _git_cache_heads_match(
            [(root_a, "a"), (root_b, "b")],
            [{"path": "a", "head": "head-a"}, {"path": "b", "head": "head-b"}],
        )

    assert result is True


def test_git_cache_head_validation_rejects_empty_source_roots() -> None:
    """Empty root metadata is invalid and should not create a zero-worker executor."""
    assert _git_cache_heads_match([], []) is False


def test_get_validated_cache_rejects_incomplete_git_metadata_when_nested_head_changes(tmp_path: Path) -> None:
    """Incomplete git metadata should not hide new files in previously empty submodules."""
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    (repo / "src.py").write_text("print('root')\n")
    (nested / "README.txt").write_text("not indexed as code\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
        git_roots_version=None,
    )
    (nested / "new.py").write_text("print('new')\n")
    _git_commit_all(nested, "add indexed file")

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_build_git_cache_metadata_includes_empty_nested_git_roots(tmp_path: Path) -> None:
    """Metadata should track nested git roots even before they have cached files."""
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    (repo / "src.py").write_text("print('root')\n")
    (nested / "README.txt").write_text("not indexed as code\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    _init_git_repo(repo)
    _git_commit_all(repo)

    result = cache_module.build_git_cache_metadata(repo, ["src.py"])

    assert result == [{"path": "", "head": _git_head(repo)}, {"path": "nested", "head": _git_head(nested)}]


def test_refresh_git_cache_metadata_restores_empty_tracked_gitlink_roots(tmp_path: Path) -> None:
    """Refreshing plan metadata should recover tracked empty git roots before saving."""
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    (repo / "src.py").write_text("print('root')\n")
    (nested / "README.txt").write_text("not indexed as code\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    _init_git_repo(repo)
    _git_commit_all(repo)

    result = refresh_git_cache_metadata(repo, [{"path": "", "head": "old-head"}])

    assert result == [{"path": "", "head": _git_head(repo)}, {"path": "nested", "head": _git_head(nested)}]


def test_git_cache_validation_checks_source_roots_concurrently(tmp_path: Path) -> None:
    """Multi-root hot validation should not serialize independent git root checks."""
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    (repo / "src.py").write_text("print('root')\n")
    (nested / "lib.py").write_text("print('nested')\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    _init_git_repo(repo)
    _git_commit_all(repo)
    ready = threading.Event()
    lock = threading.Lock()
    entered = 0

    def fake_root_files(
        display_root: Path,
        source_root: Path,
        source_rel: str,
        child_roots: list[str],
        extensions: set[str],
        write_time: float,
        stored_files: set[str],
        trust_git_heads: bool,
    ) -> tuple[bool, list[str]]:
        nonlocal entered
        with lock:
            entered += 1
            if entered == 2:
                ready.set()
        if not ready.wait(0.2):
            return False, []
        return True, [path if source_rel == "" else f"{source_rel}/{path}" for path in stored_files]

    with patch("semble.cache._git_cache_root_files", side_effect=fake_root_files):
        result = _git_cache_is_current(
            repo,
            {".py"},
            ["nested/lib.py", "src.py"],
            float("inf"),
            git_roots=[{"path": "", "head": _git_head(repo)}, {"path": "nested", "head": _git_head(nested)}],
        )

    assert result is True


def test_git_cache_root_files_ignores_submodules_in_status(tmp_path: Path) -> None:
    """Parent root status should not rescan child git roots that are validated separately."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('root')\n")
    calls = []

    def fake_run_git(cwd: Path, *args: str) -> bytes:
        calls.append(args)
        if args[0] == "status":
            return b""
        if args == ("ls-files", "-z", "-c"):
            return b"src.py\0"
        if args == ("ls-files", "-z", "-o", "--exclude-standard"):
            return b""
        if args[:5] == ("ls-files", "-z", "-o", "-i", "--exclude-standard"):
            return b""
        raise AssertionError(args)

    with patch("semble.cache._run_git", side_effect=fake_run_git):
        result = cache_module._git_cache_root_files(repo, repo, "", ["nested"], {".py"}, 0.0, {"src.py"}, True)

    assert result == (True, ["src.py"])
    assert ("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--ignore-submodules=all") in calls


def test_git_cache_root_files_clean_head_skips_file_listing(tmp_path: Path) -> None:
    """Clean HEAD-matched roots should trust stored files without listing every file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_run_git(cwd: Path, *args: str) -> bytes:
        calls.append(args)
        if args[0] == "status":
            return b""
        if args == ("ls-files", "-z", "-c"):
            return b"src.py\0"
        if args[:5] == ("ls-files", "-z", "-o", "-i", "--exclude-standard"):
            return b""
        raise AssertionError(args)

    with patch("semble.cache._run_git", side_effect=fake_run_git):
        result = cache_module._git_cache_root_files(repo, repo, "", [], {".py"}, 0.0, {"src.py"}, True)

    assert result == (True, ["src.py"])
    assert ("ls-files", "-z", "-c") in calls
    assert ("ls-files", "-z", "-o", "--exclude-standard") not in calls


def test_git_cache_root_files_clean_head_checks_ignored_controls(tmp_path: Path) -> None:
    """Clean HEAD fast-path must still invalidate newer ignored control files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    ignore_file = repo / ".sembleignore"
    ignore_file.write_text("generated.py\n")
    calls = []

    def fake_run_git(cwd: Path, *args: str) -> bytes:
        calls.append(args)
        if args[0] == "status":
            return b""
        if args == ("ls-files", "-z", "-c"):
            return b"src.py\0"
        if args[:5] == ("ls-files", "-z", "-o", "-i", "--exclude-standard"):
            return b".sembleignore\0"
        raise AssertionError(args)

    with patch("semble.cache._run_git", side_effect=fake_run_git):
        result = cache_module._git_cache_root_files(repo, repo, "", [], {".py"}, 0.0, {"src.py"}, True)

    assert result == (False, [])


def test_get_validated_cache_missing_stored_untracked_file_returns_none(tmp_path: Path) -> None:
    """Deleted untracked files from a prior cache should invalidate instead of crashing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('tracked')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["src.py", "probe.py"],
        git_roots=[{"path": "", "head": _git_head(repo)}],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_nested_git_repo_skips_full_walk(tmp_path: Path) -> None:
    """Nested git worktrees should validate hot cache without recursive filesystem walks."""
    repo = tmp_path / "repo"
    nested = repo / "services" / "service"
    nested.mkdir(parents=True)
    (repo / "src.py").write_text("print('root')\n")
    (nested / "lib.py").write_text("print('nested')\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(
        index_path,
        "my/model",
        ["code"],
        float("inf"),
        file_paths=["services/service/lib.py", "src.py"],
    )

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch(
            "os.walk",
            side_effect=AssertionError("git cache validation should not os.walk"),
        ):
            with patch(
                "semble.cache.walk_files",
                side_effect=AssertionError("git cache validation should not full-walk"),
            ):
                result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_clean_git_repo_checks_mtime_without_full_walk(tmp_path: Path) -> None:
    """Clean git worktrees still reject files newer than the cache metadata."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0, file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_new_nested_git_repo_returns_none_without_full_walk(tmp_path: Path) -> None:
    """New nested git worktrees should invalidate hot cache even when parent git reports a directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.txt").write_text("not code\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    nested = repo / "service"
    nested.mkdir()
    (nested / "new.py").write_text("print('new nested')\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=[])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_new_nested_git_repo_without_code_keeps_hot_cache(tmp_path: Path) -> None:
    """New nested git worktrees without indexable files should not invalidate hot cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    nested = repo / "scratch-worktree"
    nested.mkdir()
    (nested / "README.txt").write_text("not indexed as code\n")
    _init_git_repo(nested)
    _git_commit_all(nested)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result == index_path


def test_get_validated_cache_nested_git_repo_with_negated_file_returns_none_without_full_walk(tmp_path: Path) -> None:
    """Nested git dirs with Semble-included non-extension files must invalidate hot cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    nested = repo / "scratch-worktree"
    nested.mkdir()
    (nested / ".gitignore").write_text("*\n")
    (nested / ".sembleignore").write_text("!special.kjs\n")
    (nested / "special.kjs").write_text("print('included by Semble negation')\n")
    _init_git_repo(nested)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_ignored_sembleignore_returns_none_without_full_walk(tmp_path: Path) -> None:
    """Ignored Semble control files should still invalidate hot cache when newer than metadata."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    (repo / ".gitignore").write_text(".sembleignore\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    write_time = (repo / "src.py").stat().st_mtime + 1.0
    (repo / ".sembleignore").write_text("src.py\n")
    os.utime(repo / ".sembleignore", (write_time + 1.0, write_time + 1.0))
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], write_time, file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


def test_get_validated_cache_dirty_git_repo_returns_none_without_full_walk(tmp_path: Path) -> None:
    """Dirty git worktrees should invalidate hot cache without walking the filesystem tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    (repo / "new.py").write_text("print('new')\n")
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0, file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


@pytest.mark.parametrize("ignore_file", [".gitignore", ".sembleignore"])
def test_get_validated_cache_dirty_ignore_file_returns_none_without_full_walk(ignore_file: str, tmp_path: Path) -> None:
    """Dirty ignore rules should invalidate hot cache because they can change the indexed file set."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    (repo / ignore_file).write_text("old-ignore\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    (repo / ignore_file).write_text("src.py\n")
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], float("inf"), file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


@pytest.mark.parametrize("ignore_file", [".gitignore", ".sembleignore"])
def test_get_validated_cache_clean_git_repo_checks_ignore_file_mtime_without_full_walk(
    ignore_file: str, tmp_path: Path
) -> None:
    """Clean git worktrees still reject ignore rule files newer than the cache metadata."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("print('clean')\n")
    (repo / ignore_file).write_text("src.py\n")
    _init_git_repo(repo)
    _git_commit_all(repo)
    index_path = tmp_path / "index"
    _write_metadata(index_path, "my/model", ["code"], 0.0, file_paths=["src.py"])

    with patch("semble.cache.find_index_from_cache_folder", return_value=index_path):
        with patch("semble.cache.walk_files", side_effect=AssertionError("git cache validation should not full-walk")):
            result = get_validated_cache(str(repo), "my/model", [ContentType.CODE])

    assert result is None


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
