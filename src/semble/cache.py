import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from semble.index.file_walker import walk_files
from semble.index.files import get_extensions
from semble.index.types import PersistencePath
from semble.types import ContentType
from semble.utils import is_git_url, resolve_model_name


def find_index_from_cache_folder(path: str) -> Path:
    """Finds an index from a cache folder and a project path."""
    if is_git_url(path):
        data = path.encode("utf-8")
    else:
        normalized = Path(path).expanduser().resolve()
        data = str(normalized).encode("utf-8")
    subdir_path = hashlib.new("sha256", data).hexdigest()
    cache_dir = resolve_cache_folder() / subdir_path
    return cache_dir / "index"


def _windows_cache_dir(name: str) -> Path:
    """Get the default windows cache dir."""
    env_base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    base = Path(env_base) if env_base is not None else Path.home() / "AppData" / "Local"
    return base / name / "Cache"


def _macos_cache_dir(name: str) -> Path:
    """Get the default macOS cache dir."""
    return Path.home() / "Library" / "Caches" / name


def _linux_cache_dir(name: str) -> Path:
    """Get the default Linux cache dir."""
    env_base = os.getenv("XDG_CACHE_HOME")
    base = Path(env_base) if env_base else Path.home() / ".cache"
    return base / name


def resolve_cache_folder() -> Path:
    """Resolves a cache folder, respects XDG_CACHE_HOME."""
    name = "semble"
    if sys.platform == "win32":
        cache_dir = _windows_cache_dir(name)
    if sys.platform == "darwin":
        cache_dir = _macos_cache_dir(name)
    else:
        cache_dir = _linux_cache_dir(name)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache(path: str) -> None:
    """Clears the cache for the given path."""
    index_path = find_index_from_cache_folder(path)
    if index_path and index_path.exists():
        shutil.rmtree(index_path)


def get_validated_cache(path: str, model_path: str | None, content: Sequence[ContentType]) -> Path | None:
    """Validates the cache folder and returns the index path."""
    index_path = find_index_from_cache_folder(path)
    if not index_path.exists():
        return None

    persistence_path = PersistencePath.from_path(index_path)
    if persistence_path.non_existing():
        return None

    with open(persistence_path.metadata) as f:
        metadata = json.load(f)
    model_path_from_index = metadata["model_path"]
    if model_path is None:
        model_path = resolve_model_name()
    if model_path_from_index != model_path:
        return None

    content_type_strings: list[str] = metadata["content_type"]

    content_type = tuple(ContentType(string) for string in content_type_strings)
    if set(content_type) != set(content):
        return None

    if is_git_url(str(path)):
        return index_path

    write_time = metadata["time"]
    extensions = get_extensions(content_type, None)

    path_as_path = Path(path)
    for file_path in walk_files(path_as_path, extensions=extensions):
        st = file_path.stat()
        if st.st_mtime > write_time:
            return None

    return index_path
