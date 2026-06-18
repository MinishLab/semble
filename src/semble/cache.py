import hashlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from semble.index.file_walker import walk_files
from semble.index.files import FileStatus, get_extensions, get_file_status
from semble.index.types import PersistencePath
from semble.types import ContentType
from semble.utils import is_git_url, resolve_model_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from semble.index import SembleIndex


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


def _get_valid_user_cache_dir() -> Path | None:
    """Gets the user cache dir if it is set and is a valid path."""
    user_cache_location = os.getenv("SEMBLE_CACHE_LOCATION")
    if user_cache_location is None:
        return None
    user_cache_dir = Path(user_cache_location)
    if not user_cache_dir.is_absolute():
        logger.warning("SEMBLE_CACHE_LOCATION is not an absolute path: %s", user_cache_location)
        return None

    return user_cache_dir


def resolve_cache_folder() -> Path:
    """Resolves a cache folder, respects SEMBLE_CACHE_LOCATION (highest precedence), XDG_CACHE_HOME."""
    name = "semble"
    if user_cache_dir := _get_valid_user_cache_dir():
        cache_dir = user_cache_dir
    elif sys.platform == "win32":
        cache_dir = _windows_cache_dir(name)
    elif sys.platform == "darwin":
        cache_dir = _macos_cache_dir(name)
    else:
        cache_dir = _linux_cache_dir(name)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache(path: str) -> None:
    """Clears the cache for the given path."""
    index_path = find_index_from_cache_folder(path)
    if index_path.exists():
        shutil.rmtree(index_path)


def save_index_to_cache(index: "SembleIndex", path: str) -> None:
    """Save an index to the cache folder if it was freshly built."""
    if not index.loaded_from_disk:
        index.save(find_index_from_cache_folder(path))


def _metadata_matches(metadata: dict, model_path: str, content: Sequence[ContentType]) -> bool:
    """Return True if the stored metadata is compatible with the requested parameters."""
    from semble.chunking.chunking import _DESIRED_CHUNK_LENGTH_CHARS  # avoid circular import at module level

    try:
        content_type = tuple(ContentType(s) for s in metadata["content_type"])
        # chunk_size is absent in indexes built before this field was added; treat None as mismatch
        # so old caches are transparently rebuilt with the current chunk size.
        chunk_size_ok = metadata.get("chunk_size") == _DESIRED_CHUNK_LENGTH_CHARS
        return metadata["model_path"] == model_path and set(content_type) == set(content) and chunk_size_ok
    except (KeyError, ValueError):
        return False


def get_validated_cache(path: str, model_path: str | None, content: Sequence[ContentType]) -> Path | None:
    """Validates the cache folder and returns the index path."""
    index_path = find_index_from_cache_folder(path)
    if not index_path.exists():
        return None

    persistence_path = PersistencePath.from_path(index_path)
    if persistence_path.non_existing():
        return None

    if model_path is None:
        model_path = resolve_model_name()
    with open(persistence_path.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    if not _metadata_matches(metadata, model_path, content):
        return None

    if is_git_url(str(path)):
        return index_path

    write_time = metadata["time"]
    extensions = get_extensions(content)

    path_as_path = Path(path).resolve()
    stored_files: list[str] = metadata.get("file_paths", [])
    current_files = []
    for file_path in walk_files(path_as_path, extensions=extensions):
        file_status = get_file_status(file_path, write_time)
        if file_status == FileStatus.NEWER:
            return None
        if file_status != FileStatus.VALID:
            continue
        current_files.append(str(file_path.relative_to(path_as_path)))

    if set(current_files) != set(stored_files):
        return None

    return index_path
