import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pathspec import GitIgnoreSpec

from semble.index.types import PersistencePath
from semble.types import ContentType
from semble.utils import is_git_url, resolve_model_name

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from semble.index import SembleIndex


GIT_CACHE_ROOTS_VERSION = 2


def _get_extensions(content: Sequence[ContentType]) -> tuple[str, ...]:
    from semble.index.files import get_extensions as cold_get_extensions

    return cold_get_extensions(content)


def _get_file_status(file_path: Path, write_time: float | None = None) -> object:
    from semble.index.files import get_file_status as cold_get_file_status

    return cold_get_file_status(file_path, write_time)


def _file_status_names() -> tuple[object, object]:
    from semble.index.files import FileStatus

    return FileStatus.NEWER, FileStatus.VALID


def _walk_files(path: Path, extensions: Sequence[str]) -> Any:
    from semble.index.file_walker import walk_files as cold_walk_files

    return cold_walk_files(path, extensions=extensions)


get_extensions = _get_extensions
get_file_status = _get_file_status
walk_files = _walk_files


def _default_ignore_spec(source_root: Path) -> Any:
    from semble.index.file_walker import _DEFAULT_IGNORED_DIRS, IgnoreSpec

    return IgnoreSpec(source_root, GitIgnoreSpec.from_lines(sorted(_DEFAULT_IGNORED_DIRS), backend="simple"))


def _is_ignored_path(path: Path, specs: Sequence[Any]) -> tuple[bool, Any | None]:
    from semble.index.file_walker import _is_ignored

    return _is_ignored(path, specs)


def _load_ignore_for_dir(path: Path) -> GitIgnoreSpec | None:
    from semble.index.file_walker import _load_ignore_for_dir

    return _load_ignore_for_dir(path)


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


def index_needs_cache_save(index: "SembleIndex") -> bool:
    """Return whether this index still needs to be written to the shared cache."""
    return not index.loaded_from_disk and index.__dict__.get("_persisted_cache_path") is None


def save_index_to_cache(index: "SembleIndex", path: str) -> None:
    """Save an index to the cache folder if it was freshly built."""
    if index_needs_cache_save(index):
        index.save(find_index_from_cache_folder(path))


def _metadata_matches(metadata: dict, model_path: str, content: Sequence[ContentType]) -> bool:
    """Return True if the stored metadata is compatible with the requested parameters."""
    try:
        content_type = tuple(ContentType(s) for s in metadata["content_type"])
        return metadata["model_path"] == model_path and set(content_type) == set(content)
    except (KeyError, ValueError):
        return False


def _has_chunk_payloads(persistence_path: PersistencePath, metadata: dict | None = None) -> bool:
    if metadata is None:
        return persistence_path.chunks.exists() or persistence_path.chunk_store.exists()
    if "chunk_ids" in metadata:
        return persistence_path.chunk_store.exists()
    return persistence_path.chunks.exists()


def build_git_cache_metadata(path: Path, stored_files: Sequence[str]) -> list[dict[str, str]] | None:
    """Build git HEAD metadata for cache validation."""
    source_roots = _git_cache_source_roots(path, list(stored_files))
    if source_roots is None:
        return None

    git_roots = []
    for source_root, source_rel in source_roots:
        head = _git_head(source_root)
        if head is None:
            return None
        git_roots.append({"path": source_rel, "head": head})
    return git_roots


def _git_cache_is_current(
    path: Path,
    extensions: set[str],
    stored_files: list[str],
    write_time: float,
    git_roots: list[dict[str, str]] | None = None,
    git_roots_version: int | None = None,
) -> bool | None:
    git_roots_are_current = git_roots_version == GIT_CACHE_ROOTS_VERSION
    source_roots = None
    if git_roots_are_current:
        source_roots = _git_cache_source_roots_from_metadata(path, git_roots)
    if source_roots is None:
        source_roots = _git_cache_source_roots(path, stored_files)
        if source_roots is None:
            return None

    heads_match = _git_cache_heads_match(source_roots, git_roots)
    if heads_match is False:
        return False
    trust_git_heads = git_roots_are_current and heads_match is True

    root_args = []
    for source_root, source_rel in source_roots:
        child_roots = _child_git_root_paths(source_roots, source_rel)
        root_stored_paths = _stored_paths_in_root(stored_files, source_rel)
        stored_root_files = set(_local_git_paths_outside_children(root_stored_paths, child_roots))
        root_args.append((source_root, source_rel, child_roots, stored_root_files))

    def validate_root(args: tuple[Path, str, list[str], set[str]]) -> tuple[bool, list[str]] | None:
        source_root, source_rel, child_roots, stored_root_files = args
        return _git_cache_root_files(
            path,
            source_root,
            source_rel,
            child_roots,
            extensions,
            write_time,
            stored_root_files,
            trust_git_heads,
        )

    current_files = []
    with ThreadPoolExecutor(max_workers=min(32, len(root_args))) as executor:
        root_results = list(executor.map(validate_root, root_args))
    for root_result in root_results:
        if root_result is None:
            return None
        root_is_current, root_files = root_result
        if not root_is_current:
            return False
        current_files.extend(root_files)
    return set(current_files) == set(stored_files)


def _git_cache_root_files(
    display_root: Path,
    source_root: Path,
    source_rel: str,
    child_roots: list[str],
    extensions: set[str],
    write_time: float,
    stored_files: set[str],
    trust_git_heads: bool,
) -> tuple[bool, list[str]] | None:
    status = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--ignore-submodules=all",
    )
    if status is None:
        return None
    status_items = _local_git_status_items_outside_children(_git_status_items(status), child_roots)
    if any(
        _is_dirty_cache_affecting_status(status, git_path, extensions, source_root, stored_files)
        for status, git_path in status_items
    ):
        return False, []

    listing = _git_cache_listing(source_root, child_roots, extensions, stored_files, trust_git_heads)
    if listing is None:
        return None
    listing_is_current, ls_files, tracked_paths = listing
    if not listing_is_current:
        return False, []

    ignored_controls = _run_git(
        source_root,
        "ls-files",
        "-z",
        "-o",
        "-i",
        "--exclude-standard",
        "--",
        ".gitignore",
        ".sembleignore",
        ":(glob)**/.gitignore",
        ":(glob)**/.sembleignore",
    )
    if ignored_controls is None:
        return None

    controls_are_current = _git_cache_controls_are_current(
        display_root,
        source_rel,
        child_roots,
        write_time,
        ls_files,
        ignored_controls,
    )
    if not controls_are_current:
        return False, []

    candidate_paths = sorted(stored_files) if trust_git_heads else _git_ls_files_paths(ls_files, extensions)
    return _git_cache_current_files(
        display_root,
        source_rel,
        child_roots,
        write_time,
        candidate_paths,
        tracked_paths,
        trust_git_heads,
    )


def _git_cache_controls_are_current(
    display_root: Path,
    source_rel: str,
    child_roots: list[str],
    write_time: float,
    ls_files: bytes,
    ignored_controls: bytes,
) -> bool:
    control_paths = _git_cache_control_paths(ls_files) + _git_cache_control_paths(ignored_controls)
    for file_path in _local_git_paths_outside_children(control_paths, child_roots):
        global_path = _join_git_path(source_rel, file_path)
        if (display_root / global_path).stat().st_mtime > write_time:
            return False
    return True


def _git_cache_current_files(
    display_root: Path,
    source_rel: str,
    child_roots: list[str],
    write_time: float,
    candidate_paths: list[str],
    tracked_paths: set[str],
    trust_git_heads: bool,
) -> tuple[bool, list[str]]:
    current_files = []
    for file_path in _local_git_paths_outside_children(candidate_paths, child_roots):
        global_path = _join_git_path(source_rel, file_path)
        if trust_git_heads and file_path in tracked_paths:
            current_files.append(global_path)
            continue
        newer_status, valid_status = _file_status_names()
        file_status = get_file_status(display_root / global_path, write_time)
        if file_status == newer_status:
            return False, []
        if file_status == valid_status:
            current_files.append(global_path)
    return True, current_files


def _git_cache_listing(
    source_root: Path,
    child_roots: list[str],
    extensions: set[str],
    stored_files: set[str],
    trust_git_heads: bool,
) -> tuple[bool, bytes, set[str]] | None:
    if not trust_git_heads:
        ls_files = _run_git(source_root, "ls-files", "-z", "-c", "-o", "--exclude-standard")
        return None if ls_files is None else (True, ls_files, set())

    tracked_files = _run_git(source_root, "ls-files", "-z", "-c")
    untracked_files = _run_git(source_root, "ls-files", "-z", "-o", "--exclude-standard")
    if tracked_files is None or untracked_files is None:
        return None

    untracked_items = _local_git_paths_outside_children(_git_output_paths(untracked_files), child_roots)
    if any(
        _is_dirty_cache_affecting_status("??", git_path, extensions, source_root, stored_files)
        for git_path in untracked_items
    ):
        return False, b"", set()
    return True, tracked_files + untracked_files, set(_git_output_paths(tracked_files))


def _local_git_paths_outside_children(paths: list[str], child_roots: list[str]) -> list[str]:
    return [path for path in paths if not _is_under_git_root(path, child_roots)]


def _local_git_status_items_outside_children(
    items: list[tuple[str, str]], child_roots: list[str]
) -> list[tuple[str, str]]:
    return [(status, path) for status, path in items if not _is_under_git_root(path, child_roots)]


def _git_cache_source_roots_from_metadata(
    path: Path, git_roots: list[dict[str, str]] | None
) -> list[tuple[Path, str]] | None:
    if git_roots is None:
        return None

    root = path.resolve()
    source_roots = []
    seen_paths = set()
    for item in git_roots:
        source_rel = str(item.get("path", "")).strip("/")
        source_root = root if source_rel == "" else root / source_rel
        resolved_source_root = source_root.resolve()
        try:
            resolved_source_root.relative_to(root)
        except ValueError:
            return None
        if resolved_source_root in seen_paths or not (resolved_source_root / ".git").exists():
            return None
        seen_paths.add(resolved_source_root)
        source_roots.append((resolved_source_root, source_rel))
    return sorted(source_roots, key=lambda item: (len(item[1]), item[1]))


def _git_cache_source_roots(path: Path, stored_files: list[str]) -> list[tuple[Path, str]] | None:
    if not (path / ".git").exists():
        return None

    root = path.resolve()
    roots = {root: ""}
    seen_dirs: set[Path] = set()
    for file_path in stored_files:
        current = (root / file_path).parent
        while True:
            try:
                current.relative_to(root)
            except ValueError:
                break
            if current not in seen_dirs:
                seen_dirs.add(current)
                if (current / ".git").exists():
                    roots[current.resolve()] = "" if current == root else current.relative_to(root).as_posix()
            if current == root:
                break
            current = current.parent
    if not _add_tracked_gitlink_roots(roots):
        return None
    return sorted(roots.items(), key=lambda item: (len(item[1]), item[1]))


def _add_tracked_gitlink_roots(roots: dict[Path, str]) -> bool:
    source_roots = list(roots.items())
    index = 0
    while index < len(source_roots):
        source_root, source_rel = source_roots[index]
        index += 1
        output = _run_git(source_root, "ls-files", "-z", "--stage")
        if output is None:
            return False
        for git_path in _gitlink_paths(output):
            child_root = source_root / git_path
            if not (child_root / ".git").exists():
                continue
            child_rel = _join_git_path(source_rel, git_path)
            resolved_child = child_root.resolve()
            if resolved_child in roots:
                continue
            roots[resolved_child] = child_rel
            source_roots.append((resolved_child, child_rel))
    return True


def _git_cache_heads_match(source_roots: list[tuple[Path, str]], git_roots: list[dict[str, str]] | None) -> bool | None:
    if git_roots is None:
        return None

    expected = {str(item.get("path", "")): str(item.get("head", "")) for item in git_roots}
    if not source_roots or set(expected) != {source_rel for _, source_rel in source_roots}:
        return False

    with ThreadPoolExecutor(max_workers=min(32, len(source_roots))) as executor:
        heads = list(executor.map(lambda root: _git_head(root[0]), source_roots))
    if any(head is None for head in heads):
        return None
    return all(head == expected[source_rel] for head, (_, source_rel) in zip(heads, source_roots))


def _git_head(source_root: Path) -> str | None:
    result = _run_git(source_root, "rev-parse", "HEAD")
    if result is None:
        return None
    return result.decode("utf-8", errors="surrogateescape").strip()


def _stored_paths_in_root(stored_files: list[str], source_rel: str) -> list[str]:
    if source_rel == "":
        return stored_files
    prefix = f"{source_rel}/"
    return [path[len(prefix) :] for path in stored_files if path.startswith(prefix)]


def _child_git_root_paths(source_roots: list[tuple[Path, str]], source_rel: str) -> list[str]:
    children = []
    prefix = "" if source_rel == "" else f"{source_rel}/"
    for _, rel_path in source_roots:
        if rel_path == source_rel:
            continue
        if source_rel != "" and not rel_path.startswith(prefix):
            continue
        children.append(rel_path if source_rel == "" else rel_path[len(prefix) :])
    return children


def _is_under_git_root(path: str, git_roots: list[str]) -> bool:
    return any(_is_same_or_child(path, git_root) for git_root in git_roots)


def _is_same_or_child(path: str, parent: str) -> bool:
    path = path.rstrip("/")
    parent = parent.rstrip("/")
    return path == parent or path.startswith(f"{parent}/")


def _join_git_path(prefix: str, path: str) -> str:
    return path if prefix == "" else f"{prefix}/{path}"


def _run_git(cwd: Path, *args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_status_items(output: bytes) -> list[tuple[str, str]]:
    items = []
    entries = output.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        status = entry[:2].decode("ascii", errors="ignore")
        items.append((status, entry[3:].decode("utf-8", errors="surrogateescape")))
        if status[0] in {"R", "C"} and index < len(entries):
            items.append((status, entries[index].decode("utf-8", errors="surrogateescape")))
            index += 1
    return items


def _git_ls_files_paths(output: bytes, extensions: set[str]) -> list[str]:
    return [path for path in _git_output_paths(output) if Path(path).suffix.lower() in extensions]


def _git_output_paths(output: bytes) -> list[str]:
    paths = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        paths.append(raw_path.decode("utf-8", errors="surrogateescape"))
    return paths


def _gitlink_paths(output: bytes) -> list[str]:
    paths = []
    for entry in output.split(b"\0"):
        if not entry.startswith(b"160000 "):
            continue
        _, _, raw_path = entry.partition(b"\t")
        if raw_path:
            paths.append(raw_path.decode("utf-8", errors="surrogateescape"))
    return paths


def _git_cache_control_paths(output: bytes) -> list[str]:
    paths = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if _is_ignore_file(path):
            paths.append(path)
    return paths


def _is_ignore_file(path: str) -> bool:
    return path in {".gitignore", ".sembleignore"} or path.endswith(("/.gitignore", "/.sembleignore"))


def _is_cache_affecting_path(path: str, extensions: set[str], source_root: Path) -> bool:
    return _is_ignore_file(path) or Path(path).suffix.lower() in extensions or _is_nested_git_dir(source_root, path)


def _is_dirty_cache_affecting_status(
    status: str, path: str, extensions: set[str], source_root: Path, stored_files: set[str]
) -> bool:
    if status == "??":
        if _is_semble_ignored_path(source_root, path):
            return False
        if _is_nested_git_dir(source_root, path):
            return True
        if _is_ignore_file(path) or path.rstrip("/") in stored_files:
            return False
        return Path(path).suffix.lower() in extensions
    return _is_cache_affecting_path(path, extensions, source_root)


def _is_nested_git_dir(source_root: Path, path: str) -> bool:
    candidate = source_root / path.rstrip("/")
    return candidate.is_dir() and (candidate / ".git").exists()


def _is_semble_ignored_path(source_root: Path, path: str) -> bool:
    ignored, _ = _is_ignored_path(source_root / path.rstrip("/"), _ignore_specs_for_path(source_root, path))
    return ignored


def _ignore_specs_for_path(source_root: Path, path: str) -> list[Any]:
    from semble.index.file_walker import IgnoreSpec

    specs = [_default_ignore_spec(source_root)]
    current = source_root
    root_spec = _load_ignore_for_dir(current)
    if root_spec is not None:
        specs.append(IgnoreSpec(current, root_spec))

    for part in Path(path.rstrip("/")).parts[:-1]:
        current = current / part
        if not current.is_dir():
            break
        spec = _load_ignore_for_dir(current)
        if spec is not None:
            specs.append(IgnoreSpec(current, spec))
    return specs


def get_rebuild_cache(path: str, model_path: str | None, content: Sequence[ContentType]) -> Path | None:
    """Return a compatible stale cache that can seed an incremental rebuild."""
    index_path = find_index_from_cache_folder(path)
    if not index_path.exists() or is_git_url(str(path)):
        return None

    persistence_path = PersistencePath.from_path(index_path)
    required = [persistence_path.semantic_index, persistence_path.metadata]
    if any(not item.exists() for item in required) or not _has_chunk_payloads(persistence_path):
        return None

    resolved_model_path = resolve_model_name() if model_path is None else model_path
    with open(persistence_path.metadata) as f:
        metadata = json.load(f)
    if not _metadata_matches(metadata, resolved_model_path, content):
        return None
    if not _has_chunk_payloads(persistence_path, metadata):
        return None
    return index_path


def _validated_cache_metadata(
    persistence_path: PersistencePath, model_path: str, content: Sequence[ContentType]
) -> dict | None:
    if persistence_path.non_existing():
        return None
    with open(persistence_path.metadata) as f:
        metadata = json.load(f)
    if not _metadata_matches(metadata, model_path, content):
        return None
    if not _has_chunk_payloads(persistence_path, metadata):
        return None
    return metadata


def get_validated_cache(path: str, model_path: str | None, content: Sequence[ContentType]) -> Path | None:
    """Validates the cache folder and returns the index path."""
    index_path = find_index_from_cache_folder(path)
    if not index_path.exists():
        return None

    persistence_path = PersistencePath.from_path(index_path)
    resolved_model_path = resolve_model_name() if model_path is None else model_path
    metadata = _validated_cache_metadata(persistence_path, resolved_model_path, content)
    if metadata is None:
        return None

    if is_git_url(str(path)):
        return index_path

    write_time = metadata["time"]
    extensions = get_extensions(content)

    path_as_path = Path(path).resolve()
    stored_files: list[str] = metadata.get("file_paths", [])
    git_cache_is_current = _git_cache_is_current(
        path_as_path,
        extensions,
        stored_files,
        write_time,
        metadata.get("git_roots"),
        metadata.get("git_roots_version"),
    )
    if git_cache_is_current is True:
        return index_path
    if git_cache_is_current is False:
        return None

    current_files = []
    for file_path in walk_files(path_as_path, extensions=extensions):
        newer_status, valid_status = _file_status_names()
        file_status = get_file_status(file_path, write_time)
        if file_status == newer_status:
            return None
        if file_status != valid_status:
            continue
        current_files.append(str(file_path.relative_to(path_as_path)))

    if set(current_files) != set(stored_files):
        return None

    return index_path
