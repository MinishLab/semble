from __future__ import annotations

import subprocess
import time
from collections.abc import Set as AbstractSet
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, cast

from pathspec import GitIgnoreSpec

from semble.concurrency import index_worker_quota, run_with_index_worker
from semble.index.file_walker import _DEFAULT_IGNORED_DIRS, IgnoreSpec, _is_ignored, _load_ignore_for_dir


@dataclass(frozen=True, slots=True)
class SourceRoot:
    path: Path
    rel_path: str
    has_git_marker: bool = True


@dataclass(frozen=True, slots=True)
class GitWalkPlan:
    current_paths: tuple[str, ...]
    changed_paths: frozenset[str]
    deleted_paths: frozenset[str]
    source_roots: tuple[SourceRoot, ...]
    git_cache_metadata: tuple[dict[str, str], ...] | None = None
    tracked_paths: frozenset[str] = frozenset()
    clean_tracked_blob_shas: dict[str, str] = field(default_factory=dict)
    stale_roots: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class IgnoreSpecCache:
    root: Path
    specs_by_dir: dict[Path, list[IgnoreSpec]]
    control_dirs: frozenset[Path] | None = None
    control_specs: dict[Path, IgnoreSpec | None] = field(default_factory=dict)


_SourceRootStatus = tuple[bytes | None, float, str | None]


@dataclass(frozen=True, slots=True)
class SourceRootScanResult:
    files: AbstractSet[str]
    tracked: AbstractSet[str]
    changed: AbstractSet[str]
    deleted: AbstractSet[str]
    clean_tracked_blob_shas: Mapping[str, str]
    stale_root: str | None
    status_time_s: float
    ls_files_time_s: float
    head: str | None

    def __post_init__(self) -> None:
        """Freeze mutable scan payloads and validate basic path-set invariants."""
        object.__setattr__(self, "files", frozenset(self.files))
        object.__setattr__(self, "tracked", frozenset(self.tracked))
        object.__setattr__(self, "changed", frozenset(self.changed))
        object.__setattr__(self, "deleted", frozenset(self.deleted))
        object.__setattr__(self, "clean_tracked_blob_shas", MappingProxyType(dict(self.clean_tracked_blob_shas)))
        if not self.tracked <= self.files:
            raise ValueError("Tracked source-root paths must be present in files")
        if not self.changed.isdisjoint(self.deleted):
            raise ValueError("Changed and deleted source-root paths must not overlap")
        if self.status_time_s < 0.0 or self.ls_files_time_s < 0.0:
            raise ValueError("Source-root scan timings must be non-negative")


_IGNORE_CONTROL_FILENAMES = (".gitignore", ".sembleignore")
_IGNORE_CONTROL_SUFFIXES = ("/.gitignore", "/.sembleignore")
_IGNORE_CONTROL_PATHS = (*_IGNORE_CONTROL_FILENAMES, ":(glob)**/.gitignore", ":(glob)**/.sembleignore")
_T = TypeVar("_T")
_R = TypeVar("_R")


def _concurrent_ordered_map(
    items: Sequence[_T],
    fn: Callable[[_T], _R],
    key: Callable[[_T], int] | None = None,
) -> list[_R]:
    if not items:
        return []
    indexed_items = list(enumerate(items))
    if key is not None:
        indexed_items.sort(key=lambda indexed_item: key(indexed_item[1]), reverse=True)
    sentinel: Any = object()
    results: list[_R | Any] = [sentinel] * len(items)
    with ThreadPoolExecutor(max_workers=min(index_worker_quota(), len(indexed_items))) as executor:
        futures = {executor.submit(run_with_index_worker, fn, item): index for index, item in indexed_items}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [cast(_R, result) for result in results]


def build_git_walk_plan(
    root: Path,
    extensions: Iterable[str],
    previous_paths: Iterable[str],
    previous_git_heads: Mapping[str, str] | None = None,
    previous_write_time: float | None = None,
    previous_tracked_paths: Iterable[str] | None = None,
) -> GitWalkPlan | None:
    """Return a git-aware file plan, or None when git cannot safely cover previous files."""
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    diagnostics: list[str] = []
    extensions_set = frozenset(extensions)

    previous_path_set = set(previous_paths)
    previous_tracked_path_set = None if previous_tracked_paths is None else set(previous_tracked_paths)
    display_root_resolved = root.resolve()

    started = time.perf_counter()
    ignore_specs = _load_ignore_specs(root)
    source_roots = _discover_source_roots(
        root,
        ignore_specs,
        previous_path_set,
        previous_git_heads,
        discover_nested=previous_git_heads is None,
    )
    timings["git_root_discovery_s"] = time.perf_counter() - started
    if not source_roots:
        return None
    if not _source_roots_cover_paths(source_roots, previous_path_set):
        return None

    status_results = _concurrent_ordered_map(source_roots, _source_root_status)
    if previous_git_heads is not None:
        started = time.perf_counter()
        expanded_source_roots = _discover_dirty_nested_source_roots(root, ignore_specs, source_roots, status_results)
        timings["git_root_discovery_s"] += time.perf_counter() - started
        if expanded_source_roots != source_roots:
            source_roots = expanded_source_roots
            if not _source_roots_cover_paths(source_roots, previous_path_set):
                return None
            status_results = _concurrent_ordered_map(source_roots, _source_root_status)

    current_paths: set[str] = set()
    tracked_paths: set[str] = set()
    changed_paths: set[str] = set()
    deleted_paths: set[str] = set()
    stale_roots: set[str] = set()
    clean_tracked_blob_shas: dict[str, str] = {}
    child_roots_by_parent = _child_roots_by_parent(source_roots)
    previous_paths_by_root = _previous_paths_by_source_root(previous_path_set, source_roots)
    parent_control_dirs_by_root, control_specs = _parent_ignore_control_dirs_by_source_root(
        source_roots, root, ignore_specs
    )
    base_ignore_specs = tuple(ignore_specs.specs_by_dir[ignore_specs.root])
    dirty_ignore_controls = _dirty_ignore_control_paths_by_source_root(
        source_roots, status_results, previous_write_time
    )

    scan_args = [
        (
            source_root,
            root,
            display_root_resolved,
            extensions_set,
            child_roots_by_parent.get(source_root.path, ()),
            previous_paths_by_root[source_root.path],
            previous_tracked_path_set,
            None if previous_git_heads is None else previous_git_heads.get(source_root.rel_path),
            status_results[index][2],
            parent_control_dirs_by_root.get(source_root.path, frozenset()),
            control_specs,
            base_ignore_specs,
            status_results[index][0],
            status_results[index][1],
            _has_dirty_inherited_ignore_control(source_root.rel_path, dirty_ignore_controls),
        )
        for index, source_root in enumerate(source_roots)
    ]
    scan_results = _concurrent_ordered_map(
        scan_args,
        lambda args: _scan_source_root(*args),
        key=lambda args: len(args[5]),
    )

    status_time = 0.0
    ls_files_time = 0.0
    broken_roots = 0
    root_heads: dict[str, str] = {}
    for source_root, scan_result in zip(source_roots, scan_results):
        status_time += scan_result.status_time_s
        ls_files_time += scan_result.ls_files_time_s
        current_paths.update(scan_result.files)
        tracked_paths.update(scan_result.tracked)
        changed_paths.update(scan_result.changed)
        deleted_paths.update(scan_result.deleted)
        clean_tracked_blob_shas.update(scan_result.clean_tracked_blob_shas)
        if scan_result.head is not None:
            root_heads[source_root.rel_path] = scan_result.head
        if scan_result.stale_root is not None:
            broken_roots += 1
            diagnostics.append(f"broken git root: {scan_result.stale_root or '.'}")
            stale_roots.add(scan_result.stale_root)

    timings["git_status_s"] = status_time
    timings["git_ls_files_s"] = ls_files_time
    current_paths.difference_update(deleted_paths)
    tracked_paths.intersection_update(current_paths)
    changed_paths.difference_update(deleted_paths)
    for file_path in set(clean_tracked_blob_shas) - (tracked_paths - changed_paths):
        del clean_tracked_blob_shas[file_path]

    timings["source_roots_seen"] = float(len(source_roots))
    timings["source_roots_broken"] = float(broken_roots)
    timings["source_roots_stale"] = float(len(stale_roots))

    no_longer_candidates = {
        path for path in previous_path_set - current_paths if not _is_under_any_stale_root(path, stale_roots)
    }
    deleted_paths.update(no_longer_candidates)
    changed_paths.update(current_paths - previous_path_set)
    changed_paths.intersection_update(current_paths)

    timings["files_seen"] = float(len(current_paths))
    timings["files_changed"] = float(len(changed_paths))
    timings["files_deleted"] = float(len(deleted_paths))
    timings["manifest_reconcile_s"] = time.perf_counter() - total_started

    return GitWalkPlan(
        current_paths=tuple(sorted(current_paths)),
        changed_paths=frozenset(changed_paths),
        deleted_paths=frozenset(deleted_paths),
        source_roots=source_roots,
        git_cache_metadata=_git_cache_metadata_from_heads(source_roots, root_heads),
        tracked_paths=frozenset(tracked_paths),
        clean_tracked_blob_shas=clean_tracked_blob_shas,
        stale_roots=tuple(sorted(stale_roots)),
        diagnostics=tuple(diagnostics),
        timings=timings,
    )


def _discover_source_roots(
    root: Path,
    ignore_specs: IgnoreSpecCache,
    previous_paths: set[str],
    previous_git_heads: Mapping[str, str] | None = None,
    *,
    discover_nested: bool = True,
) -> tuple[SourceRoot, ...]:
    roots: dict[Path, SourceRoot] = {}
    pending: list[SourceRoot] = []

    previous_root_paths = {rel_path.strip("/") for rel_path in previous_git_heads or ()}
    for path, rel_path in _initial_source_root_candidates(root, previous_paths, previous_git_heads):
        _add_source_root(
            roots,
            pending,
            root,
            ignore_specs,
            path,
            rel_path,
            allow_missing_marker=rel_path in previous_root_paths,
        )

    if not discover_nested:
        return tuple(sorted(roots.values(), key=lambda source_root: (len(source_root.rel_path), source_root.rel_path)))

    _discover_nested_roots(roots, pending, root, ignore_specs)
    return tuple(sorted(roots.values(), key=lambda source_root: (len(source_root.rel_path), source_root.rel_path)))


def _discover_nested_roots(
    roots: dict[Path, SourceRoot],
    pending: list[SourceRoot],
    root: Path,
    ignore_specs: IgnoreSpecCache,
) -> None:
    with ThreadPoolExecutor(max_workers=index_worker_quota()) as executor:
        futures = {
            executor.submit(run_with_index_worker, _nested_git_root_paths, source_root.path): source_root
            for source_root in pending
            if source_root.has_git_marker
        }
        pending.clear()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                source_root = futures.pop(future)
                for git_path in future.result():
                    source_rel = _join_git_path(source_root.rel_path, git_path)
                    added: list[SourceRoot] = []
                    _add_source_root(roots, added, root, ignore_specs, root / source_rel, source_rel)
                    for added_root in added:
                        if added_root.has_git_marker:
                            futures[executor.submit(run_with_index_worker, _nested_git_root_paths, added_root.path)] = (
                                added_root
                            )


def _discover_dirty_nested_source_roots(
    root: Path,
    ignore_specs: IgnoreSpecCache,
    source_roots: tuple[SourceRoot, ...],
    status_results: list[_SourceRootStatus],
) -> tuple[SourceRoot, ...]:
    roots = {source_root.path: source_root for source_root in source_roots}
    pending: list[SourceRoot] = []
    for source_root, (status, _, _) in zip(source_roots, status_results):
        if not status or not source_root.has_git_marker:
            continue
        for git_path in _nested_git_root_paths(source_root.path):
            source_rel = _join_git_path(source_root.rel_path, git_path)
            _add_source_root(roots, pending, root, ignore_specs, root / source_rel, source_rel)

    _discover_nested_roots(roots, pending, root, ignore_specs)
    return tuple(sorted(roots.values(), key=lambda source_root: (len(source_root.rel_path), source_root.rel_path)))


def _git_cache_metadata_from_heads(
    source_roots: tuple[SourceRoot, ...], heads_by_root: Mapping[str, str | None]
) -> tuple[dict[str, str], ...] | None:
    if not source_roots or any(not source_root.has_git_marker for source_root in source_roots):
        return None
    metadata = []
    for source_root in source_roots:
        head = heads_by_root.get(source_root.rel_path)
        if head is None:
            return None
        metadata.append({"path": source_root.rel_path, "head": head})
    return tuple(metadata)


def _initial_source_root_candidates(
    root: Path, previous_paths: set[str], previous_git_heads: Mapping[str, str] | None
) -> Iterable[tuple[Path, str]]:
    seen: set[Path] = set()
    for path, rel_path in _all_initial_source_root_candidates(root, previous_paths, previous_git_heads):
        key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        yield path, rel_path


def _all_initial_source_root_candidates(
    root: Path, previous_paths: set[str], previous_git_heads: Mapping[str, str] | None
) -> Iterable[tuple[Path, str]]:
    if previous_git_heads is not None:
        for rel_path in previous_git_heads:
            rel_path = rel_path.strip("/")
            yield root if rel_path == "" else root / rel_path, rel_path
        return
    yield root, ""
    for previous_path in previous_paths:
        yield from _previous_path_source_root_candidates(root, previous_path)


def _previous_path_source_root_candidates(root: Path, previous_path: str) -> Iterable[tuple[Path, str]]:
    current = (root / previous_path).parent
    while True:
        try:
            current.relative_to(root)
        except ValueError:
            return
        yield current, "" if current == root else current.relative_to(root).as_posix()
        if current == root:
            return
        current = current.parent


def _add_source_root(
    roots: dict[Path, SourceRoot],
    pending: list[SourceRoot],
    display_root: Path,
    ignore_specs: IgnoreSpecCache,
    path: Path,
    rel_path: str,
    *,
    allow_missing_marker: bool = False,
) -> None:
    if path.is_symlink():
        return
    has_git_marker = _has_git_marker(path)
    if not has_git_marker and not allow_missing_marker:
        return
    if has_git_marker and rel_path and _is_path_ignored(path, display_root, ignore_specs):
        return
    resolved = path.resolve()
    if resolved in roots:
        return
    source_root = SourceRoot(resolved, rel_path, has_git_marker)
    roots[resolved] = source_root
    pending.append(source_root)


def _nested_git_root_paths(source_root: Path) -> tuple[str, ...]:
    return (*_tracked_gitlink_paths(source_root), *_untracked_git_root_paths(source_root))


def _load_ignore_specs(
    root: Path,
    control_dirs: frozenset[Path] | None = None,
    control_specs: dict[Path, IgnoreSpec | None] | None = None,
    base_specs: tuple[IgnoreSpec, ...] | None = None,
) -> IgnoreSpecCache:
    root = root.absolute()
    if base_specs is None:
        specs = [IgnoreSpec(root, GitIgnoreSpec.from_lines(sorted(_DEFAULT_IGNORED_DIRS), backend="simple"))]
        spec = _load_ignore_for_dir(root)
        if spec is not None:
            specs.append(IgnoreSpec(root, spec))
    else:
        specs = list(base_specs)
    return IgnoreSpecCache(root, {root: specs}, control_dirs, control_specs or {})


def _tracked_gitlink_paths(source_root: Path) -> tuple[str, ...]:
    output = _run_git(source_root, "ls-files", "-z", "--stage")
    if output is None:
        return ()
    paths = []
    for entry in output.split(b"\0"):
        if not entry.startswith(b"160000 "):
            continue
        _, _, raw_path = entry.partition(b"\t")
        if raw_path:
            paths.append(raw_path.decode("utf-8", errors="surrogateescape"))
    return tuple(paths)


def _untracked_git_root_paths(source_root: Path) -> tuple[str, ...]:
    output = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--ignore-submodules=all",
    )
    if output is None:
        return ()
    paths = []
    for entry in output.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        git_path = entry[3:].decode("utf-8", errors="surrogateescape").rstrip("/")
        if git_path and _has_git_marker(source_root / git_path):
            paths.append(git_path)
    return tuple(paths)


def _has_git_marker(path: Path) -> bool:
    return (path / ".git").exists()


def _is_path_ignored(path: Path, root: Path, ignore_specs: IgnoreSpecCache) -> bool:
    ignored, _ = _is_ignored(path, _ignore_specs_for_path(path, root, ignore_specs))
    return ignored


def _is_file_ignored(path: Path, specs: list[IgnoreSpec]) -> tuple[bool, bool]:
    ignored = False
    found = False
    for ignore_spec in specs:
        try:
            relative = path.relative_to(ignore_spec.base)
        except ValueError:
            continue

        relative_str = relative.as_posix()
        for pattern in ignore_spec.spec.patterns:
            if pattern.include is None:
                continue
            if pattern.match_file(relative_str) is not None:
                ignored = pattern.include
                pat = pattern.pattern
                found = not ignored and isinstance(pat, str) and bool(Path(pat.rstrip("/")).suffix)
    return ignored, found


def _source_roots_cover_paths(source_roots: tuple[SourceRoot, ...], paths: set[str]) -> bool:
    for path in paths:
        if not any(_is_same_or_child(path, source_root.rel_path) for source_root in source_roots):
            return False
    return True


def _previous_paths_in_root(paths: set[str], source_root: str) -> set[str]:
    return {path for path in paths if _is_same_or_child(path, source_root)}


def _previous_paths_by_source_root(paths: set[str], source_roots: tuple[SourceRoot, ...]) -> dict[Path, set[str]]:
    roots_by_rel_path = {source_root.rel_path: source_root.path for source_root in source_roots}
    paths_by_root: dict[Path, set[str]] = {source_root.path: set() for source_root in source_roots}
    for path in paths:
        source_root_path = _source_root_path_for_previous_path(path, roots_by_rel_path)
        if source_root_path is not None:
            paths_by_root[source_root_path].add(path)
    return paths_by_root


def _source_root_path_for_previous_path(path: str, roots_by_rel_path: Mapping[str, Path]) -> Path | None:
    parts = path.split("/")
    for end in range(len(parts) - 1, 0, -1):
        source_root_path = roots_by_rel_path.get("/".join(parts[:end]))
        if source_root_path is not None:
            return source_root_path
    return roots_by_rel_path.get("")


def _is_under_any_stale_root(path: str, stale_roots: set[str]) -> bool:
    return any(_is_same_or_child(path, source_root) for source_root in stale_roots)


def _child_roots_by_parent(source_roots: tuple[SourceRoot, ...]) -> dict[Path, tuple[SourceRoot, ...]]:
    children: dict[Path, list[SourceRoot]] = {}
    for parent in source_roots:
        for child in source_roots:
            if parent == child:
                continue
            if _is_relative_to(child.path, parent.path):
                children.setdefault(parent.path, []).append(child)
    return {path: tuple(values) for path, values in children.items()}


def _parent_ignore_control_dirs_by_source_root(
    source_roots: tuple[SourceRoot, ...], display_root: Path, ignore_specs: IgnoreSpecCache
) -> tuple[dict[Path, frozenset[Path]], dict[Path, IgnoreSpec | None]]:
    control_specs: dict[Path, IgnoreSpec | None] = {}
    dirs_by_root: dict[Path, frozenset[Path]] = {}
    for source_root in source_roots:
        dirs_by_root[source_root.path] = frozenset(
            _parent_ignore_control_dirs(source_root, display_root, ignore_specs, control_specs)
        )
    return dirs_by_root, control_specs


def _ignore_control_dirs(
    source_root: SourceRoot,
    display_root: Path,
    listed_paths: bytes,
    parent_control_dirs: frozenset[Path],
) -> frozenset[Path] | None:
    control_paths = _ignore_control_paths(source_root.path, listed_paths)
    if control_paths is None:
        return None

    dirs = set(parent_control_dirs)
    for git_path in control_paths:
        global_path = _join_git_path(source_root.rel_path, git_path)
        dirs.add((display_root / global_path).parent.absolute())
    return frozenset(dirs)


def _parent_ignore_control_dirs(
    source_root: SourceRoot,
    display_root: Path,
    ignore_specs: IgnoreSpecCache,
    control_specs: dict[Path, IgnoreSpec | None],
) -> set[Path]:
    dirs: set[Path] = set()
    current = display_root.absolute()
    for part in Path(source_root.rel_path).parts:
        current = current / part
        if _cached_ignore_spec_for_dir(current, ignore_specs, control_specs) is not None:
            dirs.add(current)
    return dirs


def _cached_ignore_spec_for_dir(
    directory: Path,
    ignore_specs: IgnoreSpecCache,
    control_specs: dict[Path, IgnoreSpec | None],
) -> IgnoreSpec | None:
    directory = directory.absolute()
    if directory in control_specs:
        return control_specs[directory]

    specs = ignore_specs.specs_by_dir.get(directory)
    if specs is not None:
        for spec in reversed(specs):
            if spec.base == directory:
                control_specs[directory] = spec
                return spec
        control_specs[directory] = None
        return None

    loaded_spec = _load_ignore_for_dir(directory)
    if loaded_spec is None:
        control_specs[directory] = None
        return None
    ignore_spec = IgnoreSpec(directory, loaded_spec)
    control_specs[directory] = ignore_spec
    return ignore_spec


def _ignore_control_paths(source_root: Path, listed_paths: bytes) -> set[str] | None:
    paths = _listed_ignore_control_paths(listed_paths)
    output = _run_git(source_root, "ls-files", "-z", "-o", "-i", "--exclude-standard", "--", *_IGNORE_CONTROL_PATHS)
    if output is None:
        return None
    paths.update(path.decode("utf-8", errors="surrogateescape") for path in output.split(b"\0") if path)
    return paths


def _listed_ignore_control_paths(listed_paths: bytes) -> set[str]:
    paths: set[str] = set()
    for path, _, _ in _listed_git_entries(listed_paths):
        if _is_ignore_control_path(path):
            paths.add(path)
    return paths


def _listed_git_entries(listed_paths: bytes) -> Iterable[tuple[str, bool, str | None]]:
    for raw_path in listed_paths.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if "\t" in path:
            metadata, git_path = path.split("\t", 1)
            parts = metadata.split()
            if len(parts) >= 2:
                yield git_path, True, parts[1]
                continue
        if len(path) >= 2 and path[1] == " ":
            yield path[2:], path[0] != "?", None
        else:
            yield path, False, None


def _source_root_status(source_root: SourceRoot) -> _SourceRootStatus:
    if not source_root.has_git_marker:
        return None, 0.0, None
    started = time.perf_counter()
    status = _run_git(
        source_root.path,
        "status",
        "--porcelain=v2",
        "--branch",
        "-z",
        "--untracked-files=normal",
        "--ignore-submodules=all",
    )
    elapsed = time.perf_counter() - started
    if status is None:
        return None, elapsed, None
    parsed = _parse_porcelain_v2_status(status)
    if parsed is not None:
        converted_status, head = parsed
        return converted_status, elapsed, head

    started = time.perf_counter()
    fallback_status = _run_git(
        source_root.path,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--ignore-submodules=all",
    )
    return fallback_status, elapsed + time.perf_counter() - started, None


def _parse_porcelain_v2_status(status: bytes) -> tuple[bytes, str | None] | None:
    head: str | None = None
    converted_entries: list[bytes] = []
    entries = status.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        is_header, entry_head = _porcelain_v2_header_head(entry)
        if is_header:
            head = entry_head or head
            continue
        converted = _porcelain_v2_entry_to_v1(entry, entries, index)
        if converted is None:
            return None
        converted_entry, index = converted
        converted_entries.extend(converted_entry)
    if not converted_entries:
        return b"", head
    return b"\0".join(converted_entries) + b"\0", head


def _porcelain_v2_header_head(entry: bytes) -> tuple[bool, str | None]:
    if entry.startswith(b"# branch.oid "):
        raw_head = entry.removeprefix(b"# branch.oid ")
        if raw_head and raw_head != b"(initial)":
            return True, raw_head.decode("ascii", errors="ignore")
        return True, None
    return entry.startswith(b"# "), None


def _porcelain_v2_entry_to_v1(
    entry: bytes,
    entries: list[bytes],
    next_index: int,
) -> tuple[tuple[bytes, ...], int] | None:
    if entry.startswith(b"? "):
        return (b"?? " + entry[2:],), next_index
    if entry.startswith(b"! "):
        return (b"!! " + entry[2:],), next_index
    if entry.startswith(b"1 "):
        converted_entry = _porcelain_v2_split_entry(entry, 8)
        return None if converted_entry is None else ((converted_entry,), next_index)
    if entry.startswith(b"2 "):
        converted_entry = _porcelain_v2_split_entry(entry, 9)
        if converted_entry is None or next_index >= len(entries):
            return None
        return (converted_entry, entries[next_index]), next_index + 1
    return None


def _porcelain_v2_split_entry(entry: bytes, maxsplit: int) -> bytes | None:
    parts = entry.split(b" ", maxsplit)
    if len(parts) != maxsplit + 1:
        return None
    return parts[1] + b" " + parts[maxsplit]


def _dirty_ignore_control_paths_by_source_root(
    source_roots: tuple[SourceRoot, ...],
    status_results: list[_SourceRootStatus],
    previous_write_time: float | None,
) -> tuple[str, ...]:
    dirty_paths: list[str] = []
    for source_root, (status, _, _) in zip(source_roots, status_results):
        if status is None:
            continue
        for git_path in _dirty_ignore_control_paths(status):
            if _ignore_control_is_covered_by_cache(source_root.path / git_path, previous_write_time):
                continue
            dirty_paths.append(_join_git_path(source_root.rel_path, git_path))
    return tuple(dirty_paths)


def _dirty_ignore_control_paths(status: bytes) -> tuple[str, ...]:
    paths = []
    entries = status.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        status_code = entry[:2].decode("ascii", errors="ignore")
        git_path = entry[3:].decode("utf-8", errors="surrogateescape")
        if status_code[0] in {"R", "C"}:
            index += 1
        if _is_ignore_control_path(git_path):
            paths.append(git_path)
    return tuple(paths)


def _is_ignore_control_path(path: str) -> bool:
    return path in _IGNORE_CONTROL_FILENAMES or path.endswith(_IGNORE_CONTROL_SUFFIXES)


def _ignore_control_is_covered_by_cache(path: Path, previous_write_time: float | None) -> bool:
    if previous_write_time is None:
        return False
    try:
        return path.stat().st_mtime <= previous_write_time
    except OSError:
        return False


def _has_dirty_inherited_ignore_control(source_root: str, dirty_ignore_controls: tuple[str, ...]) -> bool:
    for path in dirty_ignore_controls:
        control_dir = str(Path(path).parent)
        if control_dir == ".":
            control_dir = ""
        if _is_same_or_child(source_root, control_dir):
            return True
    return False


def _stale_source_root_result(
    source_root: SourceRoot,
    previous_paths: set[str],
    status_time: float,
    ls_files_time: float,
) -> SourceRootScanResult:
    files = _previous_paths_in_root(previous_paths, source_root.rel_path)
    return SourceRootScanResult(files, set(), set(), set(), {}, source_root.rel_path, status_time, ls_files_time, None)


def _scan_source_root(
    source_root: SourceRoot,
    display_root: Path,
    display_root_resolved: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    previous_paths: set[str],
    previous_tracked_paths: set[str] | None,
    previous_head: str | None,
    current_head: str | None,
    parent_control_dirs: frozenset[Path],
    control_specs: dict[Path, IgnoreSpec | None],
    base_ignore_specs: tuple[IgnoreSpec, ...],
    status_output: bytes | None,
    status_time: float,
    has_dirty_inherited_ignore_control: bool,
) -> SourceRootScanResult:
    if status_output is None:
        return _stale_source_root_result(source_root, previous_paths, status_time, 0.0)

    if previous_head is not None and previous_tracked_paths is not None and not has_dirty_inherited_ignore_control:
        if current_head is None:
            started = time.perf_counter()
            current_head = _git_head(source_root.path)
            status_time += time.perf_counter() - started
        if current_head == previous_head:
            if not status_output:
                current_tracked_paths = previous_paths & previous_tracked_paths
                return SourceRootScanResult(
                    current_tracked_paths,
                    current_tracked_paths,
                    set(),
                    set(),
                    {},
                    None,
                    status_time,
                    0.0,
                    current_head,
                )
            dirty_status_paths = _dirty_status_paths(
                source_root,
                display_root,
                display_root_resolved,
                extensions,
                child_roots,
                previous_paths,
                previous_tracked_paths,
                control_specs,
                base_ignore_specs,
                status_output,
            )
            if dirty_status_paths is not None:
                files, tracked, changed, deleted = dirty_status_paths
                return SourceRootScanResult(files, tracked, changed, deleted, {}, None, status_time, 0.0, current_head)

    started = time.perf_counter()
    listed_paths = _run_git(source_root.path, "ls-files", "--stage", "-z", "-c", "-o", "--exclude-standard")
    ls_files_time = time.perf_counter() - started
    if listed_paths is None:
        return _stale_source_root_result(source_root, previous_paths, status_time, ls_files_time)

    ignore_specs = _load_ignore_specs(
        display_root,
        _ignore_control_dirs(source_root, display_root, listed_paths, parent_control_dirs),
        control_specs,
        base_ignore_specs,
    )

    started = time.perf_counter()
    status = _git_status(
        source_root, display_root, display_root_resolved, extensions, child_roots, ignore_specs, status_output
    )
    status_time += time.perf_counter() - started

    started = time.perf_counter()
    files, tracked, blob_shas = _git_ls_files(
        listed_paths,
        source_root,
        display_root,
        display_root_resolved,
        extensions,
        child_roots,
        ignore_specs,
    )
    ls_files_time += time.perf_counter() - started

    return SourceRootScanResult(
        files,
        tracked,
        status[0],
        status[1],
        blob_shas,
        None,
        status_time,
        ls_files_time,
        current_head,
    )


def _dirty_status_paths(
    source_root: SourceRoot,
    display_root: Path,
    display_root_resolved: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    previous_paths: set[str],
    previous_tracked_paths: set[str],
    control_specs: dict[Path, IgnoreSpec | None],
    base_ignore_specs: tuple[IgnoreSpec, ...],
    status_output: bytes,
) -> tuple[set[str], set[str], set[str], set[str]] | None:
    if _dirty_ignore_control_paths(status_output):
        return None
    status_entries = _dirty_status_entries(status_output)
    if status_entries is None:
        return None

    ignore_specs = _load_ignore_specs(display_root, None, control_specs, base_ignore_specs)
    files = set(previous_paths)
    tracked = previous_paths & previous_tracked_paths
    changed: set[str] = set()
    deleted = _deleted_previous_untracked_paths(display_root, previous_paths, previous_tracked_paths)
    files.difference_update(deleted)

    for status, git_path in status_entries:
        if status == "??" and git_path.endswith("/"):
            for global_path in _indexable_paths_in_untracked_directory(
                source_root,
                git_path,
                display_root,
                display_root_resolved,
                extensions,
                child_roots,
                ignore_specs,
            ):
                files.add(global_path)
                changed.add(global_path)
            continue
        global_path = _global_git_path(source_root.path / git_path, display_root_resolved)
        if "D" in status:
            deleted.add(global_path)
            files.discard(global_path)
            tracked.discard(global_path)
            continue
        if not _is_indexable_path(global_path, display_root, extensions, child_roots, ignore_specs):
            continue
        files.add(global_path)
        changed.add(global_path)
        if status != "??":
            tracked.add(global_path)

    return files, tracked, changed, deleted


def _deleted_previous_untracked_paths(
    display_root: Path, previous_paths: set[str], previous_tracked_paths: set[str]
) -> set[str]:
    return {path for path in previous_paths - previous_tracked_paths if not (display_root / path).exists()}


def _indexable_paths_in_untracked_directory(
    source_root: SourceRoot,
    git_path: str,
    display_root: Path,
    display_root_resolved: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    ignore_specs: IgnoreSpecCache,
) -> tuple[str, ...]:
    paths = []
    pending = [source_root.path / git_path]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                continue
            global_path = _global_git_path(child, display_root_resolved)
            if any(_is_same_or_child(global_path, root.rel_path) for root in child_roots):
                continue
            if child.is_dir():
                ignored, _ = _is_file_ignored(child, _ignore_specs_for_path(child, display_root, ignore_specs))
                if not ignored:
                    pending.append(child)
                continue
            if child.is_file() and _is_indexable_path(global_path, display_root, extensions, child_roots, ignore_specs):
                paths.append(global_path)
    return tuple(sorted(paths))


def _dirty_status_entries(status_output: bytes) -> tuple[tuple[str, str], ...] | None:
    entries = []
    for entry in status_output.split(b"\0"):
        if not entry or len(entry) < 4:
            continue
        status = entry[:2].decode("ascii", errors="ignore")
        if status[0] in {"R", "C"}:
            return None
        git_path = entry[3:].decode("utf-8", errors="surrogateescape")
        if not git_path:
            continue
        if git_path.endswith("/") and status != "??":
            return None
        entries.append((status, git_path))
    return tuple(entries)


def _git_status(
    source_root: SourceRoot,
    display_root: Path,
    display_root_resolved: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    ignore_specs: IgnoreSpecCache,
    result: bytes,
) -> tuple[set[str], set[str]]:
    changed: set[str] = set()
    deleted: set[str] = set()
    entries = result.split(b"\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            continue
        status = entry[:2].decode("ascii", errors="ignore")
        git_path = entry[3:].decode("utf-8", errors="surrogateescape")
        if status[0] in {"R", "C"}:
            index += 1
        global_path = _global_git_path(source_root.path / git_path, display_root_resolved)
        if not _is_indexable_path(global_path, display_root, extensions, child_roots, ignore_specs):
            continue
        if "D" in status:
            deleted.add(global_path)
        else:
            deleted.discard(global_path)
            changed.add(global_path)
    return changed, deleted


def _git_ls_files(
    listed_paths: bytes,
    source_root: SourceRoot,
    display_root: Path,
    display_root_resolved: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    ignore_specs: IgnoreSpecCache,
) -> tuple[set[str], set[str], dict[str, str]]:
    files = set()
    tracked = set()
    blob_shas: dict[str, str] = {}
    for git_path, is_tracked, blob_sha in _listed_git_entries(listed_paths):
        global_path = _global_git_path(source_root.path / git_path, display_root_resolved)
        if _is_indexable_path(global_path, display_root, extensions, child_roots, ignore_specs):
            files.add(global_path)
            if is_tracked:
                tracked.add(global_path)
                if blob_sha is not None:
                    blob_shas[global_path] = blob_sha
    return files, tracked, blob_shas


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


def _git_head(source_root: Path) -> str | None:
    result = _run_git(source_root, "rev-parse", "HEAD")
    if result is None:
        return None
    return result.decode("utf-8", errors="surrogateescape").strip()


def _is_indexable_path(
    path: str,
    display_root: Path,
    extensions: frozenset[str],
    child_roots: tuple[SourceRoot, ...],
    ignore_specs: IgnoreSpecCache,
) -> bool:
    if any(_is_same_or_child(path, child.rel_path) for child in child_roots):
        return False
    local_path = display_root / path
    ignored, found = _is_file_ignored(local_path, _ignore_specs_for_path(local_path, display_root, ignore_specs))
    if ignored:
        return False
    return found or Path(path).suffix.lower() in extensions


def _ignore_specs_for_path(path: Path, root: Path, ignore_specs: IgnoreSpecCache) -> list[IgnoreSpec]:
    root = root.absolute()
    directory = path.parent.absolute()
    try:
        relative = directory.relative_to(root)
    except ValueError:
        return ignore_specs.specs_by_dir[ignore_specs.root]

    specs = ignore_specs.specs_by_dir[ignore_specs.root]
    current = root
    for part in relative.parts:
        current = current / part
        cached = ignore_specs.specs_by_dir.get(current)
        if cached is None:
            cached = list(specs)
            if ignore_specs.control_dirs is None or current in ignore_specs.control_dirs:
                if current in ignore_specs.control_specs:
                    spec = ignore_specs.control_specs[current]
                else:
                    loaded = _load_ignore_for_dir(current)
                    spec = None if loaded is None else IgnoreSpec(current, loaded)
                if spec is not None:
                    cached.append(spec)
            ignore_specs.specs_by_dir[current] = cached
        specs = cached
    return specs


def _global_git_path(path: Path, display_root_resolved: Path) -> str:
    return path.absolute().relative_to(display_root_resolved).as_posix()


def _join_git_path(parent: str, child: str) -> str:
    return child if parent == "" else f"{parent}/{child}"


def _is_same_or_child(path: str, parent: str) -> bool:
    if parent == "":
        return True
    return path == parent or path.startswith(f"{parent}/")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
