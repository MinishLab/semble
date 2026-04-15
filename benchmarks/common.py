from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

BENCH_ROOT = Path("/tmp/bench")
BENCHMARKS_DIR = Path(__file__).parent
ANNOTATIONS_DIR = BENCHMARKS_DIR / "annotations"
REPOS_PATH = BENCHMARKS_DIR / "repos.json"


@dataclass(frozen=True)
class Target:
    path: str
    start_line: int | None = None
    end_line: int | None = None

    @property
    def has_span(self) -> bool:
        return self.start_line is not None and self.end_line is not None


class _ChunkLike(Protocol):
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class RepoSpec:
    name: str
    language: str
    url: str
    revision: str
    benchmark_root: str | None = None

    @property
    def checkout_dir(self) -> Path:
        return BENCH_ROOT / self.name

    @property
    def benchmark_dir(self) -> Path:
        return self.checkout_dir if self.benchmark_root is None else self.checkout_dir / self.benchmark_root


@dataclass(frozen=True)
class Task:
    repo: str
    language: str
    query: str
    relevant: tuple[Target, ...]
    secondary: tuple[Target, ...]
    category: str
    category_inferred: bool

    @property
    def all_relevant(self) -> tuple[Target, ...]:
        return self.relevant + self.secondary


def infer_category(query: str) -> str:
    if " " not in query.strip():
        return "symbol"
    lowered = query.lower()
    if lowered.startswith("how ") or lowered.startswith("how does") or lowered.startswith("how are"):
        return "architecture"
    return "semantic"


def _coerce_int(value: object) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"expected int-compatible value, got {type(value).__name__}")
    return int(value)


def _coerce_mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected mapping, got {type(raw).__name__}")
    return cast(dict[str, object], raw)


def _parse_target(raw: str | dict[str, object]) -> Target:
    if isinstance(raw, str):
        return Target(path=raw)
    raw = _coerce_mapping(raw)
    start_line = raw.get("start_line")
    end_line = raw.get("end_line")
    return Target(
        path=str(raw["path"]),
        start_line=_coerce_int(start_line) if start_line is not None else None,
        end_line=_coerce_int(end_line) if end_line is not None else None,
    )


def load_repo_specs(path: Path = REPOS_PATH) -> dict[str, RepoSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: RepoSpec(**item) for item in raw}


def available_repo_specs(repo_specs: dict[str, RepoSpec] | None = None) -> dict[str, RepoSpec]:
    specs = load_repo_specs() if repo_specs is None else repo_specs
    return {
        name: spec
        for name, spec in specs.items()
        if spec.checkout_dir.exists() and (ANNOTATIONS_DIR / f"{name}.json").exists()
    }


def load_tasks(
    path: Path = ANNOTATIONS_DIR,
    repo_specs: dict[str, RepoSpec] | None = None,
) -> list[Task]:
    specs = load_repo_specs() if repo_specs is None else repo_specs
    tasks: list[Task] = []
    annotation_files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for annotation_file in annotation_files:
        if annotation_file.stem not in specs:
            continue
        raw = json.loads(annotation_file.read_text(encoding="utf-8"))
        default_repo = annotation_file.stem
        for item in raw:
            repo = item.get("repo", default_repo)
            if repo not in specs:
                continue
            spec = specs[repo]
            category = item.get("category")
            tasks.append(
                Task(
                    repo=repo,
                    language=spec.language,
                    query=item["query"],
                    relevant=tuple(_parse_target(t) for t in item.get("relevant", [])),
                    secondary=tuple(_parse_target(t) for t in item.get("secondary", [])),
                    category=category if isinstance(category, str) else infer_category(item["query"]),
                    category_inferred=category is None,
                )
            )
    return tasks


def apply_task_filters(
    tasks: list[Task],
    repos: list[str] | None = None,
    languages: list[str] | None = None,
    limit: int | None = None,
) -> list[Task]:
    filtered = [task for task in tasks if not repos or task.repo in repos]
    filtered = [task for task in filtered if not languages or task.language in languages]
    return filtered if limit is None else filtered[:limit]


def path_matches(file_path: str, relative_path: str) -> bool:
    normalized_file = file_path.replace("\\", "/")
    normalized_relative = relative_path.replace("\\", "/")
    return normalized_file == normalized_relative or normalized_file.endswith(f"/{normalized_relative}")


def span_overlaps(start_line: int, end_line: int, target: Target) -> bool:
    if not target.has_span:
        return True
    target_start: int = target.start_line  # type: ignore[assignment]
    target_end: int = target.end_line  # type: ignore[assignment]
    return not (end_line < target_start or start_line > target_end)


def target_matches_location(file_path: str, start_line: int, end_line: int, target: Target) -> bool:
    return path_matches(file_path, target.path) and span_overlaps(start_line, end_line, target)


def count_indexed_targets(chunks: list[_ChunkLike], targets: tuple[Target, ...]) -> int:
    return sum(
        1
        for target in targets
        if any(target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target) for chunk in chunks)
    )


def grouped_tasks(tasks: list[Task]) -> dict[str, list[Task]]:
    grouped: dict[str, list[Task]] = {}
    for task in tasks:
        grouped.setdefault(task.repo, []).append(task)
    return grouped
