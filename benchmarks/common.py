from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


def _parse_target(raw: str | dict[str, object]) -> Target:
    if isinstance(raw, str):
        return Target(path=raw)
    if not isinstance(raw, dict):
        raise TypeError(f"expected mapping, got {type(raw).__name__}")
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


def load_tasks(repo_specs: dict[str, RepoSpec] | None = None) -> list[Task]:
    specs = load_repo_specs() if repo_specs is None else repo_specs
    tasks: list[Task] = []
    for annotation_file in sorted(ANNOTATIONS_DIR.glob("*.json")):
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
) -> list[Task]:
    filtered = [task for task in tasks if not repos or task.repo in repos]
    return [task for task in filtered if not languages or task.language in languages]


def target_matches_location(file_path: str, start_line: int, end_line: int, target: Target) -> bool:
    norm_file = file_path.replace("\\", "/")
    norm_target = target.path.replace("\\", "/")
    if not (norm_file == norm_target or norm_file.endswith(f"/{norm_target}")):
        return False
    if not target.has_span:
        return True
    return not (end_line < target.start_line or start_line > target.end_line)  # type: ignore[operator]


def count_indexed_targets(chunks: list[_ChunkLike], targets: tuple[Target, ...]) -> int:
    return sum(
        1
        for target in targets
        if any(target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target) for chunk in chunks)
    )


def grouped_tasks(tasks: list[Task]) -> dict[str, list[Task]]:
    result: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        result[task.repo].append(task)
    return dict(result)
