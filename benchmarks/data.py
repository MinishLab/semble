import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

BENCH_ROOT = Path.home() / ".cache" / "semble-bench"
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
        """Return True if both start_line and end_line are set."""
        return self.start_line is not None and self.end_line is not None


@dataclass(frozen=True)
class RepoSpec:
    name: str
    language: str
    url: str
    revision: str
    benchmark_root: str | None = None

    @property
    def checkout_dir(self) -> Path:
        """Return the local checkout directory for this repo."""
        return BENCH_ROOT / self.name

    @property
    def benchmark_dir(self) -> Path:
        """Return the root directory to index for benchmarking."""
        return self.checkout_dir if self.benchmark_root is None else self.checkout_dir / self.benchmark_root


@dataclass(frozen=True)
class Task:
    repo: str
    language: str
    query: str
    relevant: tuple[Target, ...]
    secondary: tuple[Target, ...]
    category: str

    @property
    def all_relevant(self) -> tuple[Target, ...]:
        """Return primary and secondary relevant targets combined."""
        return self.relevant + self.secondary


def infer_category(query: str) -> str:
    """Infer a task category from the query text."""
    if " " not in query.strip():
        return "symbol"
    lowered = query.lower()
    if lowered.startswith("how ") or lowered.startswith("how does") or lowered.startswith("how are"):
        return "architecture"
    return "semantic"


def _coerce_int(value: object) -> int:
    """Coerce a string or int value to int, raising TypeError otherwise."""
    if not isinstance(value, int | str):
        raise TypeError(f"expected int-compatible value, got {type(value).__name__}")
    return int(value)


def _parse_target(raw: str | dict[str, object]) -> Target:
    """Parse a target from a string path or a mapping with optional line span."""
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
    """Load all repo specs from the JSON file at the given path."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: RepoSpec(**item) for item in raw}


def available_repo_specs() -> dict[str, RepoSpec]:
    """Return only the repo specs that have a local checkout and annotation file."""
    return {
        name: spec
        for name, spec in load_repo_specs().items()
        if spec.checkout_dir.exists() and (ANNOTATIONS_DIR / f"{name}.json").exists()
    }


def load_tasks(repo_specs: dict[str, RepoSpec] | None = None) -> list[Task]:
    """Load all benchmark tasks from annotation files, filtered to available repo specs."""
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
                )
            )
    return tasks


def apply_task_filters(
    tasks: list[Task],
    repos: list[str] | None = None,
    languages: list[str] | None = None,
) -> list[Task]:
    """Filter tasks to the given repos and/or languages; None means no filter."""
    filtered = [task for task in tasks if not repos or task.repo in repos]
    return [task for task in filtered if not languages or task.language in languages]


def path_matches(file_path: str, target_path: str) -> bool:
    """Return True if file_path ends with target_path (file-level match, no line span)."""
    norm_file = file_path.replace("\\", "/")
    norm_target = target_path.replace("\\", "/")
    return norm_file == norm_target or norm_file.endswith(f"/{norm_target}")


def target_matches_location(file_path: str, start_line: int, end_line: int, target: Target) -> bool:
    """Return True if the chunk at file_path:start_line-end_line covers the target."""
    if not path_matches(file_path, target.path):
        return False
    if not target.has_span:
        return True
    return not (end_line < target.start_line or start_line > target.end_line)  # type: ignore[operator]


def grouped_tasks(tasks: list[Task]) -> dict[str, list[Task]]:
    """Group tasks by repo name, preserving annotation order within each group."""
    groups: dict[str, list[Task]] = {}
    for task in tasks:
        groups.setdefault(task.repo, []).append(task)
    return groups


def current_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def results_path(method: str) -> Path:
    """Return the path where results for this method and the current HEAD SHA will be saved.

    :param method: Short tool/method label (e.g. 'ripgrep', 'colgrep').
    :return: Path of the form benchmarks/results/<method>-<sha12>.json.
    """
    sha = current_sha()
    results_dir = BENCHMARKS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir / f"{method}-{sha[:12]}.json"


def save_results(method: str, payload: object) -> Path:
    """Write payload to benchmarks/results/<method>-<sha12>.json and return the path.

    :param method: Short tool/method label used as the filename prefix (e.g. 'ripgrep').
    :param payload: JSON-serialisable object to write.
    :return: Path to the written file.
    """
    out_path = results_path(method)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
