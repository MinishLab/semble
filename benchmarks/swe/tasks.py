from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from benchmarks.swe.gitutils import changed_files

RESULTS_DIR = Path(__file__).parent / "results"
_HF_CHUNK = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Lite&config=default&split=test"
    "&offset={offset}&length=100"
)
DEFAULT_REPO = "pytest-dev/pytest"
DEFAULT_SEED = 42
_HF_CACHE = RESULTS_DIR / "swe_bench_lite_tasks.json"


@dataclass(frozen=True)
class SWETask:
    """A single SWE-bench Lite task."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str

    @property
    def gold_files(self) -> list[str]:
        """Paths touched by the gold patch."""
        return changed_files(self.patch)

    @property
    def short_label(self) -> str:
        """First non-blank line of the problem statement (for console output only)."""
        for line in self.problem_statement.splitlines():
            if line.strip():
                return line.strip()[:200]
        return self.problem_statement[:200]

    @classmethod
    def from_hf_row(cls, row: dict) -> SWETask:
        """Construct from a HuggingFace dataset row."""
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=row["patch"],
        )


def fetch_tasks(
    n: int, repo: str = DEFAULT_REPO, seed: int = DEFAULT_SEED, refresh_cache: bool = False
) -> list[SWETask]:
    """Fetch SWE-bench Lite tasks, seeded-randomly sampled.

    *repo* accepts a single repo, a comma-separated list, or ``"all"``.
    """
    if refresh_cache and _HF_CACHE.exists():
        _HF_CACHE.unlink()
    if _HF_CACHE.exists():
        all_rows = json.loads(_HF_CACHE.read_text())
    else:
        all_rows = []
        for offset in range(0, 300, 100):
            with urllib.request.urlopen(_HF_CHUNK.format(offset=offset), timeout=30) as r:
                all_rows += [row["row"] for row in json.loads(r.read())["rows"]]
        RESULTS_DIR.mkdir(exist_ok=True)
        _HF_CACHE.write_text(json.dumps(all_rows))

    if repo == "all":
        pool = list(all_rows)
    else:
        repos = {r.strip() for r in repo.split(",")}
        pool = [r for r in all_rows if r["repo"] in repos]

    random.Random(seed).shuffle(pool)
    return [SWETask.from_hf_row(row) for row in pool[:n]]


def resolve_tasks(
    n_tasks: int, repo: str, instance_ids: list[str] | None, seed: int, refresh_cache: bool
) -> list[SWETask]:
    """Fetch either a specific set of instance IDs or a fresh random sample."""
    if instance_ids:
        print(f"Fetching tasks for {len(instance_ids)} specific instance IDs...")
        tasks = fetch_tasks(300, "all", seed=seed, refresh_cache=refresh_cache)
        tasks = [t for t in tasks if t.instance_id in set(instance_ids)]
        missing = set(instance_ids) - {t.instance_id for t in tasks}
        if missing:
            print(f"WARNING: {len(missing)} IDs not found: {missing}")
        return tasks
    print(f"Fetching {n_tasks} randomly sampled tasks from SWE-bench Lite (repo={repo}, seed={seed})...")
    return fetch_tasks(n_tasks, repo, seed=seed, refresh_cache=refresh_cache)
