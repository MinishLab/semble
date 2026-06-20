from __future__ import annotations

import random

from swebench.harness.constants import SWEbenchInstance
from swebench.harness.utils import get_modified_files, load_swebench_dataset

_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_REPO = "pytest-dev/pytest"
DEFAULT_SEED = 42


class SWETask:
    """Thin wrapper around a :class:`SWEbenchInstance` dict with computed helpers."""

    def __init__(self, instance: SWEbenchInstance) -> None:
        """Wrap a raw ``SWEbenchInstance`` dict."""
        self._instance = instance

    @property
    def instance_id(self) -> str:
        """The unique instance identifier."""
        return self._instance["instance_id"]

    @property
    def repo(self) -> str:
        """The GitHub ``owner/repo`` string."""
        return self._instance["repo"]

    @property
    def base_commit(self) -> str:
        """The commit hash the task starts from."""
        return self._instance["base_commit"]

    @property
    def problem_statement(self) -> str:
        """The full issue text to feed to the agent."""
        return self._instance["problem_statement"]

    @property
    def patch(self) -> str:
        """The gold patch (unified diff)."""
        return self._instance["patch"]

    @property
    def gold_files(self) -> list[str]:
        """Paths touched by the gold patch."""
        return get_modified_files(self._instance["patch"])

    @property
    def short_label(self) -> str:
        """First non-blank line of the problem statement (for console output only)."""
        for line in self.problem_statement.splitlines():
            if line.strip():
                return line.strip()[:200]
        return self.problem_statement[:200]

    @property
    def raw(self) -> SWEbenchInstance:
        """The underlying ``SWEbenchInstance`` dict, for direct field access."""
        return self._instance


def fetch_tasks(n: int, repo: str = DEFAULT_REPO, seed: int = DEFAULT_SEED) -> list[SWETask]:
    """Fetch SWE-bench Lite tasks, seeded-randomly sampled.

    *repo* accepts a single repo, a comma-separated list, or ``"all"``.
    """
    instances = load_swebench_dataset(_DATASET, split="test")
    if repo == "all":
        pool = instances
    else:
        repos = {r.strip() for r in repo.split(",")}
        pool = [inst for inst in instances if inst["repo"] in repos]
    random.Random(seed).shuffle(pool)
    return [SWETask(inst) for inst in pool[:n]]


def resolve_tasks(n_tasks: int, repo: str, instance_ids: list[str] | None, seed: int) -> list[SWETask]:
    """Fetch either a specific set of instance IDs or a fresh random sample."""
    if instance_ids:
        print(f"Fetching tasks for {len(instance_ids)} specific instance IDs...")
        instances = load_swebench_dataset(_DATASET, split="test", instance_ids=instance_ids)
        return [SWETask(inst) for inst in instances]
    print(f"Fetching {n_tasks} randomly sampled tasks from SWE-bench Lite (repo={repo}, seed={seed})...")
    return fetch_tasks(n_tasks, repo, seed=seed)
