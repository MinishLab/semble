from __future__ import annotations

import contextlib
import fcntl
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

SWE_DIR = Path(__file__).parent
REPOS_DIR = SWE_DIR / "repos"
RESULTS_DIR = SWE_DIR / "results"
PROJECT_ROOT = SWE_DIR.parent.parent

DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_REPO = "pytest-dev/pytest"
DEFAULT_SEED = 42

WITH_SEMBLE = "with_semble"
WITHOUT_SEMBLE = "without_semble"

TASK_SLEEP = 30  # seconds between tasks (rate-limit buffer)
PROBLEM_MAX_CHARS = 8000
CLONE_TIMEOUT = 180  # seconds per git clone/checkout

PROMPT_BASE = """\
You are a software engineer. Fix the following GitHub issue in the repository at {repo}.

{problem}

Instructions:
- Explore the repository to understand the relevant code
- Make the minimal change needed to fix the issue
- Do NOT run the test suite
- Do NOT add new test files
"""


def variant_name(with_semble: bool, experiment: str | None = None) -> str:
    """Return the variant tag, appending the experiment suffix to with-semble only."""
    base = WITH_SEMBLE if with_semble else WITHOUT_SEMBLE
    return f"{base}_{experiment}" if experiment and with_semble else base


def agent_results_path(experiment: str | None = None) -> Path:
    """Return the merged agent-results JSON path for the given experiment."""
    return RESULTS_DIR / (f"swe_agent_{experiment}.json" if experiment else "swe_agent.json")


def prediction_path(*, with_semble: bool, experiment: str | None = None, model_slug: str | None = None) -> Path:
    """Return the prediction JSONL path for the given variant.

    Always scoped by ``model_slug`` (backend+model), so two different backends/models
    never share a file. The with-semble file is additionally scoped by ``experiment``,
    since the same backend+model may be re-run multiple times against different semble
    versions/configs. The without-semble file has no experiment suffix: the baseline is
    specific to a backend+model, not to an experiment, so it's safely reused across
    multiple with-semble reruns of the same backend+model.
    """
    variant = WITH_SEMBLE if with_semble else WITHOUT_SEMBLE
    parts = [f"predictions_{variant}"]
    if model_slug:
        parts.append(model_slug)
    if with_semble and experiment:
        parts.append(experiment)
    return RESULTS_DIR / ("_".join(parts) + ".jsonl")


def resolve_results_path(experiment: str | None = None) -> Path:
    """Return the harness resolve-results JSON path for the given experiment."""
    return RESULTS_DIR / (f"swe_resolve_{experiment}.json" if experiment else "swe_resolve.json")


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Cross-process exclusive lock keyed by *path*, to serialize read-merge-write on shared files."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@dataclass
class ParsedRun:
    """Parsed output from an agent's JSON stream — tool calls, cost, tokens, rate-limit status."""

    tool_calls: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    rate_limited: bool = False


@dataclass
class RunResult:
    """Outcome of a single agent run on one SWE-bench task variant."""

    variant: str
    backend: str = ""
    model: str = ""
    semble_version: str = ""
    tool_calls: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    touched_files: list[str] = field(default_factory=list)
    patch: str = ""
    gold_hit: bool = False
    bypass: bool = False
    error: str | None = None


@dataclass
class TaskResult:
    """Results for a single SWE-bench task across all variants."""

    instance_id: str
    gold_files: list[str]
    results: list[RunResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> TaskResult:
        """Reconstruct from a JSON-serialised dict (for merge/resume)."""
        return cls(
            instance_id=d["instance_id"],
            gold_files=d["gold_files"],
            results=[RunResult(**r) for r in d["results"]],
        )


def is_semble_tool_call(entry: str) -> bool:
    """Return True if a tool-call entry string originated from an semble MCP call or bypass."""
    return (
        entry.startswith("claude_mcp:semble/")
        or entry.startswith("codex_mcp:semble/")
        or entry.startswith("opencode_mcp:semble/")
        or "[SEMBLE_BYPASS]" in entry
    )


def is_bypass_call(entry: str) -> bool:
    """Return True if a tool-call entry string represents a shell-out bypass to the semble CLI."""
    return "[SEMBLE_BYPASS]" in entry


_SEMBLE_INVOCATION_RE = re.compile(r"(?<![\w/.-])semble(?![\w/.-])")


def is_semble_shell_invocation(command: str) -> bool:
    """Return True if *command* actually invokes the ``semble`` CLI (not just a path mention).

    A naive ``"semble" in command`` substring check false-positives on any command
    referencing an absolute path inside this project (e.g. ``.../oss/semble/...``),
    since the project itself is named ``semble``. This matches ``semble`` as a
    standalone token instead, excluding path/word characters on either side.
    """
    return bool(_SEMBLE_INVOCATION_RE.search(command))


def clone_at_commit(repo: str, commit: str, dest: Path) -> None:
    """Clone ``repo`` at ``commit`` into ``dest``, reusing a clean checkout at the right HEAD."""
    if dest.exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}", str(dest)],
        check=True,
        timeout=CLONE_TIMEOUT,
    )
    subprocess.run(["git", "checkout", "--quiet", commit], cwd=dest, check=True, timeout=CLONE_TIMEOUT)


def git_diff(repo: Path) -> str:
    """Return ``git diff HEAD`` output for ``repo``."""
    return subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True).stdout


def git_reset(repo: Path, commit: str) -> None:
    """Hard-reset ``repo`` to ``commit`` and remove all untracked/ignored files."""
    subprocess.run(["git", "reset", "--hard", commit], cwd=repo, capture_output=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=repo, capture_output=True)


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = DEFAULT_SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean of ``values`` (stdlib-only, no scipy dep)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs (``b`` vs ``c``), stdlib-only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)
