from __future__ import annotations

import math
import random
import shutil
import subprocess
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
