from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.swe.gitutils import changed_files, git_reset

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_TIMEOUT = 480  # seconds per agent run
_RETRY_SLEEP = 45  # seconds between retries on empty output

_PYPI_SEMBLE_CMD = ["uvx", "--from", "semble[mcp]", "semble"]
_LOCAL_SEMBLE_CMD = ["uv", "run", "--directory", str(_PROJECT_ROOT), "semble"]


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


def is_semble_tool_call(entry: str) -> bool:
    """Return True if a tool-call entry string originated from an semble MCP call or bypass."""
    return (
        entry.startswith("claude_mcp:semble/")
        or entry.startswith("codex_mcp:semble/")
        or entry.startswith("opencode_mcp:semble/")
        or "[SEMBLE_BYPASS]" in entry
    )


def _is_bypass_call(entry: str) -> bool:
    return "[SEMBLE_BYPASS]" in entry


@contextmanager
def _isolated_cache_env() -> Iterator[dict[str, str]]:
    """A fresh ``SEMBLE_CACHE_LOCATION`` per run, so with/without variants never share a warm index."""
    tmp = Path(tempfile.mkdtemp(prefix="semble_cache_"))
    try:
        yield {"SEMBLE_CACHE_LOCATION": str(tmp)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _path_without_semble() -> str:
    """``PATH`` with the directory containing the ``semble`` binary removed (best-effort bypass prevention)."""
    semble_bin = shutil.which("semble")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if semble_bin:
        semble_dir = str(Path(semble_bin).parent)
        path_entries = [p for p in path_entries if p != semble_dir]
    return os.pathsep.join(path_entries)


@contextmanager
def _subprocess_env(env_overrides: dict[str, str], *, with_semble: bool) -> Iterator[dict[str, str]]:
    """Env for a backend subprocess: isolated cache dir, plus PATH sanitization when semble is excluded."""
    with _isolated_cache_env() as cache_env:
        env = {**os.environ, **env_overrides, **cache_env}
        if not with_semble:
            env["PATH"] = _path_without_semble()
        yield env


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_with_timeout(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
    """Like ``subprocess.run`` with a timeout, but kills the whole process group on expiry."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


class Backend(ABC):
    """Pluggable MCP agent backend."""

    name: str = "base"
    default_model: str = ""
    _rate_limit_msg: str = "rate limited"

    def __init__(self, model: str | None = None, local_semble: bool = False) -> None:
        """Create a backend, optionally overriding the default model and using a local semble checkout."""
        self.model = model or self.default_model
        self.local_semble = local_semble

    @property
    def _semble_cmd(self) -> list[str]:
        return _LOCAL_SEMBLE_CMD if self.local_semble else _PYPI_SEMBLE_CMD

    @abstractmethod
    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[dict, str]: ...

    def _attempt_succeeded(self, parsed: dict) -> bool:
        return parsed["num_turns"] > 0

    def label(self) -> str:
        """Human-readable backend/model identifier used in console output and result files."""
        return f"{self.name}/{self.model}"

    def run(self, prompt: str, repo: Path, commit: str, *, with_semble: bool) -> RunResult:
        """Run the agent on ``prompt``, retrying on empty output, and reset ``repo`` to ``commit`` afterward."""
        variant = "with_semble" if with_semble else "without_semble"
        try:
            for attempt in range(3):
                if attempt > 0:
                    print(f"    retry {attempt} (empty output — sleeping {_RETRY_SLEEP}s)...")
                    time.sleep(_RETRY_SLEEP)
                parsed, diff = self._run_once(prompt, repo, with_semble=with_semble)
                git_reset(repo, commit)
                if parsed["rate_limited"]:
                    return RunResult(variant=variant, backend=self.name, error=self._rate_limit_msg)
                if self._attempt_succeeded(parsed):
                    break
            return self._finalize(variant, parsed, diff, empty_output=not self._attempt_succeeded(parsed))
        except subprocess.TimeoutExpired:
            git_reset(repo, commit)
            return RunResult(variant=variant, backend=self.name, model=self.model, error=f"timed out after {_TIMEOUT}s")
        except Exception as exc:
            git_reset(repo, commit)
            return RunResult(variant=variant, backend=self.name, model=self.model, error=str(exc))

    def _finalize(self, variant: str, parsed: dict, diff: str, *, empty_output: bool) -> RunResult:
        touched = changed_files(diff)
        tool_calls = parsed["tool_calls"]
        error = "empty output after retries" if empty_output else None
        return RunResult(
            variant=variant,
            backend=self.name,
            model=self.model,
            tool_calls=tool_calls,
            cost_usd=parsed["cost_usd"],
            input_tokens=parsed["input_tokens"],
            output_tokens=parsed["output_tokens"],
            num_turns=parsed["num_turns"],
            touched_files=touched,
            patch=diff,
            bypass=any(_is_bypass_call(t) for t in tool_calls),
            error=error,
        )
