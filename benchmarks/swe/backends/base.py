from __future__ import annotations

import functools
import os
import shutil
import signal
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from swebench.harness.utils import get_modified_files

from benchmarks.swe.utils import PROJECT_ROOT, ParsedRun, RunResult, git_reset, is_bypass_call, variant_name

_TIMEOUT = 480  # seconds per agent run
_RETRY_SLEEP = 45  # seconds between retries on empty output

_PYPI_SEMBLE_CMD = ["uvx", "--from", "semble[mcp]", "semble"]
_LOCAL_SEMBLE_CMD = ["uv", "run", "--directory", str(PROJECT_ROOT), "semble"]


@functools.lru_cache(maxsize=1)
def _pypi_semble_version() -> str:
    """Resolve the semble version ``uvx`` actually picks up (subject to the ``exclude-newer`` walk-up bug)."""
    try:
        proc = subprocess.run(
            [
                "uvx",
                "--from",
                "semble[mcp]",
                "python",
                "-c",
                "from semble.version import __version__; print(__version__)",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def prepare_temp_home(
    *,
    prefix: str,
    env_var: str,
    config_relpath: Path | None = None,
    config_text: str | None = None,
    extra_copies: list[tuple[Path, Path]] | None = None,
) -> tuple[Path, dict[str, str]]:
    """Create a temp home/config root and return it with the matching env override."""
    temp_home = Path(tempfile.mkdtemp(prefix=prefix))
    if config_relpath is not None and config_text is not None:
        dest = temp_home / config_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(config_text)
    for src, rel_dest in extra_copies or []:
        if not src.exists():
            continue
        dest = temp_home / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return temp_home, {env_var: str(temp_home)}


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
def subprocess_env(env_overrides: dict[str, str], *, with_semble: bool) -> Iterator[dict[str, str]]:
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


def run_with_timeout(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess:
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

    @property
    def _semble_version(self) -> str:
        if self.local_semble:
            from semble.version import __version__

            return __version__
        return _pypi_semble_version()

    @abstractmethod
    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[ParsedRun, str]: ...

    def _attempt_succeeded(self, parsed: ParsedRun) -> bool:
        return parsed.num_turns > 0

    def label(self) -> str:
        """Human-readable backend/model identifier used in console output and result files."""
        return f"{self.name}/{self.model}"

    def run(self, prompt: str, repo: Path, commit: str, *, with_semble: bool) -> RunResult:
        """Run the agent on ``prompt``, retrying on empty output, and reset ``repo`` to ``commit`` afterward."""
        variant = variant_name(with_semble)
        try:
            for attempt in range(3):
                if attempt > 0:
                    print(f"    retry {attempt} (empty output — sleeping {_RETRY_SLEEP}s)...")
                    time.sleep(_RETRY_SLEEP)
                parsed, diff = self._run_once(prompt, repo, with_semble=with_semble)
                git_reset(repo, commit)
                if parsed.rate_limited:
                    return RunResult(variant=variant, backend=self.name, model=self.model, error=self._rate_limit_msg)
                if self._attempt_succeeded(parsed):
                    break
            return self._finalize(
                variant, parsed, diff, empty_output=not self._attempt_succeeded(parsed), with_semble=with_semble
            )
        except subprocess.TimeoutExpired:
            git_reset(repo, commit)
            return RunResult(variant=variant, backend=self.name, model=self.model, error=f"timed out after {_TIMEOUT}s")
        except Exception as exc:
            git_reset(repo, commit)
            return RunResult(variant=variant, backend=self.name, model=self.model, error=str(exc))

    def _finalize(
        self, variant: str, parsed: ParsedRun, diff: str, *, empty_output: bool, with_semble: bool
    ) -> RunResult:
        touched = get_modified_files(diff)
        tool_calls = parsed.tool_calls
        error = "empty output after retries" if empty_output else None
        return RunResult(
            variant=variant,
            backend=self.name,
            model=self.model,
            semble_version=self._semble_version if with_semble else "",
            tool_calls=tool_calls,
            cost_usd=parsed.cost_usd,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            num_turns=parsed.num_turns,
            touched_files=touched,
            patch=diff,
            bypass=any(is_bypass_call(t) for t in tool_calls),
            error=error,
        )
