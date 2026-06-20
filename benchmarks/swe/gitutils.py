from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_CLONE_TIMEOUT = 180  # seconds per git clone/checkout


def _clone_at_commit(repo: str, commit: str, dest: Path) -> None:
    if dest.exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}", str(dest)],
        check=True,
        timeout=_CLONE_TIMEOUT,
    )
    subprocess.run(["git", "checkout", "--quiet", commit], cwd=dest, check=True, timeout=_CLONE_TIMEOUT)


def _git_diff(repo: Path) -> str:
    return subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True).stdout


def _git_reset(repo: Path, commit: str) -> None:
    """Hard-reset to the task's base commit and remove all untracked/ignored files."""
    subprocess.run(["git", "reset", "--hard", commit], cwd=repo, capture_output=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=repo, capture_output=True)


def _changed_files(diff_text: str) -> list[str]:
    """Paths touched by a unified diff — used for both gold-patch parsing and `git diff` output."""
    return [line[6:] for line in diff_text.splitlines() if line.startswith("+++ b/")]
