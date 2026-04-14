from __future__ import annotations

import argparse
import subprocess
import sys

from benchmarks.common import BENCH_ROOT, load_repo_specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or update pinned benchmark repositories.")
    parser.add_argument("--repo", action="append", default=[], help="Restrict to one or more repo names.")
    parser.add_argument("--check", action="store_true", help="Only verify local checkouts against pinned revisions.")
    return parser.parse_args()


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _sync_repo(name: str, url: str, revision: str) -> None:
    repo_dir = BENCH_ROOT / name
    if not repo_dir.exists():
        print(f"cloning {name} -> {repo_dir}")
        _run("git", "clone", url, str(repo_dir))
    print(f"syncing {name} @ {revision[:12]}")
    _run("git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", revision)
    _run("git", "-C", str(repo_dir), "checkout", "--detach", revision)


def _check_repo(name: str, revision: str) -> str | None:
    repo_dir = BENCH_ROOT / name
    if not (repo_dir / ".git").exists():
        return f"{name}: missing checkout at {repo_dir}"
    head = _output("git", "-C", str(repo_dir), "rev-parse", "HEAD")
    if head != revision:
        return f"{name}: expected {revision}, found {head}"
    return None


def main() -> None:
    args = _parse_args()
    specs = load_repo_specs()
    selected = {name: spec for name, spec in specs.items() if not args.repo or name in args.repo}
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)

    if args.check:
        problems = [
            problem for name, spec in selected.items() if (problem := _check_repo(name, spec.revision)) is not None
        ]
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            raise SystemExit(1)
        print(f"Verified {len(selected)} pinned repo(s).")
        return

    for name, spec in selected.items():
        _sync_repo(name, spec.url, spec.revision)

    print(f"Synced {len(selected)} pinned repo(s).")


if __name__ == "__main__":
    main()
