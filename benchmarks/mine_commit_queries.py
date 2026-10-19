"""Mine a second query set from git history: commit subject -> the source files the commit changed.

The LLM-generated annotations share most of their words with their targets, which favours lexical signals.
Commit subjects are written by each project's developers about a change, so they test independently worded queries.

For each pinned repo, a history-only clone (commits and trees, no file contents) is made in
~/.cache/semble-bench-history. Walking back from the pinned revision (no merges, newest first), a commit becomes a
query when:
- its cleaned subject (PR/issue refs and conventional-commit prefixes removed) has 4-20 words and is not a
  merge/bump/release/typo/docs/CI/test chore;
- it touches at most 6 files, 1-3 of them non-test source files, all present at the pinned revision and inside the
  benchmark root. Those files are the relevant targets.

Up to 40 deduplicated queries per repo are written to benchmarks/annotations_commits/<repo>.json in the same format
as benchmarks/annotations. Use them with ``--query-set commits``:

    uv run python -m benchmarks.mine_commit_queries
    uv run python -m benchmarks.run_benchmark --query-set commits
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from benchmarks.data import BENCHMARKS_DIR, QUERY_SETS, load_repo_specs
from semble.ranking.penalties import _file_path_penalty

HISTORY_DIR = Path.home() / ".cache" / "semble-bench-history"
_PER_REPO = 40
_MAX_COMMITS = 4000
_MAX_FILES = 6
_CODE_EXTENSIONS = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go", ".java", ".kt", ".kts", ".scala", ".rb", ".php",
    ".rs", ".cs", ".swift", ".c", ".h", ".cc", ".cpp", ".hpp", ".hh", ".cxx", ".ex", ".exs", ".sh", ".bash",
    ".hs", ".lua", ".zig", ".m", ".mm",
}  # fmt: skip
_CHORE_RE = re.compile(
    r"\b(merge|bump|release|releasing|version|changelog|typo|typos|readme|docs?|documentation|ci|lint|linter|"
    r"format|formatting|reformat|revert|dependabot|deps|dependency|dependencies|copyright|license|whitespace|"
    r"cleanup|wip|test|tests|testing|spec|specs|rubocop|clippy|prettier|black|isort)\b",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*|[a-z]+(?:\([^)]*\))?!?:\s*)+", re.IGNORECASE)
_REFS_RE = re.compile(r"\s*(?:\(#\d+\)|#\d+|\(\s*\)|gh-\d+|GH-\d+)\s*")
_VERSION_RE = re.compile(r"\bv?\d+\.\d+")


def clean_subject(subject: str) -> str | None:
    """Return the subject as a query, or None if it is a chore or too short/long to describe code."""
    text = _PREFIX_RE.sub("", _REFS_RE.sub(" ", subject)).strip(" .:-")
    text = re.sub(r"\s+", " ", text)
    if not 4 <= len(text.split()) <= 20 or _CHORE_RE.search(text) or _VERSION_RE.search(text):
        return None
    if sum(ch.isalpha() for ch in text) < 0.6 * len(text):
        return None
    return text


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _history_clone(name: str, url: str) -> Path:
    path = HISTORY_DIR / f"{name}.git"
    if not path.exists():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--bare", "--filter=blob:none", url, str(path)], check=True)
    return path


def mine_repo(name: str, url: str, revision: str, benchmark_root: str | None) -> list[dict]:
    """Return up to _PER_REPO commit queries for one repo, newest first."""
    repo = _history_clone(name, url)
    present = set(_git(repo, "ls-tree", "-r", "--name-only", revision).splitlines())
    root = (benchmark_root or "").strip("/")
    # --no-renames: rename detection would fetch file contents into the blobless clone, one by one.
    log_format = "--format=%x1e%H%x1f%s"
    log = _git(repo, "log", f"-n{_MAX_COMMITS}", "--no-merges", "--no-renames", log_format, "--name-only", revision)
    queries: list[dict] = []
    seen: set[str] = set()
    for record in log.split("\x1e")[1:]:
        head, _, files_blob = record.partition("\n")
        sha, _, subject = head.partition("\x1f")
        query = clean_subject(subject)
        if not query or query.lower() in seen:
            continue
        files = [f for f in files_blob.split("\n") if f.strip()]
        source = [f for f in files if Path(f).suffix in _CODE_EXTENSIONS and _file_path_penalty(f) == 1.0]
        if not 1 <= len(source) <= 3 or len(files) > _MAX_FILES:
            continue
        if any(f not in present for f in source) or (root and not all(f.startswith(f"{root}/") for f in source)):
            continue
        seen.add(query.lower())
        queries.append({"query": query, "relevant": source, "secondary": [], "category": "commit", "commit": sha})
        if len(queries) >= _PER_REPO:
            break
    return queries


def main() -> None:
    """Mine commit queries for every pinned repo (or the --repo subset)."""
    parser = argparse.ArgumentParser(description="Mine commit-message queries for the semble benchmark.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    args = parser.parse_args()
    out_dir = QUERY_SETS["commits"]
    out_dir.mkdir(exist_ok=True)
    specs = load_repo_specs()
    total = 0
    for name, spec in sorted(specs.items()):
        if args.repo and name not in args.repo:
            continue
        queries = mine_repo(name, spec.url, spec.revision, spec.benchmark_root)
        (out_dir / f"{name}.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
        total += len(queries)
        print(f"{name:<22} {spec.language:<11} {len(queries):>3} queries", file=sys.stderr)
    print(f"{total} queries written to {out_dir.relative_to(BENCHMARKS_DIR.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
