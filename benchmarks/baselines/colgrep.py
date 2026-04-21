import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    RepoSpec,
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    results_path,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k

_COLGREP = "colgrep"
_TOP_K = 10
_LATENCY_RUNS = 1  # subprocess calls are slow (~3s each); single run is sufficient


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float
    index_ms: float


def _run_colgrep(query: str, benchmark_dir: Path, top_k: int, *, code_only: bool = True) -> list[str]:
    """Return list of absolute file paths from colgrep JSON output."""
    cmd = [_COLGREP, "--force-cpu"]
    if code_only:
        cmd.append("--code-only")
    cmd += ["--json", "-k", str(top_k), query, str(benchmark_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [item["unit"]["file"] for item in data if "unit" in item and "file" in item["unit"]]


def _evaluate_repo(
    tasks: list[Task], benchmark_dir: Path, *, code_only: bool = True, verbose: bool = False
) -> tuple[float, float]:
    """Return (mean ndcg@10, p50 latency ms) for a list of tasks."""
    ndcg10_sum = 0.0
    latencies: list[float] = []

    for task in tasks:
        query_latencies: list[float] = []
        file_paths: list[str] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            file_paths = _run_colgrep(task.query, benchmark_dir, _TOP_K, code_only=code_only)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(sorted(query_latencies)[_LATENCY_RUNS // 2])

        deduped = list(dict.fromkeys(file_paths))

        relevant_ranks = [rank for t in task.all_relevant if (rank := file_rank(deduped, t.path)) is not None]
        q_ndcg10 = ndcg_at_k(relevant_ranks, len(task.all_relevant), _TOP_K)
        ndcg10_sum += q_ndcg10

        if verbose:
            print(
                f"  ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}  n_rel={len(task.all_relevant)}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"    targets: {', '.join(t.path for t in task.all_relevant)}", file=sys.stderr)
            print(f"    top-5:   {[Path(fp).name for fp in deduped[:5]]}", file=sys.stderr)

    latencies.sort()
    return ndcg10_sum / len(tasks), latencies[len(latencies) // 2]


def _init_index(path: Path) -> tuple[bool, float]:
    """Build (or rebuild) the colgrep index at path; return (non_empty, elapsed_ms).

    :param path: Directory to index.
    :return: Tuple of (non_empty, index_ms) where non_empty is False if colgrep reported 0 files.
    """
    subprocess.run([_COLGREP, "clear", str(path)], capture_output=True, timeout=30)
    cmd = [_COLGREP, "init", "--force-cpu", "-y", str(path)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    index_ms = (time.perf_counter() - started) * 1000
    if proc.returncode != 0:
        print(f"  WARNING: colgrep init failed for {path}: {proc.stderr.strip()}", file=sys.stderr)
    output = proc.stdout + proc.stderr
    non_empty = proc.returncode == 0 and "(0 files)" not in output
    return non_empty, index_ms


def _resolve_path(spec: RepoSpec) -> tuple[Path, float]:
    """Return the path ColGREP should index and elapsed index build time.

    Tries benchmark_dir first; if that yields 0 files falls back to checkout_dir,
    which is the project root ColGREP needs to discover the .git boundary.

    :param spec: Repo spec providing benchmark_dir and checkout_dir.
    :return: Tuple of (effective_path, index_ms).
    """
    path = spec.benchmark_dir
    ok, index_ms = _init_index(path)
    if ok:
        return path, index_ms
    # Jump straight to the project root — intermediate subdirectories can produce
    # misleading results (e.g. example-app files outranking core library files).
    root = spec.checkout_dir
    ok, index_ms = _init_index(root)
    if ok:
        print(f"  NOTE: {spec.name} — using checkout root {root} (benchmark_dir gave 0 files)", file=sys.stderr)
        return root, index_ms
    print(f"  WARN: {spec.name} — all candidate paths gave 0 files", file=sys.stderr)
    return path, index_ms


def _build_summary(results: list[RepoResult]) -> dict[str, object]:
    """Build the JSON summary dict from the current (possibly partial) results list."""
    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)
    return {
        "tool": "colgrep",
        "repos": [
            {
                "repo": r.repo,
                "language": r.language,
                "ndcg10": round(r.ndcg10, 4),
                "p50_ms": round(r.p50_ms, 1),
                "index_ms": round(r.index_ms, 0),
            }
            for r in results
        ],
        "avg_ndcg10": round(avg_ndcg10, 4),
        "avg_p50_ms": round(avg_p50, 1),
        "avg_index_ms": round(avg_index, 0),
    }


def _load_completed(out_path: Path) -> dict[str, RepoResult]:
    """Load any already-completed per-repo results from a previous (partial) run."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            entry["repo"]: RepoResult(
                repo=entry["repo"],
                language=entry["language"],
                ndcg10=entry["ndcg10"],
                p50_ms=entry["p50_ms"],
                index_ms=entry.get("index_ms", 0.0),
            )
            for entry in data.get("repos", [])
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ColGREP on the semble benchmark suite.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    parser.add_argument(
        "--no-code-only",
        action="store_true",
        help="Disable --code-only for all repos (overrides per-language default).",
    )
    return parser.parse_args()


def _run_repos(
    repo_tasks: dict[str, list[Task]],
    repo_specs: dict[str, RepoSpec],
    completed: dict[str, RepoResult],
    out_path: Path | None,
    *,
    no_code_only: bool = False,
    verbose: bool,
) -> list[RepoResult]:
    """Evaluate each repo and save incrementally; return all results."""
    results: list[RepoResult] = list(completed.values())

    print(f"{'Repo':<22} {'Language':<12} {'Index':>9} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)
    for r in sorted(results, key=lambda r: r.repo):
        print(
            f"{r.repo:<22} {r.language:<12} {r.index_ms:>8.0f}ms {r.ndcg10:>8.3f} {r.p50_ms:>7.1f}ms (cached)",
            file=sys.stderr,
        )

    for repo, repo_task_list in sorted(repo_tasks.items()):
        if repo in completed:
            continue
        spec = repo_specs[repo]
        # bash files (.sh, .bash) are excluded by --code-only; disable it for bash repos
        code_only = not no_code_only and spec.language != "bash"
        if verbose:
            print(f"\n--- {repo} (code_only={code_only}) ---", file=sys.stderr)
        path, index_ms = _resolve_path(spec)
        ndcg10, p50_ms = _evaluate_repo(repo_task_list, path, code_only=code_only, verbose=verbose)
        result = RepoResult(repo=repo, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms, index_ms=index_ms)
        results.append(result)
        print(f"{repo:<22} {spec.language:<12} {index_ms:>8.0f}ms {ndcg10:>8.3f} {p50_ms:>7.1f}ms", file=sys.stderr)
        if out_path:
            save_results("colgrep", _build_summary(results))

    return results


def main() -> None:
    """Run the ColGREP baseline benchmark."""
    args = _parse_args()
    is_full_run = not args.repo and not args.language

    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters.")

    repo_tasks = grouped_tasks(tasks)

    out_path = results_path("colgrep") if is_full_run else None
    completed = _load_completed(out_path) if out_path else {}
    if completed:
        print(f"Resuming: {len(completed)} repo(s) already done, skipping.", file=sys.stderr)

    results = _run_repos(
        repo_tasks, repo_specs, completed, out_path, no_code_only=args.no_code_only, verbose=args.verbose
    )

    if not results:
        return

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)
    avg_row = (
        f"{'Average (' + str(len(results)) + ')':<22} {'':<12} {avg_index:>8.0f}ms {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms"
    )
    print(avg_row, file=sys.stderr)

    summary = _build_summary(results)
    print(json.dumps(summary, indent=2))

    if is_full_run:
        print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
