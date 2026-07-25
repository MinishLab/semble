import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k

_BIN = "codebase-memory-mcp"
_TOP_K = 10
_LATENCY_RUNS = 3
_INDEX_TIMEOUT = 600
_SEARCH_TIMEOUT = 60

# codebase-memory-mcp normalizes non-ASCII/unsafe characters into the project name;
# benchmark repo names are already ASCII-safe slugs, but strip anything it wouldn't accept.
_PROJECT_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float
    index_ms: float


def _project_name(repo: str) -> str:
    return f"semble-bench-{_PROJECT_NAME_RE.sub('-', repo)}"


def _run_cli(args: list[str], *, timeout: int) -> dict | None:
    """Run `codebase-memory-mcp cli ...` and return the parsed structuredContent, or None on failure."""
    cmd = [_BIN, "cli", *args, "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if data.get("isError"):
        return None
    return data.get("structuredContent")


def _cleanup_index(project: str) -> None:
    _run_cli(["delete_project", "--project", project], timeout=30)


def _build_index(benchmark_dir: Path, project: str) -> tuple[bool, float]:
    """Index a repo into a fresh graph project; return (success, elapsed_ms)."""
    _cleanup_index(project)
    started = time.perf_counter()
    result = _run_cli(
        ["index_repository", "--repo-path", str(benchmark_dir), "--name", project, "--mode", "fast"],
        timeout=_INDEX_TIMEOUT,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result is not None, elapsed_ms


def _run_search(query: str, benchmark_dir: Path, project: str, *, top_k: int) -> list[str]:
    """Return absolute file paths from search_graph's BM25-ranked results."""
    result = _run_cli(
        ["search_graph", "--project", project, "--query", query, "--limit", str(top_k)],
        timeout=_SEARCH_TIMEOUT,
    )
    if result is None:
        return []
    seen: dict[str, None] = {}
    for item in result.get("results", []):
        rel = item.get("file_path", "")
        if rel:
            abs_path = str((benchmark_dir / rel).resolve())
            seen[abs_path] = None
    return list(seen)[:top_k]


def _evaluate_repo(
    tasks: list[Task],
    benchmark_dir: Path,
    project: str,
    *,
    verbose: bool = False,
) -> tuple[float, float]:
    """Return (mean ndcg@10, p50 latency ms) for a list of tasks."""
    ndcg10_sum = 0.0
    latencies: list[float] = []

    for task in tasks:
        query_latencies: list[float] = []
        file_paths: list[str] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            file_paths = _run_search(task.query, benchmark_dir, project, top_k=_TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(sorted(query_latencies)[_LATENCY_RUNS // 2])

        relevant_ranks = [rank for t in task.all_relevant if (rank := file_rank(file_paths, t.path)) is not None]
        q_ndcg10 = ndcg_at_k(relevant_ranks, len(task.all_relevant), _TOP_K)
        ndcg10_sum += q_ndcg10

        if verbose:
            print(
                f"  ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}  n_rel={len(task.all_relevant)}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"    targets: {', '.join(t.path for t in task.all_relevant)}", file=sys.stderr)
            print(f"    top-5:   {[Path(fp).name for fp in file_paths[:5]]}", file=sys.stderr)

    latencies.sort()
    return ndcg10_sum / len(tasks), latencies[len(latencies) // 2]


def _run_repo(spec: RepoSpec, tasks: list[Task], *, verbose: bool) -> RepoResult | None:
    """Index, evaluate, and clean up a single repo."""
    project = _project_name(spec.name)
    benchmark_dir = spec.benchmark_dir
    ok, index_ms = _build_index(benchmark_dir, project)
    if not ok:
        print(f"  SKIP: {spec.name} — codebase-memory-mcp indexing failed", file=sys.stderr)
        _cleanup_index(project)
        return None

    try:
        ndcg10, p50_ms = _evaluate_repo(tasks, benchmark_dir, project, verbose=verbose)
    finally:
        _cleanup_index(project)

    return RepoResult(repo=spec.name, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms, index_ms=index_ms)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark codebase-memory-mcp on the semble benchmark suite.")
    add_filter_args(parser, verbose=True)
    return parser.parse_args()


def main() -> None:
    """Run the codebase-memory-mcp baseline benchmark."""
    args = _parse_args()
    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("codebase-memory-mcp (search_graph: BM25 + structural boosting)", file=sys.stderr)
    print(f"{'Repo':<22} {'Language':<12} {'Index':>9} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        if args.verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)
        result = _run_repo(spec, repo_task_list, verbose=args.verbose)
        if result is None:
            continue
        results.append(result)
        print(
            f"{repo:<22} {spec.language:<12} {result.index_ms:>8.0f}ms {result.ndcg10:>8.3f} {result.p50_ms:>7.1f}ms",
            file=sys.stderr,
        )

    if not results:
        return

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)
    avg_label = f"Average ({len(results)})"
    print(
        f"{avg_label:<22} {'':<12} {avg_index:>8.0f}ms {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms",
        file=sys.stderr,
    )

    summary = {
        "tool": "codebase-memory-mcp",
        "note": "search_graph: SQLite FTS5 BM25 with structural boosting (Function/Method +10, Route +8, Class +5); "
        "'fast' index mode (no embeddings)",
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
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results("codebase-memory-mcp", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
