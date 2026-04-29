import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k

_TOP_K = 10
_LATENCY_RUNS = 3


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float


def _run_probe(query: str, benchmark_dir: Path, *, top_k: int, timeout: int = 30) -> list[str]:
    """Return file paths from probe JSON output, deduplicated and capped at top_k."""
    cmd = [
        "probe",
        "search",
        query,
        str(benchmark_dir),
        "--format",
        "json",
        "--max-results",
        str(top_k * 3),  # probe returns chunk-level results; over-fetch and dedup
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    # probe prefixes stdout with non-JSON header lines ("Pattern: ...\nPath: ...\n")
    # before the JSON object; skip to the first '{'.
    json_start = proc.stdout.find("{")
    if json_start < 0:
        return []
    try:
        data = json.loads(proc.stdout[json_start:])
    except json.JSONDecodeError:
        return []
    seen: dict[str, None] = {}
    for item in data.get("results", []):
        fp = item.get("file", "")
        if fp:
            seen[fp] = None
    return list(seen)[:top_k]


def _evaluate_repo(
    tasks: list[Task],
    benchmark_dir: Path,
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
            file_paths = _run_probe(task.query, benchmark_dir, top_k=_TOP_K)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark probe on the semble benchmark suite.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


def main() -> None:
    """Run the probe baseline benchmark."""
    args = _parse_args()
    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )

    print("probe (bm25, tree-sitter)", file=sys.stderr)
    print("NOTE: probe uses keyword ranking; natural-language queries disadvantage it.", file=sys.stderr)
    print(f"{'Repo':<22} {'Language':<12} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        if args.verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)
        ndcg10, p50_ms = _evaluate_repo(repo_task_list, spec.benchmark_dir, verbose=args.verbose)
        results.append(RepoResult(repo=repo, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms))
        print(f"{repo:<22} {spec.language:<12} {ndcg10:>8.3f} {p50_ms:>7.1f}ms", file=sys.stderr)

    if not results:
        return

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    print(f"{'-' * 22} {'-' * 12} {'-' * 8} {'-' * 8}", file=sys.stderr)
    avg_label = f"Average ({len(results)})"
    print(
        f"{avg_label:<22} {'':<12} {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms",
        file=sys.stderr,
    )

    summary = {
        "tool": "probe",
        "note": "BM25 + tree-sitter; no embedding model, no persistent index; natural-language queries disadvantage it",
        "repos": [
            {"repo": r.repo, "language": r.language, "ndcg10": round(r.ndcg10, 4), "p50_ms": round(r.p50_ms, 1)}
            for r in results
        ],
        "avg_ndcg10": round(avg_ndcg10, 4),
        "avg_p50_ms": round(avg_p50, 1),
    }
    save_results("probe", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
