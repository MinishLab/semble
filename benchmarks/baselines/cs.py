import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k
from benchmarks.tools import run_cs

_TOP_K = 10
_LATENCY_RUNS = 3


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float


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
            file_paths = run_cs(task.query, benchmark_dir, top_k=_TOP_K)
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


def main() -> None:
    """Run the cs (Code Spelunker) baseline benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark cs (Code Spelunker) on the semble benchmark suite.")
    add_filter_args(parser, verbose=True)
    args = parser.parse_args()

    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("cs (structural BM25 ranker, no index)", file=sys.stderr)
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
    print(f"{'Average (' + str(len(results)) + ')':<22} {'':<12} {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms", file=sys.stderr)

    summary = {
        "tool": "cs",
        "note": "structural BM25 ranker, tree-sitter aware, no persistent index",
        "repos": [
            {"repo": r.repo, "language": r.language, "ndcg10": round(r.ndcg10, 4), "p50_ms": round(r.p50_ms, 1)}
            for r in results
        ],
        "avg_ndcg10": round(avg_ndcg10, 4),
        "avg_p50_ms": round(avg_p50, 1),
    }
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results("cs", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
