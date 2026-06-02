import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from benchmarks.data import (
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k
from benchmarks.tools import run_ripgrep_count, run_ripgrep_keywords

_TOP_K = 10
_LATENCY_RUNS = 3


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float
    by_category: dict[str, float] = field(default_factory=dict)


def _evaluate_repo(
    tasks: list[Task],
    benchmark_dir: Path,
    *,
    keywords: bool = False,
    fixed_strings: bool = True,
    verbose: bool = False,
) -> tuple[float, float, dict[str, float]]:
    """Return (mean ndcg@10, p50 latency ms, per-category ndcg@10) for a list of tasks."""
    ndcg10_sum = 0.0
    latencies: list[float] = []
    category_ndcg10: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        query_latencies: list[float] = []
        file_paths: list[str] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            if keywords:
                file_paths = run_ripgrep_keywords(task.query, benchmark_dir, top_k=_TOP_K)
            else:
                file_paths = run_ripgrep_count(task.query, benchmark_dir, top_k=_TOP_K, fixed_strings=fixed_strings)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(sorted(query_latencies)[_LATENCY_RUNS // 2])

        relevant_ranks = [rank for t in task.all_relevant if (rank := file_rank(file_paths, t.path)) is not None]
        q_ndcg10 = ndcg_at_k(relevant_ranks, len(task.all_relevant), _TOP_K)
        ndcg10_sum += q_ndcg10
        category_ndcg10[task.category or "unknown"].append(q_ndcg10)

        if verbose:
            print(
                f"  ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}  n_rel={len(task.all_relevant)}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"    targets: {', '.join(t.path for t in task.all_relevant)}", file=sys.stderr)
            print(f"    top-5:   {[Path(fp).name for fp in file_paths[:5]]}", file=sys.stderr)

    latencies.sort()
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(category_ndcg10.items())}
    return ndcg10_sum / len(tasks), latencies[len(latencies) // 2], by_category


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ripgrep on the semble benchmark suite.")
    add_filter_args(parser, verbose=True)
    parser.add_argument(
        "--no-fixed-strings",
        dest="fixed_strings",
        action="store_false",
        default=True,
        help="Use regex mode instead of literal string matching.",
    )
    parser.add_argument(
        "--keywords",
        action="store_true",
        default=False,
        help=(
            "Split the query into keywords (dropping stopwords) and run a separate "
            "rg search per keyword, ranking files by distinct-keyword coverage. "
            "Models how an agent would use grep rather than passing raw queries."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the ripgrep baseline benchmark."""
    args = _parse_args()

    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    if args.keywords:
        mode_label = "keywords"
    elif args.fixed_strings:
        mode_label = "fixed-strings"
    else:
        mode_label = "regex"

    print(f"ripgrep ({mode_label})", file=sys.stderr)
    print(f"{'Repo':<22} {'Language':<12} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        if args.verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)
        ndcg10, p50_ms, by_category = _evaluate_repo(
            repo_task_list,
            spec.benchmark_dir,
            keywords=args.keywords,
            fixed_strings=args.fixed_strings,
            verbose=args.verbose,
        )
        results.append(
            RepoResult(repo=repo, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms, by_category=by_category)
        )
        print(f"{repo:<22} {spec.language:<12} {ndcg10:>8.3f} {p50_ms:>7.1f}ms", file=sys.stderr)

    if not results:
        return

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    print(f"{'-' * 22} {'-' * 12} {'-' * 8} {'-' * 8}", file=sys.stderr)
    print(f"{'Average (' + str(len(results)) + ')':<22} {'':<12} {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms", file=sys.stderr)

    all_categories = sorted({cat for r in results for cat in r.by_category})
    if all_categories:
        print(file=sys.stderr)
        print("By category (NDCG@10, mean over all repos)", file=sys.stderr)
        for cat in all_categories:
            vals = [r.by_category[cat] for r in results if cat in r.by_category]
            print(f"  {cat:<16}  {sum(vals) / len(vals):.3f}  (n={len(vals)} repos)", file=sys.stderr)

    cat_means = {
        cat: round(
            sum(r.by_category[cat] for r in results if cat in r.by_category)
            / sum(1 for r in results if cat in r.by_category),
            4,
        )
        for cat in all_categories
    }
    summary = {
        "tool": f"ripgrep-{mode_label}",
        "avg_ndcg10": round(avg_ndcg10, 4),
        "avg_p50_ms": round(avg_p50, 1),
        "by_category": cat_means,
        "repos": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results(f"ripgrep-{mode_label}", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
