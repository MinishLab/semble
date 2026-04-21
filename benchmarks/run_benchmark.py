import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from model2vec import StaticModel

from benchmarks.data import (
    RepoSpec,
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    save_results,
)
from benchmarks.metrics import ndcg_at_k, target_rank
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.types import SearchResult

_LATENCY_RUNS = 5
_DIRECT_TOP_K = 10


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    chunks: int
    ndcg5: float
    ndcg10: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    index_ms: float
    by_category: dict[str, float] = field(default_factory=dict)


def _evaluate(
    index: SembleIndex, tasks: list[Task], *, verbose: bool = False
) -> tuple[float, float, list[float], dict[str, float]]:
    """Return mean NDCG@5, NDCG@10, median query latency (ms), and per-category NDCG@10."""
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []
    category_ndcg10: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        query_latencies: list[float] = []
        results: list[SearchResult]
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_DIRECT_TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))

        relevant_ranks = [rank for t in task.all_relevant if (rank := target_rank(results, t)) is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = ndcg_at_k(relevant_ranks, n_relevant, _DIRECT_TOP_K)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10
        category_ndcg10[task.category or "unknown"].append(q_ndcg10)

        if verbose:
            category = task.category or "?"
            targets_str = ", ".join(
                t.path if not t.start_line else f"{t.path}:{t.start_line}-{t.end_line}" for t in task.all_relevant
            )
            top_files = [r.chunk.file_path for r in results[:5]]
            print(
                f"  [{category:<12}] ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}"
                f"  n_rel={n_relevant}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"               targets: {targets_str}", file=sys.stderr)
            print(f"               top-5:   {top_files}", file=sys.stderr)

    total = len(tasks)
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(category_ndcg10.items())}
    return ndcg5_sum / total, ndcg10_sum / total, latencies, by_category


def _print_summary(results: list[RepoResult]) -> None:
    """Print per-language and overall benchmark summary to stderr."""
    languages = sorted({result.language for result in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}
    columns = ["Avg", *[lang.title() for lang in languages]]

    language_ndcg10 = [
        sum(r.ndcg10 for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p50 = [
        sum(r.p50_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p90 = [
        sum(r.p90_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p95 = [
        sum(r.p95_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_p99 = [
        sum(r.p99_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    language_index = [
        sum(r.index_ms for r in language_results) / len(language_results) for language_results in by_language.values()
    ]
    avg_ndcg10 = sum(language_ndcg10) / len(language_ndcg10)
    avg_p50 = sum(language_p50) / len(language_p50)
    avg_p90 = sum(language_p90) / len(language_p90)
    avg_p95 = sum(language_p95) / len(language_p95)
    avg_p99 = sum(language_p99) / len(language_p99)
    avg_index = sum(language_index) / len(language_index)

    print(file=sys.stderr)
    print("By language", file=sys.stderr)
    for language, grouped in by_language.items():
        print(
            f"  {language}: repos={len(grouped)}"
            + f"  ndcg@5={sum(r.ndcg5 for r in grouped) / len(grouped):.3f}"
            + f"  ndcg@10={sum(r.ndcg10 for r in grouped) / len(grouped):.3f}"
            + f"  p50={sum(r.p50_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p90={sum(r.p90_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p95={sum(r.p95_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  p99={sum(r.p99_ms for r in grouped) / len(grouped):.2f}ms"
            + f"  index={sum(r.index_ms for r in grouped) / len(grouped):.0f}ms",
            file=sys.stderr,
        )

    print(file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print("Hybrid benchmark by language", file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print(f"\n  {'Metric':<28}  " + "  ".join(f"{column:>9}" for column in columns), file=sys.stderr)
    print(f"  {'-' * 28}  " + "  ".join(f"{'-' * 9:>9}" for _ in columns), file=sys.stderr)

    ndcg_row = [f"{avg_ndcg10:>9.3f}"]
    p50_row = [f"{avg_p50:>8.2f}ms"]
    p90_row = [f"{avg_p90:>8.2f}ms"]
    p95_row = [f"{avg_p95:>8.2f}ms"]
    p99_row = [f"{avg_p99:>8.2f}ms"]
    index_row = [f"{avg_index:>7.0f}ms"]
    for language, language_results in by_language.items():
        ndcg_row.append(f"{sum(r.ndcg10 for r in language_results) / len(language_results):>9.3f}")
        p50_row.append(f"{sum(r.p50_ms for r in language_results) / len(language_results):>8.2f}ms")
        p90_row.append(f"{sum(r.p90_ms for r in language_results) / len(language_results):>8.2f}ms")
        p95_row.append(f"{sum(r.p95_ms for r in language_results) / len(language_results):>8.2f}ms")
        p99_row.append(f"{sum(r.p99_ms for r in language_results) / len(language_results):>8.2f}ms")
        index_row.append(f"{sum(r.index_ms for r in language_results) / len(language_results):>7.0f}ms")

    print(f"  {'NDCG@10':<28}  " + "  ".join(ndcg_row), file=sys.stderr)
    print(f"  {'q-p50':<28}  " + "  ".join(p50_row), file=sys.stderr)
    print(f"  {'q-p90':<28}  " + "  ".join(p90_row), file=sys.stderr)
    print(f"  {'q-p95':<28}  " + "  ".join(p95_row), file=sys.stderr)
    print(f"  {'q-p99':<28}  " + "  ".join(p99_row), file=sys.stderr)
    print(f"  {'index':<28}  " + "  ".join(index_row), file=sys.stderr)

    all_categories = sorted({cat for r in results for cat in r.by_category})
    if all_categories:
        print(file=sys.stderr)
        print("By category (NDCG@10, mean over all repos)", file=sys.stderr)
        for cat in all_categories:
            vals = [r.by_category[cat] for r in results if cat in r.by_category]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            print(f"  {cat:<16}  {mean_val:.3f}  (n={len(vals)} repos)", file=sys.stderr)


def _bench_quality(
    repo_tasks: dict[str, list[Task]], model: StaticModel, specs: dict[str, RepoSpec], *, verbose: bool = False
) -> list[RepoResult]:
    """Run quality benchmarks (NDCG@5, NDCG@10, latency) for each repo."""
    print(
        f"{'Repo':<12} {'Language':<12} {'Chunks':>6} {'index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
        f" {'p95':>8} {'p99':>8}",
        file=sys.stderr,
    )
    print(
        f"{'-' * 12} {'-' * 12} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}",
        file=sys.stderr,
    )
    results: list[RepoResult] = []
    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        index_ms = (time.perf_counter() - started) * 1000
        ndcg5, ndcg10, latencies, by_category = _evaluate(index, tasks, verbose=verbose)
        p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99]).tolist()
        result = RepoResult(
            repo=repo,
            language=spec.language,
            chunks=len(index.chunks),
            ndcg5=ndcg5,
            ndcg10=ndcg10,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            index_ms=index_ms,
            by_category=by_category,
        )
        results.append(result)
        print(
            f"{repo:<12} {spec.language:<12} {len(index.chunks):>6} "
            f"{index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms {p90:>7.2f}ms {p95:>7.2f}ms {p99:>7.2f}ms",
            file=sys.stderr,
        )
    return results


def _save_results(results: list[RepoResult]) -> None:
    """Write results to benchmarks/results/semble-hybrid-<sha12>.json."""
    languages = sorted({r.language for r in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}

    lang_means = {
        lang: {
            "ndcg10": sum(r.ndcg10 for r in grouped) / len(grouped),
            "p50_ms": sum(r.p50_ms for r in grouped) / len(grouped),
            "p90_ms": sum(r.p90_ms for r in grouped) / len(grouped),
            "p95_ms": sum(r.p95_ms for r in grouped) / len(grouped),
            "p99_ms": sum(r.p99_ms for r in grouped) / len(grouped),
            "index_ms": sum(r.index_ms for r in grouped) / len(grouped),
        }
        for lang, grouped in by_language.items()
    }
    all_categories: set[str] = set()
    for r in results:
        all_categories.update(r.by_category)
    cat_means: dict[str, float] = {}
    for cat in sorted(all_categories):
        vals = [r.by_category[cat] for r in results if cat in r.by_category]
        cat_means[cat] = round(sum(vals) / len(vals), 4) if vals else 0.0

    n_repos = len(results)
    output = {
        "tool": "semble-hybrid",
        "model": _DEFAULT_MODEL_NAME,
        "summary": {
            "ndcg10": round(sum(r.ndcg10 for r in results) / n_repos, 4),
            "p50_ms": round(sum(r.p50_ms for r in results) / n_repos, 3),
            "p90_ms": round(sum(r.p90_ms for r in results) / n_repos, 3),
            "p95_ms": round(sum(r.p95_ms for r in results) / n_repos, 3),
            "p99_ms": round(sum(r.p99_ms for r in results) / n_repos, 3),
            "index_ms": round(sum(r.index_ms for r in results) / n_repos, 1),
            "by_category": cat_means,
        },
        "by_language": {
            lang: {
                "repos": len(by_language[lang]),
                "ndcg10": round(v["ndcg10"], 4),
                "p50_ms": round(v["p50_ms"], 3),
                "p90_ms": round(v["p90_ms"], 3),
                "p95_ms": round(v["p95_ms"], 3),
                "p99_ms": round(v["p99_ms"], 3),
                "index_ms": round(v["index_ms"], 1),
            }
            for lang, v in lang_means.items()
        },
        "repos": [asdict(r) for r in results],
    }

    out_path = save_results("semble-hybrid", output)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


def main() -> None:
    """Parse arguments and run the semble hybrid benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark hybrid semble search across the pinned benchmark repos.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    args = parser.parse_args()
    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters.")
    print("Loading model...", file=sys.stderr)
    started = time.perf_counter()
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
    print(file=sys.stderr)
    repo_tasks = grouped_tasks(tasks)
    results = _bench_quality(repo_tasks, model, repo_specs, verbose=args.verbose)
    _print_summary(results)
    if not args.repo and not args.language:
        _save_results(results)


if __name__ == "__main__":
    main()
