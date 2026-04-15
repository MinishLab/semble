import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
from model2vec import StaticModel

from benchmarks.data import (
    RepoSpec,
    Target,
    Task,
    apply_task_filters,
    available_repo_specs,
    load_tasks,
    target_matches_location,
)
from semble import SembleIndex
from semble.index import DEFAULT_MODEL_NAME
from semble.types import SearchResult

_LATENCY_RUNS = 5
_DIRECT_TOP_K = 10


def _target_rank(results: list[SearchResult], target: Target) -> int | None:
    """Return the 1-based rank of the first result covering target, or None."""
    for index, result in enumerate(results, 1):
        chunk = result.chunk
        if target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target):
            return index
    return None


@dataclass(frozen=True)
class RepoResult:
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


def _dcg(relevances: list[int]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    """Compute NDCG@k given the ranks of relevant results and the total relevant count."""
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = _dcg([1] * min(k, n_relevant))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


def _evaluate(index: SembleIndex, tasks: list[Task], *, verbose: bool = False) -> tuple[float, float, list[float]]:
    """Return mean NDCG@5, NDCG@10, and median query latency (ms) across all tasks."""
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []

    for task in tasks:
        query_latencies: list[float] = []
        # Bind results
        results = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_DIRECT_TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(np.median(query_latencies))

        relevant_ranks = [rank for target in task.all_relevant if (rank := _target_rank(results, target)) is not None]
        n_relevant = sum(
            1
            for target in task.all_relevant
            if any(target_matches_location(c.file_path, c.start_line, c.end_line, target) for c in index.chunks)
        )
        q_ndcg5 = _ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = _ndcg_at_k(relevant_ranks, n_relevant, 10)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10

        if verbose:
            cat = task.category or "?"
            targets_str = ", ".join(
                t.path if not t.start_line else f"{t.path}:{t.start_line}-{t.end_line}" for t in task.all_relevant
            )
            top_files = [r.chunk.file_path for r in results[:5]]
            print(
                f"  [{cat:<12}] ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}  n_rel={n_relevant}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"               targets: {targets_str}", file=sys.stderr)
            print(f"               top-5:   {top_files}", file=sys.stderr)

    total = len(tasks)
    return ndcg5_sum / total, ndcg10_sum / total, latencies


def _print_summary(results: list[RepoResult]) -> None:
    """Print per-language and overall benchmark summary to stderr."""
    languages = sorted({result.language for result in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}
    columns = ["Avg", *[lang.title() for lang in languages]]

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_p90 = sum(r.p90_ms for r in results) / len(results)
    avg_p95 = sum(r.p95_ms for r in results) / len(results)
    avg_p99 = sum(r.p99_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)

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


def _bench_quality(
    repo_tasks: dict[str, list[Task]], model: StaticModel, specs: dict[str, RepoSpec], *, verbose: bool = False
) -> list[RepoResult]:
    """Run quality benchmarks (NDCG@5, NDCG@10, latency) for each repo."""
    print(
        f"{'Repo':<12} {'language':<12} {'chunks':>6} {'index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
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
        ndcg5, ndcg10, latencies = _evaluate(index, tasks, verbose=verbose)
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
        )
        results.append(result)
        print(
            f"{repo:<12} {spec.language:<12} {len(index.chunks):>6} "
            f"{index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms {p90:>7.2f}ms {p95:>7.2f}ms {p99:>7.2f}ms",
            file=sys.stderr,
        )
    return results


def _save_results(results: list[RepoResult]) -> None:
    """Write results to benchmarks/results/<sha>.json."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        sha = "unknown"

    languages = sorted({r.language for r in results})
    by_language = {lang: [r for r in results if r.language == lang] for lang in languages}

    output = {
        "sha": sha,
        "model": DEFAULT_MODEL_NAME,
        "summary": {
            "ndcg10": round(sum(r.ndcg10 for r in results) / len(results), 4),
            "p50_ms": round(sum(r.p50_ms for r in results) / len(results), 3),
            "p90_ms": round(sum(r.p90_ms for r in results) / len(results), 3),
            "p95_ms": round(sum(r.p95_ms for r in results) / len(results), 3),
            "p99_ms": round(sum(r.p99_ms for r in results) / len(results), 3),
            "index_ms": round(sum(r.index_ms for r in results) / len(results), 1),
        },
        "by_language": {
            lang: {
                "repos": len(grouped),
                "ndcg10": round(sum(r.ndcg10 for r in grouped) / len(grouped), 4),
                "p50_ms": round(sum(r.p50_ms for r in grouped) / len(grouped), 3),
                "p90_ms": round(sum(r.p90_ms for r in grouped) / len(grouped), 3),
                "p95_ms": round(sum(r.p95_ms for r in grouped) / len(grouped), 3),
                "p99_ms": round(sum(r.p99_ms for r in grouped) / len(grouped), 3),
                "index_ms": round(sum(r.index_ms for r in grouped) / len(grouped), 1),
            }
            for lang, grouped in by_language.items()
        },
        "repos": [asdict(r) for r in results],
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{sha[:12]}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path}", file=sys.stderr)


def main() -> None:
    """Parse arguments and run the selected benchmark mode."""
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
    model = StaticModel.from_pretrained(DEFAULT_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
    print(file=sys.stderr)
    repo_tasks: dict[str, list[Task]] = {}
    for task in tasks:
        repo_tasks.setdefault(task.repo, []).append(task)
    results = _bench_quality(repo_tasks, model, repo_specs, verbose=args.verbose)
    _print_summary(results)
    if not args.repo and not args.language:
        _save_results(results)


if __name__ == "__main__":
    main()
