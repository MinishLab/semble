from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from model2vec import StaticModel

from benchmarks.common import (
    Target,
    Task,
    apply_task_filters,
    available_repo_specs,
    count_indexed_targets,
    grouped_tasks,
    load_tasks,
    target_matches_location,
)
from semble import SembleIndex
from semble.types import SearchResult

_CACHE_DIR = Path("/tmp/semble-bench-cache")
_MODEL_NAME = "Pringled/potion-code-16M"
_LATENCY_RUNS = 5
_DIRECT_TOP_K = 10


def _target_rank(results: list[SearchResult], target: Target) -> int | None:
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
    cold_ms: float | None = None
    warm_ms: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hybrid semble search across the pinned benchmark repos.")
    parser.add_argument("--cache", action="store_true", help="Show cold vs warm index time using the disk cache.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


def _is_relevant(result: SearchResult, task: Task) -> bool:
    chunk = result.chunk
    return any(
        target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target)
        for target in task.all_relevant
    )


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = _dcg([1] * min(k, n_relevant))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


def _evaluate(index: SembleIndex, tasks: list[Task], *, verbose: bool = False) -> tuple[float, float, float]:
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []

    for task in tasks:
        query_latencies: list[float] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_DIRECT_TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(sorted(query_latencies)[_LATENCY_RUNS // 2])

        chunk_results = results[:_DIRECT_TOP_K]
        relevant_ranks = [
            rank for target in task.all_relevant if (rank := _target_rank(chunk_results, target)) is not None
        ]
        n_relevant = count_indexed_targets(index.chunks, task.all_relevant)
        q_ndcg5 = _ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = _ndcg_at_k(relevant_ranks, n_relevant, 10)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10

        if verbose:
            cat = task.category or "?"
            targets_str = ", ".join(
                t.path if not t.start_line else f"{t.path}:{t.start_line}-{t.end_line}" for t in task.all_relevant
            )
            top_files = [r.chunk.file_path for r in chunk_results[:5]]
            print(
                f"  [{cat:<12}] ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}  n_rel={n_relevant}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"               targets: {targets_str}", file=sys.stderr)
            print(f"               top-5:   {top_files}", file=sys.stderr)

    total = len(tasks)
    latencies.sort()
    return ndcg5_sum / total, ndcg10_sum / total, latencies[len(latencies) // 2]


def _print_group_summary(results: list[RepoResult], group_by: str) -> None:
    print(file=sys.stderr)
    print(f"By {group_by}", file=sys.stderr)
    groups = sorted({getattr(result, group_by) for result in results})
    for value in groups:
        grouped = [result for result in results if getattr(result, group_by) == value]
        print(
            "  "
            + f"{value}: repos={len(grouped)}  ndcg@5={sum(r.ndcg5 for r in grouped) / len(grouped):.3f}"
            + f"  ndcg@10={sum(r.ndcg10 for r in grouped) / len(grouped):.3f}"
            + f"  p50={sum(r.p50_ms for r in grouped) / len(grouped):.2f}ms",
            file=sys.stderr,
        )


def _print_language_table(results: list[RepoResult]) -> None:
    languages = ["python", "javascript", "java", "go", "php", "ruby"]
    present = [language for language in languages if any(result.language == language for result in results)]
    columns = ["Avg", *[language.title() for language in present]]

    avg_ndcg10 = sum(result.ndcg10 for result in results) / len(results)
    avg_p50 = sum(result.p50_ms for result in results) / len(results)

    print(file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print("Hybrid benchmark by language", file=sys.stderr)
    print(f"{'=' * 104}", file=sys.stderr)
    print(f"\n  {'Metric':<28}  " + "  ".join(f"{column:>9}" for column in columns), file=sys.stderr)
    print(f"  {'-' * 28}  " + "  ".join(f"{'-' * 9:>9}" for _ in columns), file=sys.stderr)

    ndcg_row = [f"{avg_ndcg10:>9.3f}"]
    p50_row = [f"{avg_p50:>8.2f}ms"]
    for language in present:
        language_results = [result for result in results if result.language == language]
        ndcg_row.append(f"{sum(result.ndcg10 for result in language_results) / len(language_results):>9.3f}")
        p50_row.append(f"{sum(result.p50_ms for result in language_results) / len(language_results):>8.2f}ms")

    print(f"  {'NDCG@10':<28}  " + "  ".join(ndcg_row), file=sys.stderr)
    print(f"  {'q-p50':<28}  " + "  ".join(p50_row), file=sys.stderr)


def _bench_quality(repo_tasks: dict[str, list[Task]], model: StaticModel, *, verbose: bool = False) -> list[RepoResult]:
    print(
        f"{'Repo':<12} {'language':<12} {'chunks':>6} {'index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8}",
        file=sys.stderr,
    )
    print(f"{'-' * 12} {'-' * 12} {'-' * 6} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8}", file=sys.stderr)
    results: list[RepoResult] = []
    specs = available_repo_specs()
    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        index_ms = (time.perf_counter() - started) * 1000
        ndcg5, ndcg10, p50_ms = _evaluate(index, tasks, verbose=verbose)
        result = RepoResult(
            repo=repo, language=spec.language, chunks=len(index.chunks), ndcg5=ndcg5, ndcg10=ndcg10, p50_ms=p50_ms
        )
        results.append(result)
        print(
            f"{repo:<12} {spec.language:<12} {len(index.chunks):>6} {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50_ms:>7.2f}ms",
            file=sys.stderr,
        )
    return results


def _bench_cache(repo_tasks: dict[str, list[Task]], model: StaticModel) -> list[RepoResult]:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {_CACHE_DIR}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"{'Repo':<12} {'language':<12} {'chunks':>6} {'cold':>9} {'warm':>9} {'speedup':>8} {'NDCG@10':>8}",
        file=sys.stderr,
    )
    print(f"{'-' * 12} {'-' * 12} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)
    results: list[RepoResult] = []
    specs = available_repo_specs()
    model_ns = _MODEL_NAME.replace("/", "--")
    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        namespace_dir = _CACHE_DIR / model_ns
        if namespace_dir.exists():
            shutil.rmtree(namespace_dir)
        started = time.perf_counter()
        cold = SembleIndex.from_path(spec.benchmark_dir, model=model, cache_dir=_CACHE_DIR, model_name=_MODEL_NAME)
        cold_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        warm = SembleIndex.from_path(spec.benchmark_dir, model=model, cache_dir=_CACHE_DIR, model_name=_MODEL_NAME)
        warm_ms = (time.perf_counter() - started) * 1000
        ndcg5, ndcg10, p50_ms = _evaluate(warm, tasks)
        result = RepoResult(
            repo=repo,
            language=spec.language,
            chunks=len(cold.chunks),
            ndcg5=ndcg5,
            ndcg10=ndcg10,
            p50_ms=p50_ms,
            cold_ms=cold_ms,
            warm_ms=warm_ms,
        )
        results.append(result)
        speedup = cold_ms / warm_ms if warm_ms > 0 else float("inf")
        print(
            f"{repo:<12} {spec.language:<12} {len(cold.chunks):>6} {cold_ms:>8.0f}ms {warm_ms:>8.0f}ms {speedup:>7.1f}x {ndcg10:>8.3f}",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print("Warm time still includes file walk plus BM25/Vicinity rebuild; only embedding is skipped.", file=sys.stderr)
    return results


def main() -> None:
    args = _parse_args()
    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters.")
    print("Loading model...", file=sys.stderr)
    started = time.perf_counter()
    model = StaticModel.from_pretrained(_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f} ms", file=sys.stderr)
    print(file=sys.stderr)
    repo_tasks = grouped_tasks(tasks)
    results = _bench_cache(repo_tasks, model) if args.cache else _bench_quality(repo_tasks, model, verbose=args.verbose)
    _print_group_summary(results, "language")
    _print_language_table(results)


if __name__ == "__main__":
    main()
