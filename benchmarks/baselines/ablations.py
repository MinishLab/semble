"""Benchmark semble ablations: BM25-only and semantic-only modes, with and without semble ranking.

Four modes form a complete ablation ladder:

    bm25             — raw BM25, no embeddings, no semble ranking
    semantic         — raw dense search, no BM25, no semble ranking
    semble-bm25      — BM25 retrieval + full semble ranking stack (alpha=0)
    semble-semantic  — semantic retrieval + full semble ranking stack (alpha=1)

Together with semble-hybrid (run_benchmark.py) this isolates the contribution
of each retrieval source and the ranking layer independently.

Usage:
    uv run python -m benchmarks.baselines.ablations
    uv run python -m benchmarks.baselines.ablations --repo fastapi --verbose
    uv run python -m benchmarks.baselines.ablations --mode bm25
    uv run python -m benchmarks.baselines.ablations --mode semble-semantic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
from model2vec import StaticModel

from benchmarks.data import (
    RepoSpec,
    Target,
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    save_results,
    target_matches_location,
)
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.types import SearchResult

_TOP_K = 10
_LATENCY_RUNS = 5

_MODES = ["bm25", "semantic", "semble-bm25", "semble-semantic"]

# Maps mode name -> (search_mode, alpha) for index.search()
# alpha=None  → use raw mode (no ranking pipeline)
# alpha=0.0   → hybrid pipeline, BM25-only input
# alpha=1.0   → hybrid pipeline, semantic-only input
_MODE_PARAMS: dict[str, tuple[str, float | None]] = {
    "bm25": ("bm25", None),
    "semantic": ("semantic", None),
    "semble-bm25": ("hybrid", 0.0),
    "semble-semantic": ("hybrid", 1.0),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result for one search mode."""

    repo: str
    language: str
    mode: str
    chunks: int
    ndcg5: float
    ndcg10: float
    p50_ms: float
    p90_ms: float
    index_ms: float
    by_category: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NDCG helpers
# ---------------------------------------------------------------------------


def _dcg(relevances: list[int]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    """Compute NDCG@k given 1-based ranks of relevant results and total relevant count."""
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = _dcg([1] * min(k, n_relevant))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation (mirrors run_benchmark.py _evaluate)
# ---------------------------------------------------------------------------


def _target_rank(results: list[SearchResult], target: Target) -> int | None:
    """Return 1-based rank of the first result covering target, or None."""
    for index, result in enumerate(results, 1):
        chunk = result.chunk
        if target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target):
            return index
    return None


def _evaluate(
    index: SembleIndex,
    tasks: list[Task],
    mode: str,
    alpha: float | None,
    *,
    verbose: bool = False,
) -> tuple[float, float, list[float], dict[str, float]]:
    """Return (mean NDCG@5, NDCG@10, latency list ms, per-category NDCG@10)."""
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    latencies: list[float] = []
    cat_ndcg10: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        query_latencies: list[float] = []
        results: list[SearchResult] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_TOP_K, mode=mode, alpha=alpha)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))

        relevant_ranks = [rank for target in task.all_relevant if (rank := _target_rank(results, target)) is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = _ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = _ndcg_at_k(relevant_ranks, n_relevant, _TOP_K)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10
        cat_ndcg10[task.category or "unknown"].append(q_ndcg10)

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
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(cat_ndcg10.items())}
    return ndcg5_sum / total, ndcg10_sum / total, latencies, by_category


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _bench(
    repo_tasks: dict[str, list[Task]],
    specs: dict[str, RepoSpec],
    model: StaticModel,
    modes: list[str],
    *,
    verbose: bool = False,
) -> list[RepoResult]:
    """Index each repo once then evaluate each requested mode."""
    results: list[RepoResult] = []

    header = (
        f"{'Repo':<12} {'Language':<12} {'Mode':<10} {'Chunks':>6}"
        f" {'Index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
    )
    sep = f"{'-' * 12} {'-' * 12} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)

    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        if verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)

        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        index_ms = (time.perf_counter() - started) * 1000

        for mode in modes:
            search_mode, alpha = _MODE_PARAMS[mode]
            ndcg5, ndcg10, latencies, by_category = _evaluate(index, tasks, search_mode, alpha, verbose=verbose)
            p50, p90 = np.percentile(latencies, [50, 90]).tolist()
            result = RepoResult(
                repo=repo,
                language=spec.language,
                mode=mode,
                chunks=len(index.chunks),
                ndcg5=ndcg5,
                ndcg10=ndcg10,
                p50_ms=p50,
                p90_ms=p90,
                index_ms=index_ms,
                by_category=by_category,
            )
            results.append(result)
            print(
                f"{repo:<12} {spec.language:<12} {mode:<10} {len(index.chunks):>6}"
                f" {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f}"
                f" {p50:>7.2f}ms {p90:>7.2f}ms",
                file=sys.stderr,
            )

    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark semble BM25-only and semantic-only modes (ablations).")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=_MODES,
        help="Mode(s) to evaluate (default: both).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the semble ablation benchmarks."""
    args = _parse_args()
    modes = args.mode or _MODES

    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters.")

    print("Loading model...", file=sys.stderr)
    started = time.perf_counter()
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)
    print(file=sys.stderr)

    repo_tasks = grouped_tasks(tasks)
    results = _bench(repo_tasks, repo_specs, model, modes, verbose=args.verbose)

    if not results:
        return

    # Summary per mode
    print(file=sys.stderr)
    for mode in modes:
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        avg_ndcg10 = sum(r.ndcg10 for r in mode_results) / len(mode_results)
        avg_p50 = sum(r.p50_ms for r in mode_results) / len(mode_results)
        print(
            f"  {mode:<10}  avg ndcg@10={avg_ndcg10:.3f}  avg p50={avg_p50:.1f}ms  ({len(mode_results)} repos)",
            file=sys.stderr,
        )

    summary = {
        "tool": "semble-ablations",
        "model": _DEFAULT_MODEL_NAME,
        "by_mode": {
            mode: {
                "avg_ndcg10": round(
                    sum(r.ndcg10 for r in results if r.mode == mode)
                    / max(1, sum(1 for r in results if r.mode == mode)),
                    4,
                ),
                "avg_p50_ms": round(
                    sum(r.p50_ms for r in results if r.mode == mode)
                    / max(1, sum(1 for r in results if r.mode == mode)),
                    1,
                ),
            }
            for mode in modes
        },
        "repos": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results("semble-ablations", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
