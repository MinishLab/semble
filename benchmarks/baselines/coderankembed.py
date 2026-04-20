"""Benchmark CodeRankEmbed (semantic and hybrid) against semble on the benchmark suite.

CodeRankEmbed is a 137M-parameter transformer model from Nomic AI that uses
asymmetric encoding (separate query/document prompts).  It is substantially
slower to index and query than semble's default static model (potion-code-16M)
but may produce higher-quality embeddings.

Two modes are benchmarked:
  semantic  — dense retrieval only (no BM25)
  hybrid    — RRF fusion of dense + BM25 (same pipeline as run_benchmark.py)

Requires the benchmark extra:
    uv sync --extra benchmark
    uv run python -m benchmarks.baselines.coderankembed

Usage:
    uv run python -m benchmarks.baselines.coderankembed
    uv run python -m benchmarks.baselines.coderankembed --repo fastapi --verbose
    uv run python -m benchmarks.baselines.coderankembed --mode semantic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmarks.data import (
    RepoSpec,
    Target,
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    results_path,
    save_results,
    target_matches_location,
)
from semble import SembleIndex
from semble.types import SearchResult

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

_MODEL_NAME = "nomic-ai/CodeRankEmbed"
_TOP_K = 10
_LATENCY_RUNS = 3  # transformer inference is slow; keep runs low


# ---------------------------------------------------------------------------
# Asymmetric wrapper (query prompt vs. document encoding)
# ---------------------------------------------------------------------------


class _AsymmetricWrapper:
    """Wrap SentenceTransformer with asymmetric query/document prompts.

    Single-element lists are treated as queries; larger batches as documents.
    max_seq_length is capped to avoid OOM on CPU with long chunks.
    """

    def __init__(self, model: "SentenceTransformer", max_seq_length: int = 512) -> None:
        self._model = model
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with query or document prompt based on batch size."""
        if len(texts) == 1:
            return self._model.encode(texts, prompt_name="query", batch_size=1)  # type: ignore[return-value]
        return self._model.encode(texts, batch_size=1)  # type: ignore[return-value]


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
# Evaluation
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
            results = index.search(task.query, top_k=_TOP_K, mode=mode)
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _build_summary(results: list[RepoResult], modes: list[str]) -> dict[str, object]:
    """Build the JSON summary dict from the current (possibly partial) results list."""
    return {
        "tool": "coderankembed",
        "model": _MODEL_NAME,
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


def _load_completed(out_path: Path, modes: list[str]) -> dict[str, list[RepoResult]]:
    """Load repos where all requested modes are already saved in a previous run.

    :param out_path: Path to the existing results file (may not exist).
    :param modes: The modes we need for a repo to count as complete.
    :return: Mapping of repo name → list of RepoResult (one per mode).
    """
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        by_repo: dict[str, list[RepoResult]] = {}
        for entry in data.get("repos", []):
            r = RepoResult(**entry)
            by_repo.setdefault(r.repo, []).append(r)
        # Only count a repo as complete if every requested mode is present.
        return {repo: results for repo, results in by_repo.items() if {r.mode for r in results} >= set(modes)}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _bench(
    repo_tasks: dict[str, list[Task]],
    specs: dict[str, RepoSpec],
    model: "_AsymmetricWrapper",
    modes: list[str],
    out_path: Path | None,
    *,
    verbose: bool = False,
) -> list[RepoResult]:
    """Index each repo once, evaluate each mode, and save after every repo."""
    completed = _load_completed(out_path, modes) if out_path else {}
    if completed:
        print(f"Resuming: {len(completed)} repo(s) already done, skipping.", file=sys.stderr)

    # Seed results with already-completed repos.
    results: list[RepoResult] = [r for repo_results in completed.values() for r in repo_results]

    header = (
        f"{'Repo':<12} {'Language':<12} {'Mode':<10} {'Chunks':>6}"
        f" {'Index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
    )
    sep = f"{'-' * 12} {'-' * 12} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)

    # Print already-done rows so the table is complete on resume.
    for repo in sorted(completed):
        for r in completed[repo]:
            print(
                f"{r.repo:<12} {r.language:<12} {r.mode:<10} {r.chunks:>6}"
                f" {r.index_ms:>8.0f}ms {r.ndcg5:>8.3f} {r.ndcg10:>8.3f}"
                f" {r.p50_ms:>7.2f}ms {r.p90_ms:>7.2f}ms (cached)",
                file=sys.stderr,
            )

    for repo, tasks in sorted(repo_tasks.items()):
        if repo in completed:
            continue
        spec = specs[repo]
        if verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)

        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        index_ms = (time.perf_counter() - started) * 1000

        repo_results: list[RepoResult] = []
        for mode in modes:
            ndcg5, ndcg10, latencies, by_category = _evaluate(index, tasks, mode, verbose=verbose)
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
            repo_results.append(result)
            print(
                f"{repo:<12} {spec.language:<12} {mode:<10} {len(index.chunks):>6}"
                f" {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f}"
                f" {p50:>7.2f}ms {p90:>7.2f}ms",
                file=sys.stderr,
            )

        results.extend(repo_results)

        # Save after all modes for this repo are done.
        if out_path:
            save_results("coderankembed", _build_summary(results, modes))

    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CodeRankEmbed on the semble benchmark suite.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=["semantic", "hybrid"],
        help="Search mode(s) to evaluate (default: both).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CodeRankEmbed comparison benchmark."""
    if not _HAS_ST:
        raise SystemExit("sentence-transformers is required.\nInstall with: uv sync --extra benchmark")

    args = _parse_args()
    modes = args.mode or ["semantic", "hybrid"]
    is_full_run = not args.repo and not args.language

    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters.")

    print(f"Loading {_MODEL_NAME}...", file=sys.stderr)
    started = time.perf_counter()
    raw_model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    model = _AsymmetricWrapper(raw_model)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)
    print(file=sys.stderr)

    out_path = results_path("coderankembed") if is_full_run else None
    repo_tasks = grouped_tasks(tasks)
    results = _bench(repo_tasks, repo_specs, model, modes, out_path, verbose=args.verbose)

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

    summary = _build_summary(results, modes)
    print(json.dumps(summary, indent=2))

    if is_full_run:
        print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
