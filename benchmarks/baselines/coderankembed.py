import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

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
from benchmarks.metrics import ndcg_at_k, target_rank
from semble import SembleIndex
from semble.types import SearchResult

_MODEL_NAME = "nomic-ai/CodeRankEmbed"
_TOP_K = 10
_LATENCY_RUNS = 3  # transformer inference is slow; keep runs low


class _AsymmetricWrapper:
    """Wrap SentenceTransformer with asymmetric query/document prompts.

    Single-element lists are treated as queries; larger batches as documents.
    max_seq_length is capped to avoid OOM on CPU with long chunks.
    """

    def __init__(self, model: SentenceTransformer, max_seq_length: int = 512) -> None:
        self._model = model
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with query or document prompt based on batch size."""
        if len(texts) == 1:
            return self._model.encode(texts, prompt_name="query", batch_size=1)  # type: ignore[return-value]
        return self._model.encode(texts, batch_size=1)  # type: ignore[return-value]


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
    category_ndcg10: dict[str, list[float]] = defaultdict(list)

    for task in tasks:
        query_latencies: list[float] = []
        results: list[SearchResult]
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_TOP_K, mode=mode)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))

        relevant_ranks = [rank for t in task.all_relevant if (rank := target_rank(results, t)) is not None]
        n_relevant = len(task.all_relevant)
        q_ndcg5 = ndcg_at_k(relevant_ranks, n_relevant, 5)
        q_ndcg10 = ndcg_at_k(relevant_ranks, n_relevant, _TOP_K)
        ndcg5_sum += q_ndcg5
        ndcg10_sum += q_ndcg10
        category_ndcg10[task.category or "unknown"].append(q_ndcg10)

        if verbose:
            category = task.category or "?"
            targets_str = ", ".join(
                t.path if not t.start_line else f"{t.path}:{t.start_line}-{t.end_line}" for t in task.all_relevant
            )
            print(
                f"  [{category:<12}] ndcg@10={q_ndcg10:.3f}  ranks={relevant_ranks}"
                f"  n_rel={n_relevant}  q={task.query!r}",
                file=sys.stderr,
            )
            print(f"               targets: {targets_str}", file=sys.stderr)
            print(f"               top-5:   {[result.chunk.file_path for result in results[:5]]}", file=sys.stderr)

    total = len(tasks)
    by_category = {cat: sum(vals) / len(vals) for cat, vals in sorted(category_ndcg10.items())}
    return ndcg5_sum / total, ndcg10_sum / total, latencies, by_category


def _build_summary(results: list[RepoResult], modes: list[str]) -> dict[str, object]:
    """Build the JSON summary dict from the current (possibly partial) results list."""
    return {
        "tool": "coderankembed",
        "model": _MODEL_NAME,
        "by_mode": {
            mode: {
                "avg_ndcg10": round(
                    sum(result.ndcg10 for result in results if result.mode == mode)
                    / max(1, sum(1 for result in results if result.mode == mode)),
                    4,
                ),
                "avg_p50_ms": round(
                    sum(result.p50_ms for result in results if result.mode == mode)
                    / max(1, sum(1 for result in results if result.mode == mode)),
                    1,
                ),
            }
            for mode in modes
        },
        "repos": [asdict(result) for result in results],
    }


def _load_completed(out_path: Path, modes: list[str]) -> dict[str, list[RepoResult]]:
    """Load repos where all requested modes are already saved in a previous run."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        by_repo: dict[str, list[RepoResult]] = {}
        for entry in data.get("repos", []):
            result = RepoResult(**entry)
            by_repo.setdefault(result.repo, []).append(result)
        return {repo: results for repo, results in by_repo.items() if {result.mode for result in results} >= set(modes)}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _bench(
    repo_tasks: dict[str, list[Task]],
    specs: dict[str, RepoSpec],
    model: _AsymmetricWrapper,
    modes: list[str],
    out_path: Path | None,
    *,
    verbose: bool = False,
) -> list[RepoResult]:
    """Index each repo once, evaluate each mode, and save after every repo."""
    completed = _load_completed(out_path, modes) if out_path else {}
    if completed:
        print(f"Resuming: {len(completed)} repo(s) already done, skipping.", file=sys.stderr)

    results: list[RepoResult] = [r for repo_results in completed.values() for r in repo_results]

    header = (
        f"{'Repo':<12} {'Language':<12} {'Mode':<10} {'Chunks':>6}"
        f" {'Index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
    )
    print(header, file=sys.stderr)
    print(
        f"{'-' * 12} {'-' * 12} {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}",
        file=sys.stderr,
    )

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
                f" {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms {p90:>7.2f}ms",
                file=sys.stderr,
            )

        results.extend(repo_results)
        if out_path:
            save_results("coderankembed", _build_summary(results, modes))

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CodeRankEmbed on the semble benchmark suite.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument(
        "--mode", action="append", default=[], choices=["semantic", "hybrid"], help="Search mode(s) (default: both)."
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    return parser.parse_args()


def main() -> None:
    """Run the CodeRankEmbed baseline benchmark."""
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
    results = _bench(grouped_tasks(tasks), repo_specs, model, modes, out_path, verbose=args.verbose)

    if not results:
        return

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
