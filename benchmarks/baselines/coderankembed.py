import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    results_path,
    save_results,
)
from benchmarks.metrics import ndcg_at_k, target_rank
from semble.index.create import create_index_from_path
from semble.index.index import SembleIndex
from semble.types import ContentType, SearchResult

_MODEL_NAME = "nomic-ai/CodeRankEmbed"
_TOP_K = 10
_LATENCY_RUNS = 3  # transformer inference is slow; keep runs low
_ALPHA = 1.0  # SembleIndex.search()'s alpha: 1.0 = full semantic weight

_UNSET = object()


class _AsymmetricWrapper:
    """Wrap SentenceTransformer with asymmetric query/document prompts.

    semble only passes use_multiprocessing during index-time calls, never at query time, so its
    presence is a reliable query/document discriminator (batch size is not: single-chunk files are
    real one-element document batches).
    """

    def __init__(self, model: SentenceTransformer, max_seq_length: int = 512) -> None:
        self._model = model
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str], use_multiprocessing: object = _UNSET) -> np.ndarray:
        """Encode with the query prompt only when use_multiprocessing wasn't passed."""
        text_list = list(texts)
        if use_multiprocessing is _UNSET:
            return self._model.encode(text_list, prompt_name="query", batch_size=1)  # type: ignore[return-value]
        return self._model.encode(text_list, batch_size=1)  # type: ignore[return-value]


def _build_index(benchmark_dir: Path, model: _AsymmetricWrapper) -> SembleIndex:
    """Build a SembleIndex using CodeRankEmbed embeddings for both BM25 enrichment and dense search."""
    bm25_index, semantic_index, chunks, _manifest = create_index_from_path(
        benchmark_dir,
        model=model,  # type: ignore[arg-type]
        content=(ContentType.CODE,),  # type: ignore[arg-type]
    )
    return SembleIndex(
        model=model,  # type: ignore[arg-type]
        bm25_index=bm25_index,
        semantic_index=semantic_index,
        chunks=chunks,
        model_path=_MODEL_NAME,
        root=benchmark_dir,
    )


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
    index_ms: float
    by_category: dict[str, float] = field(default_factory=dict)


def _evaluate(
    index: SembleIndex,
    tasks: list[Task],
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
        results: list[SearchResult] = []
        for _ in range(_LATENCY_RUNS):
            started = time.perf_counter()
            results = index.search(task.query, top_k=_TOP_K, alpha=_ALPHA)
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


def _build_summary(results: list[RepoResult]) -> dict[str, object]:
    """Build the JSON summary dict from the current (possibly partial) results list."""
    n = len(results)
    return {
        "tool": "coderankembed",
        "note": f"{_MODEL_NAME} (137M params), semantic-only dense search (alpha=1.0)",
        "repos": [asdict(result) for result in results],
        "avg_ndcg10": round(sum(r.ndcg10 for r in results) / n, 4) if n else 0.0,
        "avg_p50_ms": round(sum(r.p50_ms for r in results) / n, 1) if n else 0.0,
        "avg_index_ms": round(sum(r.index_ms for r in results) / n, 1) if n else 0.0,
    }


def _load_completed(out_path: Path) -> dict[str, RepoResult]:
    """Load repos already saved in a previous run, keyed by repo name."""
    if not out_path.exists():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {entry["repo"]: RepoResult(**entry) for entry in data.get("repos", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _bench(
    repo_tasks: dict[str, list[Task]],
    specs: dict[str, RepoSpec],
    model: _AsymmetricWrapper,
    out_path: Path | None,
    *,
    verbose: bool = False,
) -> list[RepoResult]:
    """Index and evaluate each repo, saving after every repo."""
    completed = _load_completed(out_path) if out_path else {}
    if completed:
        print(f"Resuming: {len(completed)} repo(s) already done, skipping.", file=sys.stderr)

    results: list[RepoResult] = list(completed.values())

    header = (
        f"{'Repo':<12} {'Language':<12} {'Chunks':>6} {'Index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
    )
    print(header, file=sys.stderr)
    print(f"{'-' * 12} {'-' * 12} {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}", file=sys.stderr)

    for repo in sorted(completed):
        r = completed[repo]
        print(
            f"{r.repo:<12} {r.language:<12} {r.chunks:>6}"
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
        index = _build_index(spec.benchmark_dir, model)
        index_ms = (time.perf_counter() - started) * 1000

        ndcg5, ndcg10, latencies, by_category = _evaluate(index, tasks, verbose=verbose)
        p50, p90 = np.percentile(latencies, [50, 90]).tolist()
        result = RepoResult(
            repo=repo,
            language=spec.language,
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
            f"{repo:<12} {spec.language:<12} {len(index.chunks):>6}"
            f" {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms {p90:>7.2f}ms",
            file=sys.stderr,
        )

        if out_path:
            save_results("coderankembed", _build_summary(results))

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CodeRankEmbed on the semble benchmark suite.")
    add_filter_args(parser, verbose=True)
    return parser.parse_args()


def main() -> None:
    """Run the CodeRankEmbed baseline benchmark."""
    args = _parse_args()
    is_full_run = not args.repo and not args.language

    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print(f"Loading {_MODEL_NAME}...", file=sys.stderr)
    started = time.perf_counter()
    raw_model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
    model = _AsymmetricWrapper(raw_model)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)
    print(file=sys.stderr)

    out_path = results_path("coderankembed") if is_full_run else None
    results = _bench(grouped_tasks(tasks), repo_specs, model, out_path, verbose=args.verbose)

    if not results:
        return

    print(file=sys.stderr)
    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    print(f"  avg ndcg@10={avg_ndcg10:.3f}  avg p50={avg_p50:.1f}ms  ({len(results)} repos)", file=sys.stderr)

    summary = _build_summary(results)
    print(json.dumps(summary, indent=2))

    if is_full_run:
        print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
