import argparse
import json
import sys
import time
from dataclasses import asdict

import numpy as np
from model2vec import StaticModel

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
    summarize_modes,
)
from benchmarks.run_benchmark import RepoResult, evaluate
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.types import SearchMode

_MODES = ["bm25", "semantic", "semble-bm25", "semble-semantic"]

# Maps mode name -> (search_mode, alpha) for index.search()
# alpha=None  → raw mode, no ranking pipeline
# alpha=0.0   → hybrid pipeline, BM25-only input
# alpha=1.0   → hybrid pipeline, semantic-only input
_MODE_PARAMS: dict[str, tuple[SearchMode, float | None]] = {
    "bm25": (SearchMode.BM25, None),
    "semantic": (SearchMode.SEMANTIC, None),
    "semble-bm25": (SearchMode.HYBRID, 0.0),
    "semble-semantic": (SearchMode.HYBRID, 1.0),
}


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
        f"{'Repo':<12} {'Language':<12} {'Mode':<16} {'Chunks':>6} {'Tokens':>8}"
        f" {'Index':>9} {'NDCG@5':>8} {'NDCG@10':>8} {'p50':>8} {'p90':>8}"
    )
    print(header, file=sys.stderr)
    print(
        f"{'-' * 12} {'-' * 12} {'-' * 16} {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}",
        file=sys.stderr,
    )

    for repo, tasks in sorted(repo_tasks.items()):
        spec = specs[repo]
        if verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)

        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        index_ms = (time.perf_counter() - started) * 1000

        for mode in modes:
            search_mode, alpha = _MODE_PARAMS[mode]
            ndcg5, ndcg10, latencies, by_category, tokens = evaluate(
                index, tasks, mode=search_mode, alpha=alpha, verbose=verbose
            )
            p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99]).tolist()
            result = RepoResult(
                repo=repo,
                language=spec.language,
                mode=mode,
                chunks=len(index.chunks),
                tokens=tokens,
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
                f"{repo:<12} {spec.language:<12} {mode:<16} {len(index.chunks):>6} {tokens:>8}"
                f" {index_ms:>8.0f}ms {ndcg5:>8.3f} {ndcg10:>8.3f} {p50:>7.2f}ms {p90:>7.2f}ms",
                file=sys.stderr,
            )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="semble ablation benchmarks.")
    add_filter_args(parser, verbose=True)
    parser.add_argument(
        "--mode", action="append", default=[], choices=_MODES, help="Mode(s) to evaluate (default: all)."
    )
    return parser.parse_args()


def main() -> None:
    """Run the semble ablation benchmarks."""
    args = _parse_args()
    modes = args.mode or _MODES

    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("Loading model...", file=sys.stderr)
    started = time.perf_counter()
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
    print(f"Loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)
    print(file=sys.stderr)

    results = _bench(grouped_tasks(tasks), repo_specs, model, modes, verbose=args.verbose)

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
            f"  {mode:<16}  avg ndcg@10={avg_ndcg10:.3f}  avg p50={avg_p50:.1f}ms  ({len(mode_results)} repos)",
            file=sys.stderr,
        )

    summary = {
        "tool": "semble-ablations",
        "model": _DEFAULT_MODEL_NAME,
        "by_mode": summarize_modes(results, modes),
        "repos": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results("semble-ablations", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
