import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k

_CK = "ck"
_TOP_K = 10
_LATENCY_RUNS = 3
_INDEX_TIMEOUT = 1800
_SEARCH_TIMEOUT = 60


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float
    index_ms: float


def _cleanup_index(benchmark_dir: Path) -> None:
    shutil.rmtree(benchmark_dir / ".ck", ignore_errors=True)
    (benchmark_dir / ".ckignore").unlink(missing_ok=True)


def _build_index(benchmark_dir: Path) -> tuple[bool, float]:
    """Build a ck hybrid (BM25 + embedding) index for a repo; return (success, elapsed_ms)."""
    _cleanup_index(benchmark_dir)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [_CK, "--index", "--quiet", str(benchmark_dir)],
            capture_output=True,
            text=True,
            timeout=_INDEX_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"  WARNING: ck --index timed out after {_INDEX_TIMEOUT}s", file=sys.stderr)
        return False, (time.perf_counter() - started) * 1000
    elapsed_ms = (time.perf_counter() - started) * 1000
    if proc.returncode != 0:
        print(f"  WARNING: ck --index failed: {proc.stderr.strip()[:300]}", file=sys.stderr)
        return False, elapsed_ms
    return True, elapsed_ms


def _run_search(query: str, benchmark_dir: Path, *, top_k: int) -> list[str]:
    """Return absolute file paths from ck's hybrid (regex + semantic) JSON output."""
    cmd = [_CK, "--hybrid", "--json", "--quiet", "--topk", str(top_k), query, str(benchmark_dir)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEARCH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    # ck exits 1 with "No matches found" on stderr (empty stdout) rather than an empty JSON array.
    # --json with --quiet is actually JSONL (one object per line), not a wrapped JSON array.
    if not proc.stdout.strip():
        return []
    items: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    seen: dict[str, None] = {}
    for item in items:
        rel = item.get("file", "")
        if rel:
            abs_path = str((benchmark_dir / rel).resolve())
            seen[abs_path] = None
    return list(seen)[:top_k]


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
            file_paths = _run_search(task.query, benchmark_dir, top_k=_TOP_K)
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


def _run_repo(spec: RepoSpec, tasks: list[Task], *, verbose: bool) -> RepoResult | None:
    """Index, evaluate, and clean up a single repo."""
    benchmark_dir = spec.benchmark_dir
    ok, index_ms = _build_index(benchmark_dir)
    if not ok:
        print(f"  SKIP: {spec.name} — ck indexing failed", file=sys.stderr)
        _cleanup_index(benchmark_dir)
        return None

    try:
        ndcg10, p50_ms = _evaluate_repo(tasks, benchmark_dir, verbose=verbose)
    finally:
        _cleanup_index(benchmark_dir)

    return RepoResult(repo=spec.name, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms, index_ms=index_ms)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ck on the semble benchmark suite.")
    add_filter_args(parser, verbose=True)
    return parser.parse_args()


def main() -> None:
    """Run the ck baseline benchmark."""
    args = _parse_args()
    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("ck (hybrid: regex + bge-small-en-v1.5 semantic, RRF fusion)", file=sys.stderr)
    print(f"{'Repo':<22} {'Language':<12} {'Index':>9} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        if args.verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)
        result = _run_repo(spec, repo_task_list, verbose=args.verbose)
        if result is None:
            continue
        results.append(result)
        print(
            f"{repo:<22} {spec.language:<12} {result.index_ms:>8.0f}ms {result.ndcg10:>8.3f} {result.p50_ms:>7.1f}ms",
            file=sys.stderr,
        )

    if not results:
        return

    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)
    avg_label = f"Average ({len(results)})"
    print(
        f"{avg_label:<22} {'':<12} {avg_index:>8.0f}ms {avg_ndcg10:>8.3f} {avg_p50:>7.1f}ms",
        file=sys.stderr,
    )

    summary = {
        "tool": "ck",
        "note": "hybrid regex + BAAI/bge-small-en-v1.5 (33M params) semantic search, RRF fusion",
        "repos": [
            {
                "repo": r.repo,
                "language": r.language,
                "ndcg10": round(r.ndcg10, 4),
                "p50_ms": round(r.p50_ms, 1),
                "index_ms": round(r.index_ms, 0),
            }
            for r in results
        ],
        "avg_ndcg10": round(avg_ndcg10, 4),
        "avg_p50_ms": round(avg_p50, 1),
        "avg_index_ms": round(avg_index, 0),
    }
    print(json.dumps(summary, indent=2))

    if not args.repo and not args.language:
        out = save_results("ck", summary)
        print(f"\nResults saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
