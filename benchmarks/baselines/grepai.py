import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.data import (
    RepoSpec,
    Task,
    apply_task_filters,
    available_repo_specs,
    grouped_tasks,
    load_tasks,
    save_results,
)
from benchmarks.metrics import file_rank, ndcg_at_k

_GREPAI = "grepai"
_TOP_K = 10
_LATENCY_RUNS = 1  # Ollama embedding calls are slow; single run is sufficient
_INDEX_TIMEOUT = 300
_SEARCH_TIMEOUT = 60
_WATCH_READY_TIMEOUT = 120  # overridden by --timeout


@dataclass(frozen=True)
class RepoResult:
    """Per-repo benchmark result."""

    repo: str
    language: str
    ndcg10: float
    p50_ms: float
    index_ms: float


def _cleanup_index(benchmark_dir: Path) -> None:
    d = benchmark_dir / ".grepai"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _build_index(benchmark_dir: Path, *, watch_ready_timeout: int = _WATCH_READY_TIMEOUT) -> tuple[bool, float]:
    """Init and index a repo with grepai; return (success, elapsed_ms)."""
    _cleanup_index(benchmark_dir)

    init_proc = subprocess.run(
        [_GREPAI, "init", "--provider", "ollama", "--yes"],
        capture_output=True,
        text=True,
        cwd=benchmark_dir,
        timeout=30,
    )
    if init_proc.returncode != 0:
        print(f"  WARNING: grepai init failed: {init_proc.stderr.strip()}", file=sys.stderr)
        return False, 0.0

    # grepai writes progress bars with \r (no \n), so readline() blocks forever.
    # Write stdout to a temp file and poll for the sentinel string instead.
    # "Initial scan complete" appears after file scanning but BEFORE embeddings
    # finish. Wait for 3 s of output silence after that sentinel to ensure all
    # embeddings have been flushed to disk before killing watch.
    started = time.perf_counter()
    watch_proc: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryFile() as log_f:
        watch_proc = subprocess.Popen(
            [_GREPAI, "watch"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=benchmark_dir,
            start_new_session=True,  # Own process group so killpg doesn't hit us
        )
        try:
            deadline = time.perf_counter() + watch_ready_timeout
            scan_complete = False
            last_size = 0
            idle_since: float | None = None
            _IDLE_SETTLE = 3.0  # seconds of silence after scan_complete → embeddings done

            while time.perf_counter() < deadline:
                time.sleep(0.3)
                log_f.seek(0)
                content = log_f.read()
                if not scan_complete and b"Initial scan complete" in content:
                    scan_complete = True
                    idle_since = time.perf_counter()
                if scan_complete:
                    if len(content) != last_size:
                        idle_since = time.perf_counter()
                        last_size = len(content)
                    elif idle_since is not None and (time.perf_counter() - idle_since) >= _IDLE_SETTLE:
                        return True, (time.perf_counter() - started) * 1000
                if watch_proc.poll() is not None:
                    if scan_complete:
                        return True, (time.perf_counter() - started) * 1000
                    break
            print(
                f"  WARNING: grepai watch timed out after {watch_ready_timeout}s",
                file=sys.stderr,
            )
            return False, (time.perf_counter() - started) * 1000
        finally:
            try:
                os.killpg(os.getpgid(watch_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            watch_proc.wait(timeout=5)


def _run_search(query: str, benchmark_dir: Path, *, top_k: int) -> list[str]:
    """Return absolute file paths from grepai JSON search output."""
    cmd = [_GREPAI, "search", query, "--json", "-n", str(top_k)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT,
            cwd=benchmark_dir,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    # grepai returns relative paths; make them absolute.
    seen: dict[str, None] = {}
    for item in items:
        rel = item.get("file_path", "")
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


def _run_repo(
    spec: RepoSpec,
    tasks: list[Task],
    *,
    verbose: bool,
    watch_ready_timeout: int = _WATCH_READY_TIMEOUT,
) -> RepoResult | None:
    """Index, evaluate, and clean up a single repo."""
    benchmark_dir = spec.benchmark_dir
    ok, index_ms = _build_index(benchmark_dir, watch_ready_timeout=watch_ready_timeout)
    if not ok:
        print(f"  SKIP: {spec.name} — grepai indexing failed", file=sys.stderr)
        return None

    try:
        ndcg10, p50_ms = _evaluate_repo(tasks, benchmark_dir, verbose=verbose)
    finally:
        _cleanup_index(benchmark_dir)

    return RepoResult(repo=spec.name, language=spec.language, ndcg10=ndcg10, p50_ms=p50_ms, index_ms=index_ms)


def _build_summary(results: list[RepoResult]) -> dict:
    avg_ndcg10 = sum(r.ndcg10 for r in results) / len(results)
    avg_p50 = sum(r.p50_ms for r in results) / len(results)
    avg_index = sum(r.index_ms for r in results) / len(results)
    return {
        "tool": "grepai",
        "note": "nomic-embed-text via Ollama (137 M params, ~8× larger than semble's potion-code-16M)",
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


def _write_results(results: list[RepoResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_summary(results), indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark grepai on the semble benchmark suite.")
    parser.add_argument("--repo", action="append", default=[], help="Limit to one or more repo names.")
    parser.add_argument("--language", action="append", default=[], help="Limit to one or more languages.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results.")
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="JSON file to write results to; if it already exists, repos already present are skipped (resume mode).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_WATCH_READY_TIMEOUT,
        metavar="SECONDS",
        help=f"Seconds to wait for embeddings to finish (default: {_WATCH_READY_TIMEOUT}). "
        "Increase for large repos (e.g. --timeout 1800).",
    )
    return parser.parse_args()


def _load_existing(output_path: Path | None) -> dict[str, dict]:
    """Load already-completed repos from a prior run's output file."""
    if output_path is None or not output_path.exists():
        return {}
    try:
        existing_data = json.loads(output_path.read_text())
        existing = {r["repo"]: r for r in existing_data.get("repos", [])}
        print(f"Resuming: {len(existing)} repos already done, will skip them.", file=sys.stderr)
        return existing
    except (json.JSONDecodeError, KeyError):
        return {}


def main() -> None:
    """Run the grepai baseline benchmark."""
    args = _parse_args()
    repo_specs = available_repo_specs()
    tasks = apply_task_filters(
        load_tasks(repo_specs=repo_specs), repos=args.repo or None, languages=args.language or None
    )

    output_path = Path(args.output) if args.output else None
    existing = _load_existing(output_path)

    print("grepai (ollama/nomic-embed-text, 137M params)", file=sys.stderr)
    print(f"{'Repo':<22} {'Language':<12} {'Index':>9} {'NDCG@10':>8} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 9} {'-' * 8} {'-' * 8}", file=sys.stderr)

    results: list[RepoResult] = []
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        if repo in existing:
            r = existing[repo]
            results.append(
                RepoResult(
                    repo=r["repo"],
                    language=r["language"],
                    ndcg10=r["ndcg10"],
                    p50_ms=r["p50_ms"],
                    index_ms=r["index_ms"],
                )
            )
            print(f"{repo:<22} {'(skipped — already done)':<12}", file=sys.stderr)
            continue
        if args.verbose:
            print(f"\n--- {repo} ---", file=sys.stderr)
        result = _run_repo(spec, repo_task_list, verbose=args.verbose, watch_ready_timeout=args.timeout)
        if result is None:
            continue
        results.append(result)
        print(
            f"{repo:<22} {spec.language:<12} {result.index_ms:>8.0f}ms {result.ndcg10:>8.3f} {result.p50_ms:>7.1f}ms",
            file=sys.stderr,
        )

        if output_path:
            _write_results(results, output_path)

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

    summary = _build_summary(results)
    if output_path:
        _write_results(results, output_path)
    else:
        save_results("grepai", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
