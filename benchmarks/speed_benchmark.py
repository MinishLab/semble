import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

from benchmarks.data import RepoSpec, Task, available_repo_specs, load_tasks, save_results
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME

# One representative repo per language (medium size, healthy NDCG on the main benchmark).
_REPOS: list[str] = [
    "nvm",  # bash
    "libuv",  # c
    "nlohmann-json",  # cpp
    "messagepack-csharp",  # csharp
    "phoenix",  # elixir
    "gin",  # go
    "aeson",  # haskell
    "gson",  # java
    "axios",  # javascript
    "ktor",  # kotlin
    "telescope.nvim",  # lua
    "monolog",  # php
    "flask",  # python
    "rack",  # ruby
    "axum",  # rust
    "http4s",  # scala
    "alamofire",  # swift
    "trpc",  # typescript
    "zls",  # zig
]

_TOP_K = 10
_COLGREP = "colgrep"
_RG = "rg"


@dataclass(frozen=True)
class ToolResult:
    """Speed result for one tool on one repo."""

    repo: str
    language: str
    tool: str
    index_ms: float | None  # None = no index (ripgrep)
    p50_ms: float


class _CREWrapper:
    """Wrap SentenceTransformer with asymmetric query/document prompts."""

    def __init__(self, model: SentenceTransformer, max_seq_length: int = 512) -> None:
        """Initialise wrapper and cap sequence length to avoid OOM on CPU."""
        self._model = model
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode with query prompt for single items, document prompt for batches."""
        if len(texts) == 1:
            return self._model.encode(texts, prompt_name="query", batch_size=1)  # type: ignore[return-value]
        return self._model.encode(texts, batch_size=1)  # type: ignore[return-value]


def _run_ripgrep(query: str, benchmark_dir: Path) -> list[str]:
    """Run ripgrep and return top-k file paths sorted by match count."""
    cmd = [
        _RG,
        "--count",
        "--no-heading",
        "--ignore-case",
        "--hidden",
        "--glob",
        "!.git",
        "--fixed-strings",
        query,
        str(benchmark_dir),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        return []
    entries: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        *path_parts, count_str = line.split(":")
        try:
            entries.append((":".join(path_parts), int(count_str)))
        except ValueError:
            continue
    entries.sort(key=lambda x: -x[1])
    return [path for path, _ in entries[:_TOP_K]]


def _run_colgrep(query: str, benchmark_dir: Path, *, code_only: bool = True) -> list[str]:
    """Run ColGREP and return top-k file paths from the JSON output."""
    cmd = [_COLGREP, "--force-cpu", "--json", "-k", str(_TOP_K), query, str(benchmark_dir)]
    if code_only:
        cmd.append("--code-only")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [item["unit"]["file"] for item in data if "unit" in item and "file" in item["unit"]]


def _bench_semble(spec: RepoSpec, tasks: list[Task], model: object) -> tuple[float, float]:
    """Index a repo with semble and measure query latency; return (index_ms, p50_ms)."""
    started = time.perf_counter()
    index = SembleIndex.from_path(spec.benchmark_dir, model=model)
    index_ms = (time.perf_counter() - started) * 1000
    latencies: list[float] = []
    for task in tasks:
        query_latencies: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            index.search(task.query, top_k=_TOP_K, mode="hybrid")
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))
    return index_ms, float(np.median(latencies))


def _bench_coderankembed(spec: RepoSpec, tasks: list[Task], model: _CREWrapper) -> tuple[float, float]:
    """Index a repo with CodeRankEmbed via semble and measure query latency; return (index_ms, p50_ms)."""
    started = time.perf_counter()
    index = SembleIndex.from_path(spec.benchmark_dir, model=model)
    index_ms = (time.perf_counter() - started) * 1000
    latencies: list[float] = []
    for task in tasks:
        query_latencies: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            index.search(task.query, top_k=_TOP_K, mode="semantic")
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))
    return index_ms, float(np.median(latencies))


def _bench_colgrep(spec: RepoSpec, tasks: list[Task]) -> tuple[float, float] | None:
    """Index a repo with ColGREP and measure query latency; return (index_ms, p50_ms) or None if unsupported."""
    subprocess.run([_COLGREP, "clear", str(spec.benchmark_dir)], capture_output=True, timeout=30)
    started = time.perf_counter()
    proc = subprocess.run(
        [_COLGREP, "init", "--force-cpu", "-y", str(spec.benchmark_dir)], capture_output=True, text=True, timeout=300
    )
    index_ms = (time.perf_counter() - started) * 1000
    if proc.returncode != 0:
        print(f"  WARNING: colgrep init failed: {proc.stderr.strip()}", file=sys.stderr)
    if "(0 files)" in proc.stdout or "(0 files)" in proc.stderr:
        print(f"  SKIP: colgrep indexed 0 files (unsupported language?)", file=sys.stderr)
        return None
    latencies: list[float] = []
    code_only = spec.language != "bash"
    for task in tasks:
        started = time.perf_counter()
        _run_colgrep(task.query, spec.benchmark_dir, code_only=code_only)
        latencies.append((time.perf_counter() - started) * 1000)
    return index_ms, float(np.median(latencies))


def _bench_ripgrep(spec: RepoSpec, tasks: list[Task]) -> tuple[float, float]:
    """Measure ripgrep query latency (no index step); return (0.0, p50_ms)."""
    latencies: list[float] = []
    for task in tasks:
        query_latencies: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            _run_ripgrep(task.query, spec.benchmark_dir)
            query_latencies.append((time.perf_counter() - started) * 1000)
        latencies.append(float(np.median(query_latencies)))
    return 0.0, float(np.median(latencies))


def _build_summary(results: list[ToolResult], tools: list[str]) -> dict[str, object]:
    """Aggregate per-repo results into per-tool average index time and query p50."""
    by_tool: dict[str, list[ToolResult]] = {tool: [r for r in results if r.tool == tool] for tool in tools}
    summary: dict[str, object] = {}
    for tool, tool_results in by_tool.items():
        idx_vals = [r.index_ms for r in tool_results if r.index_ms is not None]
        summary[tool] = {
            "avg_index_ms": round(sum(idx_vals) / len(idx_vals), 1) if idx_vals else None,
            "avg_p50_ms": round(sum(r.p50_ms for r in tool_results) / len(tool_results), 2),
        }
    return summary


def main() -> None:
    """Run cold-start index + query latency benchmark over a curated 1-per-language subset."""
    specs = available_repo_specs()
    all_tasks = load_tasks(repo_specs=specs)
    repo_tasks: dict[str, list[Task]] = {repo: [t for t in all_tasks if t.repo == repo] for repo in _REPOS}

    print("Loading semble model...", file=sys.stderr)
    started = time.perf_counter()
    semble_model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)
    print(f"  loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)

    print("Loading CodeRankEmbed...", file=sys.stderr)
    started = time.perf_counter()
    cre_model = _CREWrapper(SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True, device="cpu"))
    print(f"  loaded in {(time.perf_counter() - started) * 1000:.0f}ms", file=sys.stderr)
    print(file=sys.stderr)

    tools = ["semble", "coderankembed", "colgrep", "ripgrep"]

    print(f"{'Repo':<22} {'Language':<14} {'Tool':<16} {'Index':>10} {'p50':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 14} {'-' * 16} {'-' * 10} {'-' * 8}", file=sys.stderr)

    all_results: list[ToolResult] = []

    for repo in _REPOS:
        spec = specs[repo]
        tasks = repo_tasks[repo]

        index_ms, p50_ms = _bench_semble(spec, tasks, semble_model)
        all_results.append(
            ToolResult(repo=repo, language=spec.language, tool="semble", index_ms=index_ms, p50_ms=p50_ms)
        )
        print(f"{repo:<22} {spec.language:<14} {'semble':<16} {index_ms:>8.0f}ms {p50_ms:>7.2f}ms", file=sys.stderr)

        index_ms, p50_ms = _bench_coderankembed(spec, tasks, cre_model)
        all_results.append(
            ToolResult(repo=repo, language=spec.language, tool="coderankembed", index_ms=index_ms, p50_ms=p50_ms)
        )
        print(f"{'':22} {spec.language:<14} {'coderankembed':<16} {index_ms:>8.0f}ms {p50_ms:>7.2f}ms", file=sys.stderr)

        colgrep_result = _bench_colgrep(spec, tasks)
        if colgrep_result is not None:
            index_ms, p50_ms = colgrep_result
            all_results.append(
                ToolResult(repo=repo, language=spec.language, tool="colgrep", index_ms=index_ms, p50_ms=p50_ms)
            )
            print(f"{'':22} {spec.language:<14} {'colgrep':<16} {index_ms:>8.0f}ms {p50_ms:>7.2f}ms", file=sys.stderr)
        else:
            print(f"{'':22} {spec.language:<14} {'colgrep':<16} {'N/A (unsupported)':>18}", file=sys.stderr)

        _, p50_ms = _bench_ripgrep(spec, tasks)
        all_results.append(ToolResult(repo=repo, language=spec.language, tool="ripgrep", index_ms=None, p50_ms=p50_ms))
        print(f"{'':22} {spec.language:<14} {'ripgrep':<16} {'N/A':>10} {p50_ms:>7.2f}ms", file=sys.stderr)

    summary = _build_summary(all_results, tools)

    print(file=sys.stderr)
    print("Summary (averages across 19 repos):", file=sys.stderr)
    for tool, stats in summary.items():
        assert isinstance(stats, dict)
        idx_str = f"{stats['avg_index_ms']:.0f}ms" if stats["avg_index_ms"] is not None else "N/A"
        print(f"  {tool:<16}  avg index={idx_str:<10}  avg p50={stats['avg_p50_ms']:.2f}ms", file=sys.stderr)

    payload = {
        "repos": _REPOS,
        "summary": summary,
        "results": [
            {
                "repo": r.repo,
                "language": r.language,
                "tool": r.tool,
                "index_ms": round(r.index_ms, 1) if r.index_ms is not None else None,
                "p50_ms": round(r.p50_ms, 2),
            }
            for r in all_results
        ],
    }
    out_path = save_results("speed", payload)
    print(f"\nResults saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
