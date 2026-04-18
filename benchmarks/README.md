# Benchmarks

Reproducible local benchmarks for `semble`.

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

## Setup

```bash
uv run python -m benchmarks.sync_repos
uv run python -m benchmarks.sync_repos --check
```

## Run

### semble (hybrid — main result)

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs (no `--repo`/`--language` filters) automatically save results to
`benchmarks/results/<sha>.json`.

### Ablations (BM25-only and semantic-only)

Isolates the contribution of each semble component using the same default model.

```bash
uv run python -m benchmarks.bench_ablations
uv run python -m benchmarks.bench_ablations --mode bm25
uv run python -m benchmarks.bench_ablations --mode semantic
```

## Comparisons

These scripts benchmark external tools against semble using the same task set
and NDCG@10 metric.  All print a machine-readable JSON summary to stdout.

### ripgrep (keyword baseline)

Requires `rg` on `$PATH` (`brew install ripgrep` / `apt install ripgrep`).
Files are ranked by match-count descending.

```bash
uv run python -m benchmarks.bench_ripgrep
uv run python -m benchmarks.bench_ripgrep --repo requests --verbose
uv run python -m benchmarks.bench_ripgrep --no-fixed-strings   # regex mode
```

### ColGREP (AST-aware search)

Requires the `colgrep` binary on `$PATH`.

```bash
uv run python -m benchmarks.bench_colgrep --init   # build indexes once
uv run python -m benchmarks.bench_colgrep
uv run python -m benchmarks.bench_colgrep --repo requests --verbose
```

### CodeRankEmbed (transformer model)

Runs `nomic-ai/CodeRankEmbed` (137M params) in semantic-only and hybrid modes.
Requires the `benchmark` extra:

```bash
uv sync --extra benchmark
uv run python -m benchmarks.bench_coderankembed
uv run python -m benchmarks.bench_coderankembed --mode semantic
uv run python -m benchmarks.bench_coderankembed --repo fastapi --verbose
```
