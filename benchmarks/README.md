# Benchmarks

Reproducible local benchmarks for `semble`.

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

## Setup

```bash
uv run python -m benchmarks.sync_repos
uv run python -m benchmarks.sync_repos --check
```

## Main benchmark

The primary benchmark — run this when iterating on semble.

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs (no `--repo`/`--language` filters) automatically save results to
`benchmarks/results/semble-hybrid-<sha12>.json`.

## Baselines

One-shot benchmarks in `benchmarks/baselines/`. Results are already saved in
`benchmarks/results/` — only re-run if you need to reproduce them.

### Ablations

Isolates the contribution of each semble component (retrieval source vs. ranking stack).

```bash
uv run python -m benchmarks.baselines.ablations
uv run python -m benchmarks.baselines.ablations --mode bm25
uv run python -m benchmarks.baselines.ablations --mode semble-semantic
```

### ripgrep (keyword baseline)

Requires `rg` on `$PATH` (`brew install ripgrep` / `apt install ripgrep`).

```bash
uv run python -m benchmarks.baselines.ripgrep
uv run python -m benchmarks.baselines.ripgrep --no-fixed-strings   # regex mode
```

### ColGREP (AST-aware search)

Requires the `colgrep` binary on `$PATH`.

```bash
uv run python -m benchmarks.baselines.colgrep --init   # build indexes once
uv run python -m benchmarks.baselines.colgrep
```

### CodeRankEmbed (transformer model)

Runs `nomic-ai/CodeRankEmbed` (137M params) in semantic and hybrid modes.
Requires the `benchmark` extra:

```bash
uv sync --extra benchmark
uv run python -m benchmarks.baselines.coderankembed
uv run python -m benchmarks.baselines.coderankembed --mode semantic
```
