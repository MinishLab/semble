# Benchmarks

Reproducible local benchmarks for `semble` across 63 repositories in 19 languages (Dart excluded — colgrep does not support it).

## Results

### Main results

Quality is measured as mean NDCG@10 over all benchmark tasks. Speed numbers are averages across 19 repositories (one per language) with all tools forced to CPU for a fair comparison.

| Method | NDCG@10 | Index time | Query p50 |
|---|---|---|---|
| ripgrep | 0.123 | — | 12 ms |
| colgrep | 0.577 | 5.8 s | 124 ms |
| coderankembed semantic | 0.762 | 57 s | 16 ms |
| **semble** | **0.852** | **263 ms** | **1.5 ms** |
| coderankembed hybrid | 0.860 | 57 s | 16 ms |

### Ablations

Isolates the contribution of the retrieval source vs. the ranking stack.
`raw` = no ranking pipeline (plain BM25 or plain ANN); `+ ranking` = same retrieval fed through semble's hybrid ranker.

| Retrieval | Raw NDCG@10 | + semble ranking |
|---|---|---|
| BM25 | 0.675 | 0.834 |
| Semantic (potion-code-16M) | 0.650 | 0.821 |
| BM25 + Semantic (full hybrid) | — | **0.852** |

The ranking stack adds roughly +0.16–0.17 NDCG@10 regardless of the retrieval source.

## Setup

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

```bash
uv run python -m benchmarks.sync_repos
uv run python -m benchmarks.sync_repos --check
```

## Running benchmarks

### Main benchmark

The primary benchmark — run this when iterating on semble.

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs (no `--repo`/`--language` filters) automatically save results to
`benchmarks/results/semble-hybrid-<sha12>.json`.

### Speed benchmark

Measures cold-start index time and query latency across 19 repositories (one per language). All tools are forced to CPU.

```bash
uv run python -m benchmarks.speed_benchmark
```

Results are saved to `benchmarks/results/speed-<sha12>.json`.

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
