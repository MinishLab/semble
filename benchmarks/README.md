# Benchmarks

Quality and speed benchmarks for `semble` across 63 repositories in 19 languages, with 1,258 annotated queries. Dart is excluded — colgrep does not support it.

## Results

Quality is NDCG@10 averaged across all queries. Index time and query p50 are from the speed benchmark (one repo per language, cold start).

| Method | NDCG@10 | Index time | Query p50 |
|---|---|---|---|
| ripgrep | 0.123 | — | 12 ms |
| colgrep | 0.577 | 5.8 s | 124 ms |
| coderankembed semantic | 0.762 | 57 s | 16 ms |
| **semble** | **0.852** | **263 ms** | **1.5 ms** |
| coderankembed hybrid | 0.860 | 57 s | 16 ms |

semble reaches 0.852 NDCG@10, close to coderankembed hybrid (0.860, a 137M-param transformer), while indexing 218x faster (263 ms vs 57 s) and querying 11x faster (1.5 ms vs 16 ms).

### By query category

Queries fall into three categories: symbol (named entity lookup), semantic (behavior/concepts), and architecture (design and structure).

| Method | Architecture | Semantic | Symbol |
|---|---|---|---|
| coderankembed hybrid | 0.811 | 0.863 | 0.941 |
| **semble** | 0.802 | 0.846 | **0.958** |
| coderankembed semantic | 0.690 | 0.777 | 0.845 |

semble leads on symbol queries (0.958 vs 0.941) where BM25 is strong. Architecture queries are the hardest category for all methods; coderankembed hybrid holds a small lead there (0.811 vs 0.802).

## Ablations

`raw` returns retrieval scores directly; `+ ranking` feeds them through semble's hybrid ranker.

| Retrieval | Raw NDCG@10 | + semble ranking | delta |
|---|---|---|---|
| BM25 | 0.675 | 0.834 | +0.159 |
| Semantic (potion-code-16M) | 0.650 | 0.821 | +0.171 |
| BM25 + Semantic (full hybrid) | — | **0.852** | — |

The ranking stack adds roughly +0.16 NDCG@10 for both retrieval methods. Combining BM25 and semantic retrieval adds another ~0.02 on top of that.

<details>
<summary>By query category</summary>

| Mode | Architecture | Semantic | Symbol |
|---|---|---|---|
| BM25 raw | 0.628 | 0.676 | 0.719 |
| Semantic raw | 0.626 | 0.666 | 0.629 |
| semble BM25 (+ ranking) | 0.770 | 0.819 | 0.957 |
| semble semantic (+ ranking) | 0.757 | 0.808 | 0.943 |
| **semble hybrid** | **0.802** | **0.846** | **0.958** |

</details>

## Running the benchmarks

### Setup

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

```bash
uv run python -m benchmarks.sync_repos          # clone / update
uv run python -m benchmarks.sync_repos --check  # verify only
```

Benchmark configuration:
- All tools run CPU-only (no GPU)
- semble model: `minishlab/potion-code-16M`
- coderankembed model: `nomic-ai/CodeRankEmbed` (137M params)
- Speed benchmark: 19 repos (one per language), cold-start index, 5 query runs per repo

### semble

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs save results to `benchmarks/results/semble-hybrid-<sha12>.json`.

### Speed benchmark

```bash
uv run python -m benchmarks.speed_benchmark
```

Results are saved to `benchmarks/results/speed-<sha12>.json`.

### Ablations

```bash
uv run python -m benchmarks.baselines.ablations
uv run python -m benchmarks.baselines.ablations --mode bm25
uv run python -m benchmarks.baselines.ablations --mode semble-semantic
```

### ripgrep

Requires `rg` on `$PATH` (`brew install ripgrep` / `apt install ripgrep`).

```bash
uv run python -m benchmarks.baselines.ripgrep
uv run python -m benchmarks.baselines.ripgrep --no-fixed-strings
```

### ColGREP

Requires the `colgrep` binary on `$PATH`.

```bash
uv run python -m benchmarks.baselines.colgrep --init   # build indexes once
uv run python -m benchmarks.baselines.colgrep
```

### CodeRankEmbed

Requires the `benchmark` extra (`uv sync --extra benchmark`).

```bash
uv run python -m benchmarks.baselines.coderankembed
uv run python -m benchmarks.baselines.coderankembed --mode semantic
```
