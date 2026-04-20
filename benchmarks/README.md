# Benchmarks

Reproducible benchmarks for `semble` measuring search quality (NDCG@10) and speed (index time, query latency) across **63 repositories in 19 languages**. Dart is excluded from all benchmarks — colgrep does not index Dart files.

## Table of Contents

- [Results](#results)
  - [Setup](#setup)
  - [Main results](#main-results)
  - [Results by query category](#results-by-query-category)
  - [Key findings](#key-findings)
- [Ablations](#ablations)
- [Reproducing the results](#reproducing-the-results)

## Results

### Setup

| | |
|---|---|
| **Quality benchmark** | 63 repositories, 19 languages, ~900 annotated queries |
| **Speed benchmark** | 19 repositories, 1 per language (cold-start index, 5 query runs each) |
| **Hardware** | CPU-only for all tools (no GPU) |
| **semble model** | `minishlab/potion-code-16M` |
| **coderankembed model** | `nomic-ai/CodeRankEmbed` (137M params) |

### Main results

Quality is NDCG@10 averaged across all benchmark queries. Index time and query p50 are averages from the speed benchmark.

| Method | NDCG@10 | Index time | Query p50 |
|---|---|---|---|
| ripgrep | 0.123 | — | 12 ms |
| colgrep | 0.577 | 5.8 s | 124 ms |
| coderankembed semantic | 0.762 | 57 s | 16 ms |
| **semble** | **0.852** | **263 ms** | **1.5 ms** |
| coderankembed hybrid | 0.860 | 57 s | 16 ms |

### Results by query category

Queries are grouped into three categories. Symbol queries look up named entities (functions, classes, variables); semantic queries ask about behavior or concepts; architecture queries ask about design and structure.

| Method | Architecture | Semantic | Symbol |
|---|---|---|---|
| coderankembed hybrid | **0.811** | **0.863** | 0.941 |
| **semble** | 0.802 | 0.846 | **0.958** |
| coderankembed semantic | 0.690 | 0.777 | 0.845 |

### Key findings

- **semble matches coderankembed quality at a fraction of the cost.** NDCG@10 of 0.852 vs 0.860 for coderankembed hybrid — a 137M-parameter transformer model — while indexing **218× faster** (263 ms vs 57 s) and querying **11× faster** (1.5 ms vs 16 ms).
- **semble leads on symbol queries (0.958).** BM25 is strong at exact name matching, and the ranking stack amplifies this further. coderankembed hybrid scores 0.941.
- **Architecture queries are the hardest for all methods.** semble scores 0.802; coderankembed hybrid 0.811. This gap suggests that pure transformer retrieval still has an edge on highly conceptual queries.
- **colgrep is slow and limited.** At 5.8 s to index and 124 ms per query, it is both slower and less accurate than semble.

## Ablations

The ablation study isolates the contribution of the retrieval source versus the ranking stack. `raw` means the retrieval scores are returned directly (no reranking); `+ ranking` feeds those results through semble's hybrid ranker.

| Retrieval | Raw NDCG@10 | + semble ranking | Δ |
|---|---|---|---|
| BM25 | 0.675 | 0.834 | +0.159 |
| Semantic (potion-code-16M) | 0.650 | 0.821 | +0.171 |
| BM25 + Semantic (full hybrid) | — | **0.852** | — |

The ranking stack contributes roughly **+0.16 NDCG@10 regardless of the retrieval source**, and combining both retrieval methods adds a further ~0.018 on top. The full hybrid (0.852) outperforms coderankembed semantic (0.762) despite using a 9× smaller model.

<details>
<summary>Ablations by query category</summary>

| Mode | Architecture | Semantic | Symbol |
|---|---|---|---|
| BM25 raw | 0.628 | 0.676 | 0.719 |
| Semantic raw | 0.626 | 0.666 | 0.629 |
| semble BM25 (+ ranking) | 0.770 | 0.819 | 0.957 |
| semble semantic (+ ranking) | 0.757 | 0.808 | 0.943 |
| semble hybrid | **0.802** | **0.846** | **0.958** |

The ranking stack helps most on symbol queries for BM25 (+0.238) and on semantic queries for the semantic retriever. Architecture queries see the smallest absolute gain but still benefit consistently.

</details>

## Reproducing the results

### Sync repositories

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

```bash
uv run python -m benchmarks.sync_repos          # clone / update
uv run python -m benchmarks.sync_repos --check  # verify only
```

### semble (main benchmark)

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs (no `--repo`/`--language` filters) save results to
`benchmarks/results/semble-hybrid-<sha12>.json`.

### Speed benchmark

Measures cold-start index time and query latency across 19 repositories (one per language) with all tools forced to CPU.

```bash
uv run python -m benchmarks.speed_benchmark
```

Results are saved to `benchmarks/results/speed-<sha12>.json`.

### Ablations

Isolates the contribution of each semble component.

```bash
uv run python -m benchmarks.baselines.ablations
uv run python -m benchmarks.baselines.ablations --mode bm25
uv run python -m benchmarks.baselines.ablations --mode semble-semantic
```

### ripgrep

Requires `rg` on `$PATH` (`brew install ripgrep` / `apt install ripgrep`).

```bash
uv run python -m benchmarks.baselines.ripgrep
uv run python -m benchmarks.baselines.ripgrep --no-fixed-strings   # regex mode
```

### ColGREP

Requires the `colgrep` binary on `$PATH`.

```bash
uv run python -m benchmarks.baselines.colgrep --init   # build indexes once
uv run python -m benchmarks.baselines.colgrep
```

### CodeRankEmbed

Runs `nomic-ai/CodeRankEmbed` (137M params) in semantic and hybrid modes. Requires the `benchmark` extra:

```bash
uv sync --extra benchmark
uv run python -m benchmarks.baselines.coderankembed
uv run python -m benchmarks.baselines.coderankembed --mode semantic
```
