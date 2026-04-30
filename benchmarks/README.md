# Benchmarks

Quality and speed benchmarks for `semble`.

- [Main results](#main-results)
- [Token efficiency](#token-efficiency)
- [By language](#by-language)
- [Ablations](#ablations)
- [Dataset](#dataset)
- [Methods](#methods)
- [Excluded methods](#excluded-methods)
- [Running the benchmarks](#running-the-benchmarks)

## Main results

Quality and speed across all methods.

| Method | NDCG@10 | Index | Query p50 |
|---|---:|---:|---:|
| CodeRankEmbed Hybrid | 0.862 | 57 s | 16 ms |
| **semble** | **0.854** | **263 ms** | **1.5 ms** |
| CodeRankEmbed | 0.765 | 57 s | 16 ms |
| ColGREP | 0.693 | 5.8 s | 124 ms |
| BM25 | 0.673 | 263 ms | 0.02 ms |
| grepai | 0.561 | 35 s | 48 ms |
| probe | 0.387 | — | 207 ms |
| ripgrep | 0.126 | — | 12 ms |

| ![Speed vs quality (cold)](../assets/images/speed_vs_ndcg_cold.png) | ![Speed vs quality (warm)](../assets/images/speed_vs_ndcg_warm.png) |
|:--:|:--:|
| *Time to first result (index + query) vs NDCG@10* | *Query latency on a warm index vs NDCG@10* |

The 137M-param CodeRankEmbed Hybrid wins NDCG@10 by 0.008. semble wins index time by 218x and query latency by 11x.

NDCG@10 is averaged across all queries. Speed numbers use one repo per language, CPU only: cold-start index time and warm query p50 (median across 5 consecutive runs).

## Token efficiency

Coding agents (Claude Code, OpenCode, etc.) typically find code by running `grep` on keywords and reading the matched files. We model that workflow and compare it against semble's chunk retrieval across our full benchmark of 1251 queries.

![Token efficiency: recall vs. retrieved tokens](../assets/images/token_efficiency.png)

### Expected tokens per query

For each query: tokens consumed at first relevant hit, or 32k if the method never finds anything. Averaged across all 1251 queries.

| Method | Expected tokens | Savings |
|---|---:|---:|
| ripgrep + read file | 45,692 | baseline |
| **semble** | **566** | **98% fewer** |

### Recall at fixed token budgets

A relevant file is "covered" once any retrieved unit comes from it.

| Method | 500 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **semble** | **0.685** | **0.849** | **0.938** | **0.976** | **0.991** | **0.996** | **0.996** |
| ripgrep + read file | 0.001 | 0.008 | 0.037 | 0.088 | 0.212 | 0.379 | 0.583 |

<details>
<summary>Methodology</summary>

semble returns the top-50 ranked chunks. `ripgrep+read` splits the query into keywords (dropping stopwords and short words), runs `rg --fixed-strings --ignore-case` for each keyword, then reads matched files in full ranked by how many distinct keywords they contain. Both methods search the same set of file types and ignored directories. Tokens are counted with `cl100k_base` via `tiktoken`. A relevant file is "covered" once any retrieved unit overlaps its annotated span.

</details>

## By language

NDCG@10 per language, sorted by CodeRankEmbed Hybrid (CRE in the table). Best score per row is bolded.

| Language | semble | CRE Hybrid | CRE | ColGREP | grepai | probe | ripgrep |
|---|---:|---:|---:|---:|---:|---:|---:|
| scala | 0.909 | **0.922** | 0.845 | 0.765 | 0.330 | 0.392 | 0.180 |
| cpp | **0.915** | 0.913 | 0.846 | 0.626 | 0.731 | 0.375 | 0.126 |
| ruby | **0.909** | **0.909** | 0.769 | 0.708 | 0.643 | 0.382 | 0.230 |
| elixir | 0.894 | **0.905** | 0.869 | 0.808 | 0.669 | 0.412 | 0.134 |
| javascript | 0.917 | 0.903 | **0.920** | 0.823 | 0.675 | 0.588 | 0.176 |
| zig | **0.913** | 0.901 | 0.807 | 0.474 | 0.755 | 0.369 | 0.000 |
| csharp | 0.885 | **0.889** | 0.743 | 0.614 | 0.277 | 0.392 | 0.117 |
| go | **0.895** | 0.884 | 0.676 | 0.785 | 0.722 | 0.410 | 0.133 |
| python | 0.867 | **0.880** | 0.794 | 0.777 | 0.634 | 0.488 | 0.202 |
| php | 0.858 | **0.874** | 0.758 | 0.663 | 0.402 | 0.340 | 0.123 |
| swift | 0.860 | **0.873** | 0.721 | 0.710 | 0.429 | 0.280 | 0.160 |
| bash | 0.825 | 0.852 | **0.892** | 0.706 | 0.723 | 0.226 | 0.000 |
| lua | 0.823 | **0.847** | 0.803 | 0.798 | 0.699 | 0.336 | 0.000 |
| java | **0.849** | 0.841 | 0.706 | 0.641 | 0.386 | 0.536 | 0.198 |
| kotlin | 0.821 | **0.830** | 0.670 | 0.637 | 0.478 | 0.335 | 0.166 |
| rust | **0.856** | 0.827 | 0.627 | 0.662 | 0.519 | 0.242 | 0.162 |
| c | 0.741 | **0.806** | 0.706 | 0.676 | 0.555 | 0.384 | 0.000 |
| haskell | 0.765 | 0.771 | **0.776** | 0.683 | 0.483 | 0.313 | 0.000 |
| typescript | 0.706 | **0.708** | 0.545 | 0.430 | 0.394 | 0.354 | 0.128 |
| **overall** | **0.854** | **0.862** | **0.765** | **0.693** | **0.561** | **0.387** | **0.126** |

## Ablations

`raw` returns retrieval scores directly; `+ ranking` feeds them through semble's hybrid ranker.

| Retrieval | Raw | + ranking |
|---|---:|---:|
| BM25 | 0.675 | 0.834 |
| potion-code-16M | 0.650 | 0.821 |
| BM25 + potion-code-16M | — | **0.854** |

<details>
<summary>By query category</summary>

| Mode | Architecture | Semantic | Symbol |
|---|---:|---:|---:|
| BM25 raw | 0.628 | 0.676 | 0.719 |
| potion-code-16M raw | 0.626 | 0.666 | 0.629 |
| semble BM25 (+ ranking) | 0.770 | 0.819 | 0.957 |
| semble potion-code-16M (+ ranking) | 0.757 | 0.808 | 0.943 |
| **semble hybrid** | **0.802** | **0.846** | **0.958** |

</details>

## Dataset

~1,250 queries over 63 repositories in 19 languages, grouped into three categories:

| Category | Queries | What it tests |
|---|---:|---|
| semantic | 711 | Code that implements a specific behavior or concept |
| architecture | 343 | Design decisions, module boundaries, structural patterns |
| symbol | 204 | Named entity lookup (function, class, type, variable) |

<details>
<summary>Notes</summary>

**Languages**: three repos per language (nine for Python): bash, C, C++, C#, Elixir, Go, Haskell, Java, JavaScript, Kotlin, Lua, PHP, Python, Ruby, Rust, Scala, Swift, TypeScript, Zig. Repos are pinned by revision in `repos.json`.

**How the benchmark was built**: queries and ground-truth relevance labels are generated by Claude Sonnet 4.6. The same model is used as LLM-as-judge to verify label quality.

</details>

## Methods

- **[ripgrep](https://github.com/BurntSushi/ripgrep)**: fast regex search over files, included as a raw keyword-match baseline.
- **[probe](https://github.com/buger/probe)**: BM25 keyword ranking backed by tree-sitter parse trees. No persistent index; scans on the fly.
- **[ColGREP](https://github.com/lightonai/next-plaid/tree/main/colgrep)**: late-interaction code retrieval built on next-plaid with the [LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) model.
- **[grepai](https://github.com/nicholasgasior/grepai)**: semantic search using [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) (137M params) via a local Ollama daemon.
- **[CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)**: 137M-param transformer embedding model for code retrieval. *CodeRankEmbed Hybrid* fuses its dense scores with BM25.
- **[semble](https://github.com/your-repo/semble)**: this library. [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) static embeddings + BM25 + the semble reranking stack.

## Excluded methods

Two tools were considered but not included in the benchmark:

- **[codanna](https://codanna.io)**: symbol-level semantic search with fastembed. Excluded because it does not support Haskell, Bash, Zig, Scala, Elixir, or Ruby (6 of the 19 benchmark languages, covering 20 of 63 repos (~38% of tasks)).
- **[claude-context](https://github.com/zilliztech/claude-context)**: retrieval-augmented code search using OpenAI embeddings and a vector database. Excluded because it requires a paid OpenAI API key and a running vector-DB service.

## Running the benchmarks

Repos are pinned in `repos.json` and cloned into `~/.cache/semble-bench`:

```bash
uv run python -m benchmarks.sync_repos          # clone / update
uv run python -m benchmarks.sync_repos --check  # verify only
```

All tools run CPU-only. semble uses `minishlab/potion-code-16M`; CodeRankEmbed uses `nomic-ai/CodeRankEmbed` (137M params). The speed benchmark touches one repo per language with a cold-start index and 5 query runs per repo.

<details>
<summary>semble</summary>

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs write to `benchmarks/results/semble-hybrid-<sha12>.json`.

</details>

<details>
<summary>Speed benchmark</summary>

```bash
uv run python -m benchmarks.speed_benchmark
```

Writes to `benchmarks/results/speed-<sha12>.json`.

</details>

<details>
<summary>Ablations</summary>

```bash
uv run python -m benchmarks.baselines.ablations
uv run python -m benchmarks.baselines.ablations --mode bm25
uv run python -m benchmarks.baselines.ablations --mode semble-semantic
```

</details>

<details>
<summary>probe</summary>

Needs `probe` on `$PATH` (`npm install -g @buger/probe`).

```bash
uv run python -m benchmarks.baselines.probe
uv run python -m benchmarks.baselines.probe --repo fastapi --repo axios
```

</details>

<details>
<summary>grepai</summary>

Needs `grepai` on `$PATH` and Ollama running with `nomic-embed-text` pulled:

```bash
ollama pull nomic-embed-text
```

```bash
uv run python -m benchmarks.baselines.grepai
uv run python -m benchmarks.baselines.grepai --repo fastapi --repo axios
```

Large repos take several minutes to index. Use `--timeout <seconds>` (default 120) for repos with many files:

```bash
uv run python -m benchmarks.baselines.grepai --timeout 1800 --output results.json
```

The `--output` flag enables resume mode: already-completed repos are skipped on restart.

</details>

<details>
<summary>ripgrep</summary>

Needs `rg` on `$PATH` (`brew install ripgrep` / `apt install ripgrep`).

```bash
uv run python -m benchmarks.baselines.ripgrep
uv run python -m benchmarks.baselines.ripgrep --no-fixed-strings
```

</details>

<details>
<summary>ColGREP</summary>

Needs the `colgrep` binary on `$PATH`.

```bash
uv run python -m benchmarks.baselines.colgrep
uv run python -m benchmarks.baselines.colgrep --repo fastapi --repo axios
```

Runs with `--code-only` everywhere except bash repos (bash-it, bats-core, nvm), which use `--no-code-only` because ColGREP's code filter excludes `.sh`/`.bash` files.

</details>

<details>
<summary>CodeRankEmbed</summary>

Requires the `benchmark` extra (`uv sync --extra benchmark`).

```bash
uv run python -m benchmarks.baselines.coderankembed
uv run python -m benchmarks.baselines.coderankembed --mode semantic
```

</details>

<details>
<summary>Context-efficiency benchmark</summary>

Requires the `benchmark` extra (`uv sync --extra benchmark`) and `rg` on `$PATH`.

```bash
# Recall vs. token-budget across all queries; plots automatically.
uv run python -m benchmarks.token_efficiency recall
uv run python -m benchmarks.token_efficiency recall --repo fastapi

# Regenerate the plot from a saved recall payload.
uv run python -m benchmarks.token_efficiency plot
```

Writes `benchmarks/results/token-efficiency-<sha12>.json` and `assets/images/token_efficiency.png`.

</details>

<details>
<summary>Plots</summary>

```bash
uv run python -m benchmarks.plot
```

Writes `speed_vs_ndcg_cold.png` and `speed_vs_ndcg_warm.png` to `assets/images/`.

</details>
