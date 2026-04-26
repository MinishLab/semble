
<h2 align="center">
  <img width="30%" alt="semble logo" src="assets/images/semble_logo.png"><br/>
  Fast and Accurate Code Search for Agents
</h2>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/semble/"><img src="https://img.shields.io/pypi/v/semble?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://app.codecov.io/gh/MinishLabl/semble">
      <img src="https://codecov.io/gh/MinishLabl/semble/graph/badge.svg?token=SZKRFKPPCG" alt="Codecov">
    </a>
    <a href="https://github.com/MinishLab/semble/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>

[Quickstart](#quickstart) •
[Main Features](#main-features) •
[MCP Server](#mcp-server) •
[How it works](#how-it-works) •
[Benchmarks](#benchmarks)

</div>

Semble is a code search library built for agents. It returns the exact code snippets they need instantly, cutting both token usage and waiting time on every step. Indexing and searching a full codebase end-to-end takes under a second, with ~200x faster indexing and ~10x faster queries than a code-specialized transformer, at 99% of its retrieval quality (see [benchmarks](#benchmarks)). Everything runs on CPU with no API keys, GPU, or external services. Run it as an [MCP server](#mcp-server) and any agent (Claude Code, Cursor, Codex, OpenCode, etc.) gets instant access to any repo, cloned and indexed on demand.

## Quickstart

```bash
pip install semble  # Install with pip
uv add semble       # Install with uv
```

```python
from semble import SembleIndex

# Index a local directory
index = SembleIndex.from_path("./my-project")

# Index a remote git repository
index = SembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search the index with a natural-language or code query
results = index.search("save model to disk", top_k=3)

# Find code similar to a specific result
related = index.find_related(results[0], top_k=3)

# Each result exposes the matched chunk
result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."
```

## Main Features

- **Fast**: indexes a repo in ~250 ms and answers queries in ~1.5 ms, all on CPU.
- **Accurate**: NDCG@10 of 0.854 on our [benchmarks](#benchmarks), on par with code-specialized transformer models, at a fraction of the size and cost.
- **Local and remote**: pass a local path or a git URL.
- **MCP server**: drop-in tool for Claude Code, Cursor, Codex, OpenCode, and any other MCP-compatible agent.
- **Zero setup**: runs on CPU with no API keys, GPU, or external services required.

## MCP Server

Semble can run as an MCP server so agents can search any codebase directly. Repos are cloned and indexed on demand, and indexes are cached for the lifetime of the session.

### Setup

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

#### Claude Code
```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble
```

#### Codex
Add to `~/.codex/config.toml`:
```toml
[mcp_servers.semble]
command = "uvx"
args = ["--from", "semble[mcp]", "semble"]
```

#### OpenCode
Add to `~/.opencode/config.json`:
```json
{
  "mcp": {
    "semble": {
      "type": "local",
      "command": ["uvx", "--from", "semble[mcp]", "semble"]
    }
  }
}
```

#### Cursor
Add to `~/.cursor/mcp.json` (or `.cursor/mcp.json` in your project):
```json
{
  "mcpServers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

### Tools

| Tool | Description |
|------|-------------|
| `search` | Search a codebase with a natural-language or code query. Pass `repo` as a git URL or local path. |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |

## How it works

Semble splits each file into code-aware chunks using [Chonkie](https://github.com/chonkie-inc/chonkie), then scores every query against the chunks with two complementary retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec) embeddings using the code-specialized [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) model for semantic similarity, and [BM25](https://github.com/xhluca/bm25s) for lexical matches on identifiers and API names. The two score lists are fused with Reciprocal Rank Fusion (RRF).

After fusing, results are reranked with a set of code-aware signals:

<details>
<summary><b>Ranking signals</b></summary>

- **Adaptive weighting.** Symbol-like queries (`Foo::bar`, `_private`, `getUserById`) get more lexical weight, while natural-language queries stay balanced between semantic and lexical retrievers.
- **Definition boosts.** A chunk that defines the queried symbol (a `class`, `def`, `func`, etc.) is ranked above chunks that merely reference it.
- **Identifier stems.** Query tokens are stemmed and matched against identifier stems in a chunk, giving an additional weight to chunks that contain them. For example, querying `parse config` boosts chunks containing `parseConfig`, `ConfigParser`, or `config_parser`.
- **File coherence.** When multiple chunks from the same file match the query, the file is boosted so the top result reflects broad file-level relevance rather than a single out-of-context chunk.
- **Noise penalties.** Test files, `compat/`/`legacy/` shims, example code, and `.d.ts` declaration stubs are down-ranked so canonical implementations surface first.

</details>

Because the embedding model is static with no transformer forward pass at query time, all of this runs in milliseconds on CPU.

## Benchmarks

We benchmark quality and speed across all methods on ~1,250 queries over 63 repositories in 19 languages. The x-axis is total latency (index + first query); the y-axis is NDCG@10. Marker size reflects model parameter count.

![Speed vs quality](assets/images/speed_vs_ndcg_cold.png)

| Method | NDCG@10 | Index time | Query p50 |
|--------|--------:|-----------:|----------:|
| CodeRankEmbed Hybrid | 0.862 | 57 s | 16 ms |
| **semble** | **0.854** | **263 ms** | **1.5 ms** |
| CodeRankEmbed | 0.765 | 57 s | 16 ms |
| ColGREP | 0.693 | 5.8 s | 124 ms |
| BM25 | 0.673 | 263 ms | 0.02 ms |
| ripgrep | 0.126 | — | 12 ms |

Semble achieves 99% of the performance of the 137M-parameter [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) Hybrid, while indexing 218x faster and answering queries 11x faster. See [benchmarks](benchmarks/README.md) for per-language results, ablations, and methodology.

## License

MIT

## Citing

If you use Semble in your research, please cite the following:

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
