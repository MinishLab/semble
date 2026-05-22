
<h2 align="center">
  <img width="30%" alt="semble logo" src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/semble_logo.png"><br/>
  Fast and Accurate Code Search for Agents<br/>
  <sub>Uses ~98% fewer tokens than grep+read</sub>
</h2>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/semble/"><img src="https://img.shields.io/pypi/v/semble?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://app.codecov.io/gh/MinishLab/semble">
      <img src="https://codecov.io/gh/MinishLab/semble/graph/badge.svg?token=SZKRFKPPCG" alt="Codecov">
    </a>
    <a href="https://github.com/MinishLab/semble/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>

[Quickstart](#quickstart) •
[MCP Server](#mcp-server) •
[Bash / AGENTS.md](#bash-agentsmd) •
[CLI](#cli) •
[Benchmarks](#benchmarks)

</div>

Semble is a code search library built for agents. It returns the exact code snippets they need instantly, using ~98% fewer tokens than grep+read. Indexing and searching a full codebase end-to-end takes under a second, with ~200x faster indexing and ~10x faster queries than a code-specialized transformer, at 99% of its retrieval quality (see [benchmarks](#benchmarks)). Everything runs on CPU with no API keys, GPU, or external services. Run it as an [MCP server](#mcp-server) or call it from the shell via [AGENTS.md](#bash-agentsmd) and any agent (Claude Code, Cursor, Codex, OpenCode, etc.) gets instant access to any repo.

## Quickstart

Your agent queries Semble in natural language (e.g. `"How is authentication handled?"`) and gets back only the relevant code snippets, without grepping or reading full files. Set it up as an MCP server or via AGENTS.md:

### MCP (Claude Code)

Add Semble to Claude Code (requires [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble
```

Using another agent harness? See [MCP Server](#mcp-server) below for per-agent setup.

### Bash / AGENTS.md

Install Semble, then add the snippet below to your `AGENTS.md` or `CLAUDE.md`:

```bash
pip install semble       # Install with pip
uv tool install semble   # Or install with uv
```

<details>
<summary>AGENTS.md / CLAUDE.md snippet</summary>

```markdown
## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
​```

If you anticipate doing more than one search, use `semble index` to create an index.

​```bash
semble index ./my-project -o my_index
​```

You can then reuse this index later on:

​```bash
semble search "save_pretrained" --index my_index
​```

An index is not automatically updated, so if the code changes significantly, reindex. If you notice stale results while resolving searches to files, reindex.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

​```bash
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble search "authentication" ./my-project --content all
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

Like search, `find-related` also accepts an `--index` argument.

`path` defaults to the current directory when omitted; git URLs are accepted.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

### Workflow

1. Index the repo using `semble index -o cached_index`.
2. Start with `semble search` to find relevant chunks. Pass the index to achieve results faster.
3. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
4. Inspect full files only when the returned chunk does not give enough context.
5. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
6. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
```

</details>

Note that sub-agents cannot call MCP tools directly, see [Bash / AGENTS.md](#bash-agentsmd) and [sub-agent setup](#sub-agent-setup) below for details.

<details>
<summary>Updating Semble</summary>

```bash
pip install --upgrade semble   # with pip
uv tool upgrade semble         # with uv
uv cache clean semble          # for MCP users (restart your MCP client after)
```

</details>

## Main Features

- **Fast**: indexes an average repo in ~250 ms and answers queries in ~1.5 ms, all on CPU.
- **Accurate**: NDCG@10 of 0.854 on our [benchmarks](#benchmarks), on par with code-specialized transformer models, at a fraction of the size and cost.
- **Token-efficient**: returns only the relevant chunks, using [~98% fewer tokens than grep+read](#benchmarks).
- **Zero setup**: runs on CPU with no API keys, GPU, or external services required.
- **MCP server**: works with Claude Code, Cursor, Codex, OpenCode, VS Code, and any other MCP-compatible agent.
- **Local and remote**: pass a local path or a git URL.

## MCP Server

Semble can run as an MCP server so agents can search any codebase directly. Repos are cloned and indexed on demand, and indexes are cached for the lifetime of the session. Local paths are watched for file changes and re-indexed automatically.

### Setup

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

<details>
<summary>Claude Code</summary>

```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble
```

</details>

<details>
<summary>Cursor</summary>

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

</details>

<details>
<summary>Codex</summary>

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.semble]
command = "uvx"
args = ["--from", "semble[mcp]", "semble"]
```

</details>

<details>
<summary>OpenCode</summary>

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

</details>

<details>
<summary>VS Code</summary>

Add to `.vscode/mcp.json` in your project (or your user profile's `mcp.json`):

```json
{
  "servers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

</details>

<details>
<summary>GitHub Copilot CLI</summary>

Add to `~/.copilot/mcp-config.json`:

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

</details>

<details>
<summary>Windsurf</summary>

Add to `~/.codeium/windsurf/mcp_config.json`:

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

</details>

<details>
<summary>Gemini CLI</summary>

Add to `~/.gemini/settings.json`:

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

</details>

<details>
<summary>Kiro</summary>

Add to `~/.kiro/settings/mcp.json` (or `.kiro/settings/mcp.json` in your project):

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

</details>

<details>
<summary>Zed</summary>

Add to `~/.config/zed/settings.json` (or `.zed/settings.json` in your project):

```json
{
  "context_servers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

</details>


### Tools

| Tool | Description |
|------|-------------|
| `search` | Search a codebase with a natural-language or code query. Pass `repo` as a local directory path or an https:// git URL. |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |

By default the MCP server indexes only code files. To also index documentation, config, or everything, append `--content docs`, `--content config`, or `--content all` to the server command, or a combination, e.g. `--content code docs`. For example, in Claude Code: `claude mcp add semble -s user -- uvx --from "semble[mcp]" semble --content all`.


<a id="bash-agentsmd"></a>

## Bash / AGENTS.md

An alternative to MCP is to invoke Semble via Bash. Sub-agents cannot call MCP tools directly, so this is the only option for sub-agent support; it can also be used alongside MCP for the top-level agent.

To add Bash support, append the following to your `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or equivalent:

```markdown
## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
​```

If you anticipate doing more than one search, use `semble index` to create an index.

​```bash
semble index ./my-project -o my_index
​```

You can then reuse this index later on:

​```bash
semble search "save_pretrained" --index my_index
​```

An index is not automatically updated, so if the code changes significantly, reindex. If you notice stale results while resolving searches to files, reindex.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

​```bash
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble search "authentication" ./my-project --content all
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

Like search, `find-related` also accepts an `--index` argument.

`path` defaults to the current directory when omitted; git URLs are accepted.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

### Workflow

1. Index the repo using `semble index -o cached_index`.
2. Start with `semble search` to find relevant chunks. Pass the index to achieve results faster.
3. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
4. Inspect full files only when the returned chunk does not give enough context.
5. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
6. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
```

### Sub-agent setup

Claude Code, Gemini CLI, Cursor, OpenCode, GitHub Copilot CLI, and Kiro all support a dedicated semble search sub-agent. Run `semble init` once in your project root:

```bash
semble init                      # Claude Code  → .claude/agents/semble-search.md
semble init --agent gemini       # Gemini CLI   → .gemini/agents/semble-search.md
semble init --agent cursor       # Cursor       → .cursor/agents/semble-search.md
semble init --agent opencode     # OpenCode     → .opencode/agents/semble-search.md
semble init --agent copilot      # Copilot CLI  → .github/agents/semble-search.md
semble init --agent kiro         # Kiro         → .kiro/agents/semble-search.md
```

If semble is not on `$PATH`, prefix the command with `uvx --from "semble[mcp]"`.

## CLI

Semble also ships as a standalone CLI. This is useful in scripts or anywhere you want search results without an MCP session.

```bash
# Index a local repository
semble index ./my-project -o my-index

# Search a local repo
semble search "authentication flow" ./my-project
# Or with index (significantly faster)
# the index flag applies to all commands below.
semble search "authentication flow" --index my-index

# Search for a symbol or identifier
semble search "save_pretrained" ./my-project

# Search a remote repo (cloned on demand)
semble search "save model to disk" https://github.com/MinishLab/model2vec

# Limit results
semble search "save model to disk" ./my-project --top-k 10

# Search docs and prose (markdown, rst, etc.) instead of code
semble search "deployment guide" ./my-project --content docs

# Search config files (yaml, toml, terraform, etc.)
semble search "database host port" ./my-project --content config

# Search everything (code, docs, and config)
semble search "authentication" ./my-project --content all

# Find code similar to a known location
semble find-related src/auth.py 42 ./my-project
```

`--content` accepts `code` (default), `docs`, `config`, or `all`. `path` defaults to the current directory when omitted; git URLs are accepted. If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

<details>
<summary>Savings</summary>

`semble savings` shows how many tokens semble has saved across all your searches:

```bash
semble savings           # summary by period
semble savings --verbose # also show breakdown by call type
```

```
  Semble Token Savings
  ════════════════════════════════════════════════════════════════
  Period        Calls   Savings
  ────────────────────────────────────────────────────────────────
  Today         42      [███████████████░]  ~58.4k tokens (95%)
  Last 7 days   287     [██████████████░░]  ~312.4k tokens (90%)
  All time      1.4k    [██████████████░░]  ~1.2M tokens (89%)
```

Savings are calculated as follows: for each call, semble records the total character count of the unique files containing returned chunks and the character count of the snippets returned. Estimated tokens saved is `(file chars − snippet chars) / 4` (4 chars per token). This is a conservative estimate: the baseline is reading matched files in full, which is how coding agents often explore unfamiliar code.

Stats are stored in `~/.semble/savings.jsonl`.

</details>

<details>
<summary>Library usage</summary>

Semble can also be used as a Python library for programmatic access, useful when building custom tooling or integrating search directly into your own code.

```python
from semble import ContentType, SembleIndex

# Index a local directory (code only, the default)
index = SembleIndex.from_path("./my-project")

# Index docs and prose (markdown, rst, etc.)
index = SembleIndex.from_path("./my-project", content=ContentType.DOCS)

# Index everything (code, docs, and config)
index = SembleIndex.from_path("./my-project", content=[ContentType.CODE, ContentType.DOCS, ContentType.CONFIG])

# Index code and docs together
index = SembleIndex.from_path("./my-project", content=[ContentType.CODE, ContentType.DOCS])

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

</details>

## Benchmarks

We benchmark quality and speed across ~1,250 queries over 63 repositories in 19 languages (left), and token efficiency against grep+read at equivalent recall levels (right).

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/speed_vs_ndcg_cold.png" alt="Speed vs quality"></td>
<td><img src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/token_efficiency.png" alt="Token efficiency: recall vs. retrieved tokens"></td>
</tr>
</table>

The quality benchmark (left) scores retrieval quality (NDCG@10) against total latency; semble achieves 99% of the quality of the 137M-parameter [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) Hybrid while indexing 218x faster. The token efficiency benchmark (right) measures how many tokens each method needs to reach a given recall level; semble uses 98% fewer tokens on average and hits 94% recall at only 2k tokens, while grep+read needs a full 100k context window to reach 85%. See [benchmarks](benchmarks/README.md) for per-language results, ablations, and full methodology.

## How it works

Semble splits each file into code-aware chunks using [tree-sitter](https://github.com/tree-sitter/py-tree-sitter), then scores every query against the chunks with two complementary retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec) embeddings using the code-specialized [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) model for semantic similarity, and [BM25](https://github.com/xhluca/bm25s) for lexical matches on identifiers and API names. The two score lists are fused with Reciprocal Rank Fusion (RRF).

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

## License

MIT

## Citing

If you use Semble in your research, please cite the following:

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast and Accurate Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19785932},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
