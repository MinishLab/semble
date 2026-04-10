# semble

Instant local code search for agents.

## Install

```bash
pip install semble
```

## Python API

```python
from semble import SearchMode, SembleIndex

index = SembleIndex.from_path("./my-project")

# Hybrid search (semantic + BM25, default)
results = index.search("how does authentication work?", top_k=5)
for r in results:
    print(r.chunk.location, f"score={r.score:.3f}")
    print(r.chunk.content[:200])

# Keyword-only
results = index.search("JWT token", mode=SearchMode.BM25)
```

## Search modes

| Mode | Description |
|------|-------------|
| `hybrid` | Semantic + BM25, normalized and combined (default) |
| `semantic` | Embedding similarity only |
| `bm25` | Keyword search only |

## Disk embedding cache

Embeddings are cached to `~/.cache/semble` by default so re-indexing unchanged files is instant. When using a custom encoder, pass `model_name` to enable caching:

```python
index = SembleIndex.from_path("./my-project", model=my_model, model_name="my-org/my-model")
```

Only embeddings are cached; BM25 and the ANNS index are always rebuilt fresh.

## MCP server

Semble can run as an MCP server so agents (Claude Code, Cursor, etc.) can search your codebase directly.

Install with the MCP extra:

```bash
pip install "semble[mcp]"
```

Register with Claude Code:

```bash
claude mcp add semble -- uvx --from "semble[mcp]" semble /path/to/repo
```

This indexes the directory at startup and exposes two tools:

| Tool | Description |
|------|-------------|
| `search` | Search with a natural-language or code query. Supports `hybrid` (default), `semantic`, and `bm25` modes. |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |
