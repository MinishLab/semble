# semble

Instant local code search for agents.

## Install

```bash
pip install semble
```

## Python API

```python
from semble import SembleIndex

index = SembleIndex.from_path("./my-project")

results = index.search("how does authentication work?", top_k=5)
for result in results:
    print(result.chunk.location, f"score={result.score:.3f}")
    print(result.chunk.content[:200])
```

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
| `search` | Search your codebase with a natural-language or code query. |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |
