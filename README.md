# semble

Instant local code search for agents. Static embeddings, hybrid search, zero API keys.

- **No API keys, no GPU** — uses [potion-code-16M](https://huggingface.co/Pringled/potion-code-16M), a 16M parameter static embedding model that runs on CPU in milliseconds
- **Hybrid search** — combines semantic embeddings + BM25 keyword search for best-of-both recall
- **Symbol search** — grep-style search for function/class definitions across any language
- **Fast indexing** — a typical Python package indexes in under 500ms

## Install

```bash
pip install semble
```

For better chunking (recommended), install with the code extra:

```bash
pip install "semble[code]"
```

## Python API

```python
from semble import SembleIndex

index = SembleIndex()
index.index_directory("./my-project")

# Hybrid search (semantic + BM25)
results = index.search("how does authentication work?", top_k=5)
for r in results:
    print(r.chunk.location, f"score={r.score:.3f}")
    print(r.chunk.content[:200])

# Keyword-only
results = index.search("JWT token", mode="bm25")

# Symbol search (find function/class definitions)
results = index.search("UserService", mode="symbol")

# Related code at a given location (for editor integrations)
related = index.get_context("src/auth.py", line=42, top_k=3)
```

## CLI

```bash
# Index a directory and print stats
semble index .

# Search (hybrid by default)
semble search "how does auth work?" .

# Symbol search
semble search "authenticate" . --mode symbol

# More results, semantic only
semble search "request validation" . -k 10 --mode semantic
```

## Search modes

| Mode | Description |
|------|-------------|
| `hybrid` | Semantic + BM25, normalized and combined (default) |
| `semantic` | Embedding similarity only |
| `bm25` | Keyword search only |
| `symbol` | Grep-style search for function/class definitions |
