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

## Python API

```python
from semble import SearchMode, SembleIndex

index = SembleIndex()
index.index("./my-project")

# Hybrid search (semantic + BM25)
results = index.search("how does authentication work?", top_k=5)
for r in results:
    print(r.chunk.location, f"score={r.score:.3f}")
    print(r.chunk.content[:200])

# Keyword-only
results = index.search("JWT token", mode=SearchMode.BM25)

# Symbol search (find function/class definitions)
results = index.search("UserService", mode=SearchMode.SYMBOL)
```

## Search modes

| Mode | Description |
|------|-------------|
| `hybrid` | Semantic + BM25, normalized and combined (default) |
| `semantic` | Embedding similarity only |
| `bm25` | Keyword search only |
| `symbol` | Grep-style search for function/class definitions |
