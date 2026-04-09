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
