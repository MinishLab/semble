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

Pass `cache_dir` and `model_id` to persist embeddings between runs. Only
embeddings are cached on disk; BM25 and the ANNS index are always rebuilt
in-memory (fast, no staleness risk). `~` is expanded automatically.

```python
# First run: embeds everything and writes to ~/.cache/semble
index = SembleIndex.from_path(
    "./my-project",
    cache_dir="~/.cache/semble",
    model_id="Pringled/potion-code-16M",   # built-in default
)

# Subsequent runs: loads all embeddings from disk, skips encoding entirely
index = SembleIndex.from_path(
    "./my-project",
    cache_dir="~/.cache/semble",
    model_id="Pringled/potion-code-16M",
)
results = index.search("authenticate", mode="bm25")   # works; BM25 needs no model
results = index.search("authenticate")                # also works; default model loaded lazily
```

`model_id` is the cache namespace — embeddings from different models never collide.
The built-in default is `"Pringled/potion-code-16M"`.

**Custom models:** when using a non-default model you must supply the matching
`model` object every time you call `from_path` / `index`. If `model` is `None`
and the recorded `model_id` is not the built-in default, semantic/hybrid search
will raise `ValueError` rather than silently load an incompatible model.

```python
from my_encoder import MyModel

model = MyModel()
index = SembleIndex.from_path(
    "./my-project",
    model=model,
    cache_dir="~/.cache/semble",
    model_id="my-org/my-model",
)
```
