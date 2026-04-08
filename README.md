# semble

Instant local code search for agents. Static embeddings, hybrid search, zero API keys.

## Install

```bash
pip install semble
```

## Python

```python
from semble import SembleIndex

index = SembleIndex()
index.index_directory(".")

results = index.search("how does auth work?", top_k=5)
for result in results:
    print(result.chunk.location, result.score)
```

## CLI

```bash
semble index .
semble search "how does auth work?" .
semble search "authenticate" . --mode symbol
```
