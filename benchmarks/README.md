# Benchmarks

Reproducible local benchmarks for `semble`.

Pinned repositories live in `repos.json` and are checked out into `/tmp/bench`.

## Setup

```bash
uv run python -m benchmarks.sync_repos
uv run python -m benchmarks.sync_repos --check
```

## Run

```bash
uv run python -m benchmarks.bench_hybrid
uv run python -m benchmarks.bench_hybrid --cache
uv run python -m benchmarks.bench_hybrid --repo fastapi --repo axios
uv run python -m benchmarks.bench_hybrid --language python
```

`--cache` measures cold vs warm index time. Warm time still includes the file walk and
BM25/Vicinity rebuild; only embedding is skipped.
