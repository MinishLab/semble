# Benchmarks

Reproducible local benchmarks for `semble`.

Pinned repositories live in `repos.json` and are checked out into `~/.cache/semble-bench`.

## Setup

```bash
uv run python -m benchmarks.sync_repos
uv run python -m benchmarks.sync_repos --check
```

## Run

```bash
uv run python -m benchmarks.run_benchmark
uv run python -m benchmarks.run_benchmark --repo fastapi --repo axios
uv run python -m benchmarks.run_benchmark --language python
```

Full runs (no `--repo`/`--language` filters) automatically save results to
`benchmarks/results/<sha>.json`.
