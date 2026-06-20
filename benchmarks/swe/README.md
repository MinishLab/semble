# SWE-bench Semble Experiment

Evaluates whether semble's MCP server helps coding agents (Claude Code, Codex, opencode) fix real-world GitHub issues faster and more accurately. All three backends are wired up the same way: an MCP server, present or absent.

## Files

| File | Purpose |
|------|---------|
| `agent_run.py` | Main run loop, CLI, summary printing, and prediction file generation. |
| `evaluate.py` | Runs the official SWE-bench Docker test harness on generated patches to get real resolve rates. |
| `tasks.py` | `SWETask` wrapper plus SWE-bench Lite task loading and sampling. |
| `utils.py` | Shared constants, result dataclasses, git helpers, and lightweight stats helpers. |

### Directories

| Directory | Contents |
|-----------|----------|
| `backends/` | Backend implementations for `claude`, `codex`, and `opencode`, plus the shared backend base class. |
| `results/` | `swe_agent.json` (raw results), `predictions_*.jsonl` (SWE-bench format patches), harness output JSONs |
| `repos/` | Cloned repositories at base commits (gitignored, auto-cloned) |

## CLI

### agent_run.py — run agents on SWE-bench tasks

```bash
# Claude Code (default)
uv run python -m benchmarks.swe.agent_run --backend claude --model claude-haiku-4-5-20251001 --tasks 5

# Codex (OpenAI)
uv run python -m benchmarks.swe.agent_run --backend codex --model gpt-5.4-mini --tasks 5

# opencode (opencode go)
uv run python -m benchmarks.swe.agent_run --backend opencode --model opencode-go/deepseek-v4-pro --tasks 5

# Specific tasks (e.g. re-run failures)
uv run python -m benchmarks.swe.agent_run --backend codex --model gpt-5.4-mini \
  --instance-ids pytest-dev__pytest-7220 pytest-dev__pytest-7373

# Different repo
uv run python -m benchmarks.swe.agent_run --backend codex --model gpt-5.4-mini --tasks 5 --repo psf/requests

# Use the local branch's semble instead of the PyPI release
uv run python -m benchmarks.swe.agent_run --backend claude --tasks 5 --local-semble
```

**Flags:**
- `--backend` — `claude`, `codex`, or `opencode` (default: `claude`)
- `--model` — model name (defaults vary by backend)
- `--tasks` — number of tasks from SWE-bench Lite (default: 5, ignored if `--instance-ids` given)
- `--instance-ids` — specific instance IDs to run (space-separated)
- `--repo` — GitHub repo, e.g. `pytest-dev/pytest` (default)
- `--resume` — skip tasks already successfully completed for the current backend+model
- `--experiment` — saves to `swe_agent_{NAME}.json` and tags the with-semble variant
- `--with-semble-only` — only run the WITH semble variant (skip WITHOUT semble)
- `--local-semble` — use the local branch's semble (`uv run semble`) instead of the installed PyPI version

**How it works:**
1. Fetches task definitions from HuggingFace (SWE-bench Lite)
2. For each task, clones the repo at the base commit
3. Runs the agent twice per task, each time wiring or omitting the semble MCP server:
   - **With Semble** — semble registered as an MCP server
   - **Without Semble** — no MCP servers registered at all
4. Records gold file hits, token usage, cost, and the generated patch
5. After every completed variant, atomically updates `results/swe_agent.json` or `results/swe_agent_{EXPERIMENT}.json`
6. Generates SWE-bench prediction JSONL files for `evaluate.py`

### Long-running usage

This runner is intended to survive model/session limits across many restarts.

Recommended pattern:

```bash
uv run python -m benchmarks.swe.agent_run \
  --backend codex \
  --model gpt-5.4-mini \
  --repo all \
  --tasks 200 \
  --resume
```

If the process stops because the agent hits a limit, just run the same command again with `--resume`.

Durability behavior:
- Each completed variant is flushed to disk immediately, not only at the end of the full benchmark.
- Writes to `swe_agent*.json` and `predictions_*.jsonl` are atomic file replacements.
- Resume skips variants already present for the same `(backend, model, variant)` slot.
- If the process dies during a variant before that variant returns, only that in-progress variant is lost; earlier completed variants remain on disk.

For experiment runs, keep the same `--experiment NAME` value on every restart.

### evaluate.py — real test-suite verification

```bash
# Run on all instances with predictions
uv run python -m benchmarks.swe.evaluate

# Run on specific instances
uv run python -m benchmarks.swe.evaluate --instance-ids pytest-dev__pytest-11143 pytest-dev__pytest-7220
```

Requires Docker (OrbStack works). Pulls SWE-bench Docker images and runs the actual pytest test suite on each patch. Produces `results/swe_resolve.json` or `results/swe_resolve_{EXPERIMENT}.json` with real resolve rates.

## Output format

### swe_agent.json (merged run results)
```json
[{
  "instance_id": "pytest-dev__pytest-11143",
  "gold_files": ["src/_pytest/assertion/rewrite.py"],
  "results": [
    {"variant": "with_semble", "backend": "claude", "model": "claude-haiku-4-5-20251001", "gold_hit": true, "cost_usd": 0.092, "num_turns": 12, ...},
    {"variant": "without_semble", "backend": "claude", "gold_hit": true, "cost_usd": 0.368, "num_turns": 37, ...}
  ]
}]
```

### predictions_*.jsonl (SWE-bench format)
```json
{"instance_id": "pytest-dev__pytest-11143", "model_patch": "diff --git ...", "model_name_or_path": "claude-claude-haiku-4-5-20251001-with_semble"}
```

For experiment runs, only `predictions_with_semble_{NAME}.jsonl` is suffixed. `predictions_without_semble.jsonl` remains the shared baseline file that `evaluate.py` reads.

### swe_resolve.json (harness results)
```json
{"instances": [...], "results": {"with_semble": {"pytest-dev__pytest-11143": true, ...}, "without_semble": {...}}}
```

## Architecture

```text
agent_run.py
  ├── uses tasks.py for dataset loading / sampling
  ├── uses utils.py for shared constants, result types, git helpers, and stats
  ├── uses backends/ for agent-specific execution
  └── run(backend, tasks, repo, instance_ids)
      ├── Fetch tasks from HuggingFace
      ├── For each task × each variant (with/without):
      │   ├── Clone repo at base commit
      │   ├── Run agent with prompt
      │   ├── git diff → patch
      │   └── git reset
      └── Save + print summary

backends/
  ├── base.py         → shared subprocess / timeout / reset logic
  ├── claude.py       → claude -p ... --mcp-config
  ├── codex.py        → codex exec --json ... with CODEX_HOME override
  └── opencode.py     → opencode run ... with XDG_CONFIG_HOME override

evaluate.py
  └── Run swebench.harness.run_evaluation on predictions_*.jsonl
```

## Semble control per backend

All three backends are MCP-only — semble is either registered as an MCP server, or no MCP servers exist at all for that run.

| Backend | With Semble | Without Semble |
|---------|------------|----------------|
| Claude | `--mcp-config <temp file with semble entry>` | `--mcp-config <temp file with empty mcpServers>`, both with `--strict-mcp-config` |
| Codex | `[mcp_servers.semble]` in a temp `config.toml`, `CODEX_HOME` override | Same temp config with the `[mcp_servers.semble]` section stripped |
| opencode | `mcp.semble.enabled: true` in a temp `opencode.json`, `XDG_CONFIG_HOME` override | Same temp config with `mcp.semble.enabled: false` |

## Validation

Resume/persistence smoke test:

```bash
uv run pytest tests/test_swe_agent_run.py
```

This test covers:
- persisting a completed variant before an interrupted second variant
- resuming and running only the missing variant
- overwriting an existing `(backend, model, variant)` result instead of duplicating it
