# Installation

## Install semble

```bash
pip install semble        # with pip
uv tool install semble    # with uv (recommended)
```

To use semble as an MCP server, install the MCP extras:

```bash
pip install "semble[mcp]"
uv tool install "semble[mcp]"
```

<details>
<summary>Updating semble</summary>

```bash
pip install --upgrade semble    # with pip
uv tool upgrade semble          # with uv
uv cache clean semble           # for MCP users (restart your MCP client after)
```

</details>

---

## MCP Setup

The MCP server lets your agent search any codebase directly in natural language. Repos are indexed on demand and cached for the session; local paths are re-indexed automatically on file changes.

> All MCP setups require [uv](https://docs.astral.sh/uv/getting-started/installation/).

### Claude Code

```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble
```

### Gemini CLI

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json` (or `.cursor/mcp.json` in your project):

```json
{
  "mcpServers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.semble]
command = "uvx"
args = ["--from", "semble[mcp]", "semble"]
```

### OpenCode

Add to `~/.opencode/config.json`:

```json
{
  "mcp": {
    "semble": {
      "type": "local",
      "command": ["uvx", "--from", "semble[mcp]", "semble"]
    }
  }
}
```

### Zed

Add to `~/.config/zed/settings.json` (or `.zed/settings.json` in your project):

```json
{
  "context_servers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

### GitHub Copilot CLI

Add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

### Kiro

Add to `~/.kiro/settings/mcp.json` (or `.kiro/settings/mcp.json` in your project):

```json
{
  "mcpServers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `search` | Search a codebase with a natural-language or code query. Pass `repo` as a local path or an `https://` git URL. |
| `find_related` | Given a file path and line number, return chunks semantically similar to the code at that location. |

---

## Sub-agent Setup (`semble init`)

Sub-agents run in their own context window and cannot call MCP tools directly. Use `semble init` to install a dedicated semble search agent so sub-agents can call semble via Bash instead.

Run once in your project root:

```bash
semble init                      # Claude Code (default)
semble init --agent gemini
semble init --agent cursor
semble init --agent opencode
semble init --agent copilot
semble init --agent kiro

# if semble is not on $PATH:
uvx --from "semble[mcp]" semble init --agent copilot
```

| Agent | File written | Source |
|-------|-------------|--------|
| `claude` | `.claude/agents/semble-search.md` | [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code/sub-agents) |
| `gemini` | `.gemini/agents/semble-search.md` | [Gemini CLI docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/core/subagents.md) |
| `cursor` | `.cursor/agents/semble-search.md` | [Cursor docs](https://cursor.com/docs/subagents) |
| `opencode` | `.opencode/agents/semble-search.md` | [OpenCode docs](https://opencode.ai/docs/agents/) |
| `copilot` | `.github/agents/semble-search.md` | [Copilot CLI docs](https://docs.github.com/en/copilot/how-tos/copilot-cli/use-copilot-cli/invoke-custom-agents) |
| `kiro` | `.kiro/agents/semble-search.md` | [Kiro docs](https://kiro.dev/docs/chat/subagents/) |

Add `--force` to overwrite an existing file.

**Codex** uses a skills system rather than an agents directory. Add the [Bash / AGENTS.md snippet](#bash--agentsmd) to your `AGENTS.md` instead — Codex sub-agents will pick it up automatically.

**Zed** is an editor rather than a CLI agent and does not have a sub-agent directory. Use the [MCP setup](#zed) above.

**Hermes** supports subagents via tool delegation rather than a filesystem agents directory. Use the [Bash / AGENTS.md snippet](#bash--agentsmd) to give Hermes access to semble.

---

## Bash / AGENTS.md

An alternative to MCP is to call semble via Bash. This works with every agent and is the only option for sub-agents that cannot access MCP tools (Codex, and any other agent that doesn't support `semble init`).

Install semble, then append the snippet below to your project's `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or equivalent:

```markdown
## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

`path` defaults to the current directory when omitted; git URLs are accepted.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

### Workflow

1. Start with `semble search` to find relevant chunks.
2. Inspect full files only when the returned chunk is not enough context.
3. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
4. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
```
