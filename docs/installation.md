# Installation

## Recommended: `semble install`

The interactive installer detects your installed agents and configures any combination of three integrations globally:

- **[MCP server](#mcp-server)**: exposes Semble as a native tool your agent can call directly.
- **[AGENTS.md](#instructions-agentsmd--claudemd)**: adds a Semble usage guide to the agent's config file (`CLAUDE.md`, `AGENTS.md`, etc.).
- **[Sub-agent](#sub-agent)**: installs a dedicated `semble-search` sub-agent for harnesses that support it.

Install the CLI with [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv tool install semble
semble install
```

To undo:

```bash
semble uninstall
```

Supported agents: Claude Code, Cursor, Gemini CLI, Kiro, OpenCode, GitHub Copilot, Codex, VS Code, Windsurf, Zed, Reasonix, Pi, Command Code, and Antigravity.

> **Pi prerequisite:** Pi requires the MCP extension to be installed before semble can connect. Run `pi install npm:pi-mcp-extension` once, then `semble install`.

---

## Manual setup

### MCP server

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

<details>
<summary>Claude Code</summary>

```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble
```

</details>

<details>
<summary>Cursor</summary>

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

</details>

<details>
<summary>Codex</summary>

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.semble]
command = "uvx"
args = ["--from", "semble[mcp]", "semble"]
```

</details>

<details>
<summary>OpenCode</summary>

Add to `~/.config/opencode/opencode.jsonc`:

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

</details>

<details>
<summary>VS Code</summary>

Add to `.vscode/mcp.json` in your project (or your user profile's `mcp.json`):

```json
{
  "servers": {
    "semble": {
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

</details>

<details>
<summary>GitHub Copilot CLI</summary>

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

</details>

<details>
<summary>Windsurf</summary>

Add to `~/.codeium/windsurf/mcp_config.json`:

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

</details>

<details>
<summary>Gemini CLI</summary>

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

</details>

<details>
<summary>Kiro</summary>

Add to `~/.kiro/settings/mcp.json` (or `.kiro/settings/mcp.json` in your project):

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

</details>

<details>
<summary>Zed</summary>

Add to `~/.config/zed/settings.json` (or `.zed/settings.json` in your project):

```json
{
  "context_servers": {
    "semble": {
      "source": "custom",
      "command": "uvx",
      "args": ["--from", "semble[mcp]", "semble"]
    }
  }
}
```

</details>

<details>
<summary>Reasonix</summary>

Add to `~/.reasonix/config.json` (the backwards-compatible MCP config path read by all Reasonix versions):

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

</details>

<details>
<summary>Pi</summary>

First install the Pi MCP extension (one-time prerequisite):

```bash
pi install npm:pi-mcp-extension
```

Then add to `~/.pi/agent/mcp.json`:

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

</details>

<details>
<summary>Antigravity</summary>

Add to `~/.gemini/config/mcp_config.json`:

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

</details>

<details>
<summary>Command Code</summary>

Add to `~/.commandcode/mcp.json`:

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

Or use the CLI:

```bash
cmd mcp add --scope user semble -- uvx --from "semble[mcp]" semble
```

</details>

By default the MCP server indexes only code files. To also index documentation, config, or everything, append `--content docs`, `--content config`, or `--content all` to the server command. For example, in Claude Code:

```bash
claude mcp add semble -s user -- uvx --from "semble[mcp]" semble --content all
```

### Instructions (AGENTS.md / CLAUDE.md)

Add the snippet below to your `AGENTS.md` or `CLAUDE.md` so your agent knows when and how to call the semble CLI:

```markdown
## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project --max-snippet-lines 10  # first 10 lines only, concise
semble search "save_pretrained" ./my-project                          # full chunk content
semble search "save model to disk" ./my-project --top-k 10           # more results
​```

The index is built on first run (and cached for subsequent runs) and invalidated automatically when files change.

Use `--content docs` to search documentation and prose, `--content config` for config files (yaml, toml, etc.), or `--content all` to search code, docs, and config:

​```bash
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble search "authentication" ./my-project --content all
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

`path` defaults to the current directory when omitted; git URLs are accepted.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

### Workflow

1. Start with `semble search` to find relevant chunks. The index is built and cached automatically.
2. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
3. Navigate directly to the returned file and line — do not re-search or grep for the same content.
4. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
5. Use grep only when you need every occurrence of a literal string across the whole repo (e.g., all callers of a renamed function).
```

### Sub-agent

For harnesses that support sub-agents (Claude Code, Cursor, Gemini CLI, Kiro, OpenCode, GitHub Copilot, Codex, Reasonix, Pi, Command Code, Antigravity), you can install a dedicated `semble-search` sub-agent. Copy the appropriate file from [`src/semble/agents/`](../src/semble/agents/) to your agent's agents directory:

> **Pi prerequisite:** Pi sub-agents require the Pi agents extension. Run `pi install npm:pi-agents` once before installing.

| Agent | File | Destination |
|---|---|---|
| Claude Code | `claude.md` | `~/.claude/agents/semble-search.md` |
| Cursor | `cursor.md` | `~/.cursor/agents/semble-search.md` |
| Gemini CLI | `gemini.md` | `~/.gemini/agents/semble-search.md` |
| Kiro | `kiro.md` | `~/.kiro/agents/semble-search.md` |
| OpenCode | `opencode.md` | `~/.config/opencode/agents/semble-search.md` |
| GitHub Copilot | `copilot.md` | `~/.copilot/agents/semble-search.agent.md` |
| Codex | `codex.toml` | `~/.codex/agents/semble-search.toml` |
| Reasonix | `reasonix.md` | `~/.reasonix/skills/semble-search.md` |
| Pi | `pi.md` | `~/.pi/agents/semble-search.md` |
| Command Code | `commandcode.md` | `~/.commandcode/agents/semble-search.md` |
| Antigravity | `antigravity.md` | `~/.gemini/config/skills/semble-search/SKILL.md` |
