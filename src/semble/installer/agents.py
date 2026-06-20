from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_HOME = Path.home()

Action = Literal["created", "updated", "unchanged", "not-found", "removed", "error", "skipped"]
Mode = Literal["install", "uninstall"]

SEMBLE_START = "<!-- SEMBLE_START -->"
SEMBLE_END = "<!-- SEMBLE_END -->"

_STDIO_SERVER_CONFIG: dict[str, object] = {
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
    "type": "stdio",
}

_OPENCODE_SERVER_CONFIG: dict[str, object] = {
    "command": ["uvx", "--from", "semble[mcp]", "semble"],
    "type": "local",  # opencode uses "local"/"remote", not "stdio"
    "enabled": True,
}

_BARE_STDIO_SERVER_CONFIG: dict[str, object] = {  # Windsurf: command/args only, no "type"
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
}

_ZED_SERVER_CONFIG: dict[str, object] = {  # Zed requires "source": "custom" for manual servers
    "source": "custom",
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
}

INSTRUCTIONS = f"""\
{SEMBLE_START}
## Semble Code Search

A `semble` MCP server is available with two tools:
- `mcp__semble__search` — search the codebase with a natural-language or code query.
- `mcp__semble__find_related` — find code similar to a specific file and line.

Use `mcp__semble__search` to find where something is implemented — instead of using Grep or Glob to discover files. After semble returns the file and line, navigate there directly and read that file. Do not grep for the same content again.

Pass `--content docs` to search documentation and prose, `--content config` for config files, or `--content all` to search code, docs, and config together.

For CLI fallback or sub-agents without MCP access, use:

```bash
semble search "authentication flow" ./my-project --max-snippet-lines 10
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble find-related src/auth.py 42 ./my-project
semble search "save model to disk" ./my-project --top-k 10
```

The index is built on first run and cached automatically. If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble`.

### Workflow

1. Call `mcp__semble__search` with a query describing what the code does or its name. The tool returns results with 10 lines of context each (function/class signature + first body lines, enough to confirm the location).
2. Navigate directly to the top result's file and line. Read only the function or class at that location.
3. Make the edit. Do not re-search or grep for the same content.
4. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
5. Optionally use `mcp__semble__find_related` with `file_path` and `line` to discover similar code elsewhere.
6. Use Grep only when you need every occurrence of a literal string across the whole repo (e.g., all callers of a renamed function).
{SEMBLE_END}
"""


@dataclass(frozen=True)
class McpConfig:
    """MCP integration config for one agent."""

    path: Path
    key: str
    entry: dict[str, object]
    format: Literal["json", "toml"] = "json"


@dataclass(frozen=True)
class WriteResult:
    """Result of a single file write operation."""

    path: Path
    action: Action


@dataclass(frozen=True)
class AgentTarget:
    """Configuration for a single coding agent integration target."""

    id: str
    display_name: str
    binary: str | None  # for shutil.which detection
    config_dir: Path | None  # directory existence check for detection
    mcp: McpConfig | None
    instructions_path: Path | None  # None = not supported for this agent
    subagent_path: Path | None = None  # global (user-level) sub-agent file; None = unsupported

    def resolved_mcp_path(self) -> Path | None:
        """Return the resolved MCP config path, or None if MCP is unsupported."""
        return self.mcp.path if self.mcp else None


def _opencode_mcp_path() -> Path:
    """Return the opencode config path, preferring .jsonc over .json."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "opencode" if xdg else _HOME / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    json_ = base / "opencode.json"
    return jsonc if jsonc.exists() else (json_ if json_.exists() else jsonc)


def _vscode_mcp_path() -> Path:
    """Return the user-level VS Code mcp.json path for the current OS."""
    if sys.platform == "darwin":
        base = _HOME / "Library" / "Application Support" / "Code" / "User"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", _HOME)) / "Code" / "User"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", _HOME / ".config")) / "Code" / "User"
    return base / "mcp.json"


AGENTS: list[AgentTarget] = [
    AgentTarget(
        id="claude",
        display_name="Claude Code",
        binary="claude",
        config_dir=_HOME / ".claude",
        mcp=McpConfig(_HOME / ".claude.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".claude" / "CLAUDE.md",
        subagent_path=_HOME / ".claude" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="cursor",
        display_name="Cursor",
        binary="cursor",
        config_dir=_HOME / ".cursor",
        mcp=McpConfig(_HOME / ".cursor" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=None,  # Cursor instructions are project-local .mdc files
        subagent_path=_HOME / ".cursor" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="gemini",
        display_name="Gemini CLI",
        binary="gemini",
        config_dir=_HOME / ".gemini",
        mcp=McpConfig(_HOME / ".gemini" / "settings.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="kiro",
        display_name="Kiro",
        binary="kiro",
        config_dir=_HOME / ".kiro",
        mcp=McpConfig(_HOME / ".kiro" / "settings" / "mcp.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".kiro" / "steering" / "semble.md",
        subagent_path=_HOME / ".kiro" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="opencode",
        display_name="Opencode",
        binary="opencode",
        config_dir=_HOME / ".config" / "opencode",
        mcp=McpConfig(_opencode_mcp_path(), "mcp", _OPENCODE_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "opencode" / "AGENTS.md",
        subagent_path=_HOME / ".config" / "opencode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="copilot",
        display_name="GitHub Copilot",
        binary=None,
        config_dir=_HOME / ".config" / "github-copilot",
        mcp=McpConfig(_HOME / ".copilot" / "mcp-config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".copilot" / "agents" / "semble-search.agent.md",
    ),
    AgentTarget(
        id="codex",
        display_name="Codex",
        binary="codex",
        config_dir=_HOME / ".codex",
        mcp=McpConfig(_HOME / ".codex" / "config.toml", "mcp_servers", _STDIO_SERVER_CONFIG, format="toml"),
        instructions_path=_HOME / ".codex" / "AGENTS.md",
        subagent_path=_HOME / ".codex" / "agents" / "semble-search.toml",
    ),
    AgentTarget(
        id="vscode",
        display_name="VS Code",
        binary="code",
        config_dir=None,
        mcp=McpConfig(_vscode_mcp_path(), "servers", _STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="windsurf",
        display_name="Windsurf",
        binary="windsurf",
        config_dir=_HOME / ".codeium" / "windsurf",
        mcp=McpConfig(_HOME / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="zed",
        display_name="Zed",
        binary="zed",
        config_dir=_HOME / ".config" / "zed",
        mcp=McpConfig(_HOME / ".config" / "zed" / "settings.json", "context_servers", _ZED_SERVER_CONFIG),
        instructions_path=None,
    ),
    AgentTarget(
        id="reasonix",
        display_name="Reasonix",
        binary="reasonix",
        config_dir=_HOME / ".config" / "reasonix",
        # ~/.reasonix/config.json is the legacy v0.x path still read by v1.x for backwards compat.
        # The v1.x canonical config is ~/.config/reasonix/config.toml ([[plugins]]), but the JSON
        # path requires no special TOML handling and works for new users who have never had v0.x.
        mcp=McpConfig(_HOME / ".reasonix" / "config.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".config" / "reasonix" / "REASONIX.md",
        subagent_path=_HOME / ".reasonix" / "skills" / "semble-search.md",
    ),
    AgentTarget(
        id="pi",
        display_name="Pi",
        binary="pi",
        config_dir=_HOME / ".pi",
        mcp=McpConfig(_HOME / ".pi" / "agent" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=None,
        subagent_path=_HOME / ".pi" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="commandcode",
        display_name="Command Code",
        binary=None,
        config_dir=_HOME / ".commandcode",
        mcp=McpConfig(_HOME / ".commandcode" / "mcp.json", "mcpServers", _BARE_STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".commandcode" / "AGENTS.md",
        subagent_path=_HOME / ".commandcode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="antigravity",
        display_name="Antigravity",
        binary="agy",
        config_dir=_HOME / ".gemini" / "antigravity-cli",
        mcp=McpConfig(_HOME / ".gemini" / "config" / "mcp_config.json", "mcpServers", _STDIO_SERVER_CONFIG),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "config" / "skills" / "semble-search" / "SKILL.md",
    ),
]


def is_detected(agent: AgentTarget) -> bool:
    """Return True if the agent appears to be installed."""
    if agent.binary and shutil.which(agent.binary):
        return True
    return bool(agent.config_dir and agent.config_dir.exists())
