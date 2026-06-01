from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, Literal, Sequence

_HOME = Path.home()

SEMBLE_START = "<!-- SEMBLE_START -->"
SEMBLE_END = "<!-- SEMBLE_END -->"

_STDIO_ENTRY: dict[str, object] = {
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
    "type": "stdio",
}

_OPENCODE_ENTRY: dict[str, object] = {
    "command": ["uvx", "--from", "semble[mcp]", "semble"],
    "type": "stdio",
    "enabled": True,
}

_INSTRUCTIONS = f"""\
{SEMBLE_START}
## Semble Code Search

A `semble` MCP server is available with two tools:
- `mcp__semble__search` — search the codebase with a natural-language or code query.
- `mcp__semble__find_related` — find code similar to a specific file and line.

Always call `mcp__semble__search` before using Grep, Glob, or Read to explore the codebase. Use Grep/Glob/Read only for exact path lookup, exhaustive literal matches, or when the returned chunk lacks enough context.

For CLI fallback or sub-agents without MCP access, use:

```bash
semble search "authentication flow" .
semble search "deployment guide" . --content docs
semble find-related src/auth.py 42 .
```

The index is built on first run and cached automatically. If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble`.
{SEMBLE_END}
"""

Action = Literal["created", "updated", "unchanged", "not-found", "removed"]

_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


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
    mcp_path: Path | None  # None = MCP not supported (or resolved dynamically, e.g. opencode)
    mcp_key: str  # top-level JSON key: "mcpServers" or "mcp"
    mcp_entry: dict[str, object]  # value written under mcp_key.semble
    instructions_path: Path | None  # None = not supported for this agent
    subagent_path: Path | None = None  # global (user-level) sub-agent file; None = unsupported


def _opencode_mcp_path() -> Path:
    """Return the opencode config path, preferring .jsonc over .json."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) / "opencode" if xdg else _HOME / ".config" / "opencode"
    jsonc = base / "opencode.jsonc"
    json_ = base / "opencode.json"
    return jsonc if jsonc.exists() else (json_ if json_.exists() else jsonc)


AGENTS: list[AgentTarget] = [
    AgentTarget(
        id="claude",
        display_name="Claude Code",
        binary="claude",
        config_dir=_HOME / ".claude",
        mcp_path=_HOME / ".claude.json",
        mcp_key="mcpServers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=_HOME / ".claude" / "CLAUDE.md",
        subagent_path=_HOME / ".claude" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="cursor",
        display_name="Cursor",
        binary="cursor",
        config_dir=_HOME / ".cursor",
        mcp_path=_HOME / ".cursor" / "mcp.json",
        mcp_key="mcpServers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=None,  # Cursor instructions are project-local .mdc files
        subagent_path=_HOME / ".cursor" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="gemini",
        display_name="Gemini CLI",
        binary="gemini",
        config_dir=_HOME / ".gemini",
        mcp_path=_HOME / ".gemini" / "settings.json",
        mcp_key="mcpServers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="kiro",
        display_name="Kiro",
        binary="kiro",
        config_dir=_HOME / ".kiro",
        mcp_path=_HOME / ".kiro" / "settings" / "mcp.json",
        mcp_key="mcpServers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=_HOME / ".kiro" / "steering" / "semble.md",
        subagent_path=_HOME / ".kiro" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="opencode",
        display_name="Opencode",
        binary="opencode",
        config_dir=_HOME / ".config" / "opencode",
        mcp_path=None,  # resolved dynamically via _mcp_path()
        mcp_key="mcp",
        mcp_entry=_OPENCODE_ENTRY,
        instructions_path=_HOME / ".config" / "opencode" / "AGENTS.md",
        subagent_path=_HOME / ".config" / "opencode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="copilot",
        display_name="GitHub Copilot",
        binary=None,
        config_dir=_HOME / ".config" / "github-copilot",
        mcp_path=None,  # no stable global MCP config path
        mcp_key="mcpServers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=None,
        subagent_path=_HOME / ".copilot" / "agents" / "semble-search.agent.md",
    ),
]


def _mcp_path(agent: AgentTarget) -> Path | None:
    """Resolve the agent's MCP config path, or None if MCP is unsupported."""
    return _opencode_mcp_path() if agent.id == "opencode" else agent.mcp_path


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON or JSONC file, stripping line comments. Returns {} if missing or unparseable."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*", "", text)  # strip // line comments (JSONC)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, data: dict[str, object]) -> None:
    """Write data to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def merge_mcp(agent: AgentTarget) -> WriteResult:
    """Surgically add the semble MCP entry to the agent's config file."""
    path = _mcp_path(agent)
    assert path is not None
    existed = path.exists()

    config = _read_json(path)
    section = config.get(agent.mcp_key)
    if not isinstance(section, dict):
        section = {}
    section["semble"] = agent.mcp_entry
    config[agent.mcp_key] = section
    _write_json(path, config)

    return WriteResult(path=path, action="updated" if existed else "created")


def remove_mcp(agent: AgentTarget) -> WriteResult:
    """Surgically remove the semble MCP entry from the agent's config file."""
    path = _mcp_path(agent)
    assert path is not None

    if not path.exists():
        return WriteResult(path=path, action="not-found")

    config = _read_json(path)
    section = config.get(agent.mcp_key)
    if not isinstance(section, dict) or "semble" not in section:
        return WriteResult(path=path, action="not-found")

    del section["semble"]
    if not section:
        del config[agent.mcp_key]
    _write_json(path, config)
    return WriteResult(path=path, action="removed")


def replace_or_append_marked(path: Path, content: str) -> Action:
    """Replace the marked semble section in path, or append it if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    existing = path.read_text(encoding="utf-8") if existed else ""

    start_idx = existing.find(SEMBLE_START)
    end_idx = existing.find(SEMBLE_END)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        before = existing[:start_idx]
        after = existing[end_idx + len(SEMBLE_END) :]
        updated = before + content.strip("\n") + "\n" + after.lstrip("\n")
        if updated == existing:
            return "unchanged"
        path.write_text(updated, encoding="utf-8")
        return "updated"

    separator = "\n\n" if existing and not existing.endswith("\n\n") else "\n" if existing else ""
    path.write_text(existing + separator + content, encoding="utf-8")
    return "created" if not existed else "updated"


def remove_marked(path: Path) -> Action:
    """Remove the marked semble section from path."""
    if not path.exists():
        return "not-found"

    existing = path.read_text(encoding="utf-8")
    start_idx = existing.find(SEMBLE_START)
    end_idx = existing.find(SEMBLE_END)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return "not-found"

    before = existing[:start_idx].rstrip("\n")
    after = existing[end_idx + len(SEMBLE_END) :].lstrip("\n")
    updated = (before + "\n" + after).strip("\n") + ("\n" if existing.endswith("\n") else "")

    if updated == existing:
        return "unchanged"

    if updated.strip():
        path.write_text(updated, encoding="utf-8")
    else:
        path.unlink()
    return "removed"


def _install_mcp(agent: AgentTarget) -> WriteResult | None:
    return merge_mcp(agent) if _mcp_path(agent) is not None else None


def _uninstall_mcp(agent: AgentTarget) -> WriteResult | None:
    return remove_mcp(agent) if _mcp_path(agent) is not None else None


def _install_instructions(agent: AgentTarget) -> WriteResult | None:
    if agent.instructions_path is None:
        return None
    return WriteResult(agent.instructions_path, replace_or_append_marked(agent.instructions_path, _INSTRUCTIONS))


def _uninstall_instructions(agent: AgentTarget) -> WriteResult | None:
    if agent.instructions_path is None:
        return None
    return WriteResult(agent.instructions_path, remove_marked(agent.instructions_path))


def _install_subagent(agent: AgentTarget) -> WriteResult | None:
    dest = agent.subagent_path
    if dest is None:
        return None
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath(f"agents/{agent.id}.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    return WriteResult(dest, "updated" if existed else "created")


def _uninstall_subagent(agent: AgentTarget) -> WriteResult | None:
    dest = agent.subagent_path
    if dest is None:
        return None
    if not dest.exists():
        return WriteResult(dest, "not-found")
    dest.unlink()
    return WriteResult(dest, "removed")


@dataclass(frozen=True)
class _Integration:
    id: str
    label: str
    desc: str
    install: Callable[[AgentTarget], WriteResult | None]
    uninstall: Callable[[AgentTarget], WriteResult | None]
    plan_path: Callable[[AgentTarget], Path | None]


INTEGRATIONS: list[_Integration] = [
    _Integration(
        "mcp", "MCP server", "registers semble as a tool in the agent", _install_mcp, _uninstall_mcp, _mcp_path
    ),
    _Integration(
        "instructions",
        "Instructions",
        "adds usage guide to the agent's config file",
        _install_instructions,
        _uninstall_instructions,
        lambda a: a.instructions_path,
    ),
    _Integration(
        "subagent",
        "Sub-agent",
        "installs a global semble-search sub-agent (available in all projects)",
        _install_subagent,
        _uninstall_subagent,
        lambda a: a.subagent_path,
    ),
]


def is_detected(agent: AgentTarget) -> bool:
    """Return True if the agent appears to be installed."""
    if agent.binary and shutil.which(agent.binary):
        return True
    return bool(agent.config_dir and agent.config_dir.exists())


def _tick(ok: bool) -> str:
    return f"{_GREEN}✓{_RESET}" if ok else f"{_DIM}–{_RESET}"


def _checkbox(prompt: str, items: Sequence[tuple[str, object, bool]]) -> list[object] | None:
    """Multi-select checkbox: arrows navigate, space toggles, enter confirms.

    :param prompt: Question shown above the list.
    :param items: (label, value, checked) tuples for each selectable row.
    :return: Selected values, or None if the user cancelled (Ctrl-C).
    """
    import questionary

    # prompt_toolkit defaults "selected" to reverse-video (a filled block); override it
    # so checked rows show a clean green ● and the cursor row is just bold.
    style = questionary.Style(
        [
            ("pointer", "bold"),
            ("highlighted", "noreverse bold"),
            ("selected", "noreverse fg:ansigreen"),
        ]
    )
    choices = [questionary.Choice(title=label, value=value, checked=checked) for label, value, checked in items]
    instruction = "(↑↓ move · space select · a all · enter confirm)"
    return questionary.checkbox(prompt, choices=choices, style=style, instruction=instruction).ask()


def _print_plan(agents: list[AgentTarget], integrations: list[_Integration]) -> None:
    print(f"\n  {_BOLD}Plan:{_RESET}\n")
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            path = integ.plan_path(agent)
            ok = path is not None
            print(f"    {integ.label:<13} {_tick(ok)}  {path if ok else '(not supported)'}")
    print()


def _apply(mode: Literal["install", "uninstall"], agents: list[AgentTarget], integrations: list[_Integration]) -> None:
    print()
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            result = (integ.install if mode == "install" else integ.uninstall)(agent)
            if result is None:
                print(f"    {_DIM}– {integ.id}: not supported{_RESET}")
                continue
            ok = result.action not in ("not-found", "unchanged")
            print(f"    {_tick(ok)} {integ.id} ({result.action}) → {result.path}")
        print()


def _run(mode: Literal["install", "uninstall"]) -> None:
    import questionary

    install = mode == "install"
    print(f"\n  {_BOLD}{'Semble Installer' if install else 'Semble Uninstaller'}{_RESET}\n")

    agent_items: list[tuple[str, object, bool]] = []
    for agent in AGENTS:
        detected = is_detected(agent)
        label = f"{agent.display_name}{'  (detected)' if detected else ''}"
        agent_items.append((label, agent, detected and install))  # pre-check detected agents on install
    chosen_agents = _checkbox(
        f"Select agents to {'configure' if install else 'remove configuration from'}:", agent_items
    )
    if not chosen_agents:
        print("Nothing selected. Exiting.")
        sys.exit(0)

    integ_items = [(f"{i.label}  —  {i.desc}", i, True) for i in INTEGRATIONS]
    chosen_integrations = _checkbox(f"Select integrations to {'enable' if install else 'remove'}:", integ_items)
    if not chosen_integrations:
        print("Nothing selected. Exiting.")
        sys.exit(0)

    _print_plan(chosen_agents, chosen_integrations)  # type: ignore[arg-type]

    question = "Proceed?" if install else "Remove semble configuration?"
    if not questionary.confirm(question, default=install).ask():
        print("Cancelled.")
        sys.exit(0)

    _apply(mode, chosen_agents, chosen_integrations)  # type: ignore[arg-type]
    footer = " Restart your agents to pick up the changes." if install else ""
    print(f"  {_GREEN}Done!{_RESET}{footer}\n")


def run_install() -> None:
    """Interactive semble installer."""
    _run("install")


def run_uninstall() -> None:
    """Interactive semble uninstaller."""
    _run("uninstall")
