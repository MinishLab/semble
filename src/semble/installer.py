from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, Literal, NoReturn, Sequence, TypeVar, cast

import questionary
from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, download, get_parser

_HOME = Path.home()

# Types
Action = Literal["created", "updated", "unchanged", "not-found", "removed", "error", "skipped"]
Mode = Literal["install", "uninstall"]
PathResolver = Callable[[], Path]
JsonObjectResult = tuple[Node, bytes] | Literal["skipped", "error"]
_T = TypeVar("_T")

# Styling / output
_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_ACTION_DETAIL: dict[str, str] = {
    "skipped": "JSON5 grammar unavailable — add manually (see README)",
    "error": "could not parse or edit config",
}

# Config fragments
SEMBLE_START = "<!-- SEMBLE_START -->"
SEMBLE_END = "<!-- SEMBLE_END -->"

_STDIO_ENTRY: dict[str, object] = {
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
    "type": "stdio",
}

_OPENCODE_ENTRY: dict[str, object] = {
    "command": ["uvx", "--from", "semble[mcp]", "semble"],
    "type": "local",  # opencode uses "local"/"remote", not "stdio"
    "enabled": True,
}

_BARE_STDIO_ENTRY: dict[str, object] = {  # Windsurf: command/args only, no "type"
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
}

_ZED_ENTRY: dict[str, object] = {  # Zed requires "source": "custom" for manual servers
    "source": "custom",
    "command": "uvx",
    "args": ["--from", "semble[mcp]", "semble"],
}

# Codex config is TOML; we add/remove this table as text (no stdlib TOML writer on py3.10).
_CODEX_MCP_HEADER = "[mcp_servers.semble]"
_CODEX_MCP_BLOCK = '[mcp_servers.semble]\ncommand = "uvx"\nargs = ["--from", "semble[mcp]", "semble"]\n'

_INSTRUCTIONS = f"""\
{SEMBLE_START}
## Semble Code Search

A `semble` MCP server is available with two tools:
- `mcp__semble__search` — search the codebase with a natural-language or code query.
- `mcp__semble__find_related` — find code similar to a specific file and line.

Always call `mcp__semble__search` before using Grep, Glob, or Read to explore the codebase. Use Grep/Glob/Read only for exact path lookup, exhaustive literal matches, or when the returned chunk lacks enough context.

Pass `--content docs` to search documentation and prose, `--content config` for config files, or `--content all` to search code, docs, and config together.

For CLI fallback or sub-agents without MCP access, use:

```bash
semble search "authentication flow" ./my-project
semble search "deployment guide" ./my-project --content docs
semble search "database host port" ./my-project --content config
semble find-related src/auth.py 42 ./my-project
semble search "save model to disk" ./my-project --top-k 10
```

The index is built on first run and cached automatically. If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble`.

### Workflow

1. Start with `mcp__semble__search` to find relevant chunks.
2. Use `--content docs` for documentation, `--content config` for config files, or `--content all` for everything.
3. Inspect full files only when the returned chunk does not give enough context.
4. Optionally use `mcp__semble__find_related` with a promising result's `file_path` and `line` to discover related implementations.
5. Use Grep/Glob/Read only when you need exhaustive literal matches or quick confirmation of an exact string.
{SEMBLE_END}
"""


@dataclass(frozen=True)
class McpConfig:
    """MCP integration config for one agent."""

    path: Path | PathResolver
    key: str
    entry: dict[str, object]
    format: Literal["json", "toml"] = "json"

    def resolved_path(self) -> Path:
        """Return the resolved config path."""
        return self.path() if callable(self.path) else self.path


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
        return self.mcp.resolved_path() if self.mcp else None


@dataclass(frozen=True)
class _Integration:
    """Descriptor for one installer integration (MCP server, instructions, sub-agent)."""

    id: str
    label: str
    desc: str
    apply: Callable[[AgentTarget, Mode], WriteResult | None]
    plan_path: Callable[[AgentTarget], Path | None]


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
        mcp=McpConfig(_HOME / ".claude.json", "mcpServers", _STDIO_ENTRY),
        instructions_path=_HOME / ".claude" / "CLAUDE.md",
        subagent_path=_HOME / ".claude" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="cursor",
        display_name="Cursor",
        binary="cursor",
        config_dir=_HOME / ".cursor",
        mcp=McpConfig(_HOME / ".cursor" / "mcp.json", "mcpServers", _STDIO_ENTRY),
        instructions_path=None,  # Cursor instructions are project-local .mdc files
        subagent_path=_HOME / ".cursor" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="gemini",
        display_name="Gemini CLI",
        binary="gemini",
        config_dir=_HOME / ".gemini",
        mcp=McpConfig(_HOME / ".gemini" / "settings.json", "mcpServers", _STDIO_ENTRY),
        instructions_path=_HOME / ".gemini" / "GEMINI.md",
        subagent_path=_HOME / ".gemini" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="kiro",
        display_name="Kiro",
        binary="kiro",
        config_dir=_HOME / ".kiro",
        mcp=McpConfig(_HOME / ".kiro" / "settings" / "mcp.json", "mcpServers", _STDIO_ENTRY),
        instructions_path=_HOME / ".kiro" / "steering" / "semble.md",
        subagent_path=_HOME / ".kiro" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="opencode",
        display_name="Opencode",
        binary="opencode",
        config_dir=_HOME / ".config" / "opencode",
        mcp=McpConfig(_opencode_mcp_path, "mcp", _OPENCODE_ENTRY),
        instructions_path=_HOME / ".config" / "opencode" / "AGENTS.md",
        subagent_path=_HOME / ".config" / "opencode" / "agents" / "semble-search.md",
    ),
    AgentTarget(
        id="copilot",
        display_name="GitHub Copilot",
        binary=None,
        config_dir=_HOME / ".config" / "github-copilot",
        mcp=McpConfig(_HOME / ".copilot" / "mcp-config.json", "mcpServers", _BARE_STDIO_ENTRY),
        instructions_path=None,
        subagent_path=_HOME / ".copilot" / "agents" / "semble-search.agent.md",
    ),
    AgentTarget(
        id="codex",
        display_name="Codex",
        binary="codex",
        config_dir=_HOME / ".codex",
        mcp=McpConfig(_HOME / ".codex" / "config.toml", "mcp_servers", _STDIO_ENTRY, format="toml"),
        instructions_path=_HOME / ".codex" / "AGENTS.md",
    ),
    AgentTarget(
        id="vscode",
        display_name="VS Code",
        binary="code",
        config_dir=None,
        mcp=McpConfig(_vscode_mcp_path, "servers", _STDIO_ENTRY),
        instructions_path=None,
    ),
    AgentTarget(
        id="windsurf",
        display_name="Windsurf",
        binary="windsurf",
        config_dir=_HOME / ".codeium" / "windsurf",
        mcp=McpConfig(_HOME / ".codeium" / "windsurf" / "mcp_config.json", "mcpServers", _BARE_STDIO_ENTRY),
        instructions_path=None,
    ),
    AgentTarget(
        id="zed",
        display_name="Zed",
        binary="zed",
        config_dir=_HOME / ".config" / "zed",
        mcp=McpConfig(_HOME / ".config" / "zed" / "settings.json", "context_servers", _ZED_ENTRY),
        instructions_path=None,
    ),
]


def _json5_parser() -> Parser | None:
    """Return a tree-sitter JSON5 parser, downloading the grammar if needed.

    "json5" ships in tree-sitter-language-pack but isn't in its typed language list, hence the cast.
    """
    try:
        download(["json5"])
        return get_parser(cast(SupportedLanguage, "json5"))
    except Exception:
        return None


def _json5_object(text: str) -> JsonObjectResult:
    """Parse text as JSON5; return (object node, source bytes), "skipped" if grammar unavailable, or "error" if unparseable."""
    parser = _json5_parser()
    if parser is None:
        return "skipped"
    src = text.encode("utf-8")
    root = parser.parse(src).root_node
    if root.has_error:
        return "error"
    objects = [c for c in root.named_children if c.type == "object"]
    return (objects[0], src) if objects else "error"


def _member(obj: Node, src: bytes, key: str) -> Node | None:
    """Return the member of object `obj` whose key equals `key`, or None."""
    for node in obj.named_children:
        if node.type != "member":
            continue
        children = [c for c in node.named_children if c.type != "comment"]
        if children and src[children[0].start_byte : children[0].end_byte].decode("utf-8").strip("\"'") == key:
            return node
    return None


def _value_of(member: Node) -> Node:
    """Return a member's value node (its last non-comment child)."""
    return [c for c in member.named_children if c.type != "comment"][1]


def _insert_first_member(src: bytes, obj: Node, member_text: str) -> bytes:
    """Insert member_text as the first member of object `obj`, indented one level past its brace."""
    brace = obj.start_byte  # the '{'
    line_start = src.rfind(b"\n", 0, brace) + 1
    indent = b" " * (len(src[line_start:brace]) - len(src[line_start:brace].lstrip()) + 2)
    comma = b"," if obj.named_children else b""
    return src[: brace + 1] + b"\n" + indent + member_text.encode("utf-8") + comma + src[brace + 1 :]


def _delete_member(src: bytes, member: Node) -> bytes:
    """Remove `member` plus one adjacent comma and its leading line indentation."""
    start, end = member.start_byte, member.end_byte
    after = end
    while after < len(src) and src[after : after + 1] in (b" ", b"\t"):
        after += 1
    if after < len(src) and src[after : after + 1] == b",":  # prefer a trailing comma
        end = after + 1
    else:
        before = start
        while before > 0 and src[before - 1 : before] in (b" ", b"\t"):
            before -= 1
        if before > 0 and src[before - 1 : before] == b"\n":
            before -= 1  # step over newline to find comma on preceding line
            while before > 0 and src[before - 1 : before] in (b" ", b"\t"):
                before -= 1
        if before > 0 and src[before - 1 : before] == b",":
            start = before - 1
    while start > 0 and src[start - 1 : start] in (b" ", b"\t"):
        start -= 1
    if start > 0 and src[start - 1 : start] == b"\n":
        start -= 1  # drop the now-empty line
    return src[:start] + src[end:]


def _reparse_ok(text: str) -> bool:
    """True if text still parses as error-free JSON5 — the guard run before every write."""
    parser = _json5_parser()
    return parser is not None and not parser.parse(text.encode("utf-8")).root_node.has_error


def merge_json_member(path: Path, section_key: str, member_key: str, value: dict[str, object]) -> Action:
    """Add or update `section_key.member_key = value` in a JSON5 config file, preserving comments and formatting."""
    existed = path.exists()
    text = path.read_text(encoding="utf-8") if existed else ""

    if not text.strip():  # missing or empty: write a clean fresh file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({section_key: {member_key: value}}, indent=2) + "\n", encoding="utf-8")
        return "updated" if existed else "created"

    located = _json5_object(text)
    if not isinstance(located, tuple):
        return located
    obj, src = located
    section_key_json = json.dumps(section_key)
    member_key_json = json.dumps(member_key)
    value_json = json.dumps(value)

    section = _member(obj, src, section_key)
    if section is None:
        new_src = _insert_first_member(src, obj, f"{section_key_json}: {{{member_key_json}: {value_json}}}")
    elif _value_of(section).type != "object":
        return "error"
    elif (existing := _member(_value_of(section), src, member_key)) is not None:
        val_node = _value_of(existing)
        new_src = src[: val_node.start_byte] + value_json.encode("utf-8") + src[val_node.end_byte :]
    else:
        new_src = _insert_first_member(src, _value_of(section), f"{member_key_json}: {value_json}")

    new_text = new_src.decode("utf-8")
    if new_text == text:
        return "unchanged"
    if not _reparse_ok(new_text):
        return "error"
    path.write_text(new_text, encoding="utf-8")
    return "updated" if existed else "created"


def remove_json_member(path: Path, section_key: str, member_key: str) -> Action:
    """Remove `section_key.member_key` from a JSON5 config file, leaving everything else intact."""
    if not path.exists():
        return "not-found"

    located = _json5_object(path.read_text(encoding="utf-8"))
    if not isinstance(located, tuple):
        return located
    obj, src = located

    section = _member(obj, src, section_key)
    if section is None or _value_of(section).type != "object":
        return "not-found"
    member = _member(_value_of(section), src, member_key)
    if member is None:
        return "not-found"

    new_text = _delete_member(src, member).decode("utf-8")
    if not _reparse_ok(new_text):
        return "error"
    path.write_text(new_text, encoding="utf-8")
    return "removed"


def merge_mcp(agent: AgentTarget) -> WriteResult:
    """Add the semble MCP entry to the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.resolved_path()
    return WriteResult(path, merge_json_member(path, agent.mcp.key, "semble", agent.mcp.entry))


def remove_mcp(agent: AgentTarget) -> WriteResult:
    """Remove the semble MCP entry from the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.resolved_path()
    return WriteResult(path, remove_json_member(path, agent.mcp.key, "semble"))


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

    if updated.strip():
        path.write_text(updated, encoding="utf-8")
    else:
        path.unlink()
    return "removed"


def _strip_toml_section(text: str, header: str) -> str:
    """Drop the TOML table beginning at `header` (a [table] line) up to the next table or EOF."""
    result, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.split("#")[0].strip() == header:
            skipping = True
            continue
        if skipping and not stripped.startswith("["):
            continue
        skipping = False
        result.append(line)
    return "".join(result)


def _merge_toml_block(path: Path) -> Action:
    """Add (or refresh) the semble [mcp_servers.semble] table in a Codex config.toml as text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    existing = path.read_text(encoding="utf-8") if existed else ""
    if _CODEX_MCP_BLOCK in existing:
        return "unchanged"
    base = _strip_toml_section(existing, _CODEX_MCP_HEADER).rstrip("\n")
    path.write_text((base + "\n\n" if base else "") + _CODEX_MCP_BLOCK, encoding="utf-8")
    return "created" if not existed else "updated"


def _remove_toml_block(path: Path) -> Action:
    """Remove the semble [mcp_servers.semble] table from a Codex config.toml, leaving the rest."""
    if not path.exists():
        return "not-found"
    existing = path.read_text(encoding="utf-8")
    if _CODEX_MCP_HEADER not in existing:
        return "not-found"
    remaining = _strip_toml_section(existing, _CODEX_MCP_HEADER).strip("\n")
    if remaining:
        path.write_text(remaining + "\n", encoding="utf-8")
    else:
        path.unlink()
    return "removed"


def _apply_mcp(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the MCP server integration for one agent."""
    if agent.mcp is None:
        return None
    path = agent.mcp.resolved_path()
    if agent.mcp.format == "toml":
        return WriteResult(path, _merge_toml_block(path) if mode == "install" else _remove_toml_block(path))
    return merge_mcp(agent) if mode == "install" else remove_mcp(agent)


def _apply_instructions(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the instructions block integration for one agent."""
    path = agent.instructions_path
    if path is None:
        return None
    action = replace_or_append_marked(path, _INSTRUCTIONS) if mode == "install" else remove_marked(path)
    return WriteResult(path, action)


def _apply_subagent(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the global sub-agent file for one agent."""
    dest = agent.subagent_path
    if dest is None:
        return None
    if mode == "uninstall":
        if not dest.exists():
            return WriteResult(dest, "not-found")
        dest.unlink()
        return WriteResult(dest, "removed")
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.write_text(files("semble").joinpath(f"agents/{agent.id}.md").read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        return WriteResult(dest, "error")
    return WriteResult(dest, "updated" if existed else "created")


_INTEGRATIONS: list[_Integration] = [
    _Integration(
        "mcp", "MCP server", "registers semble as a tool in the agent", _apply_mcp, AgentTarget.resolved_mcp_path
    ),
    _Integration(
        "instructions",
        "Instructions",
        "adds usage guide to the agent's config file",
        _apply_instructions,
        lambda a: a.instructions_path,
    ),
    _Integration(
        "subagent",
        "Sub-agent",
        "installs a global semble-search sub-agent (available in all projects)",
        _apply_subagent,
        lambda a: a.subagent_path,
    ),
]


def is_detected(agent: AgentTarget) -> bool:
    """Return True if the agent appears to be installed."""
    if agent.binary and shutil.which(agent.binary):
        return True
    return bool(agent.config_dir and agent.config_dir.exists())


def _tick(ok: bool) -> str:
    """Return a green ✓ or dim – for use in apply output."""
    return f"{_GREEN}✓{_RESET}" if ok else f"{_DIM}–{_RESET}"


def _exit(message: str) -> NoReturn:
    """Print message and exit with code 0."""
    print(message)
    sys.exit(0)


def _checkbox(prompt: str, items: Sequence[tuple[str, _T, bool]]) -> list[_T] | None:
    """Show an interactive multi-select checkbox; return selected values or None if cancelled."""
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
    """Print what will be written or removed for each selected agent and integration."""
    print(f"\n  {_BOLD}Plan:{_RESET}\n")
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            path = integ.plan_path(agent)
            ok = path is not None
            print(f"    {integ.label:<13} {_tick(ok)}  {path if ok else '(not supported)'}")
    print()


def _apply(mode: Mode, agents: list[AgentTarget], integrations: list[_Integration]) -> None:
    """Execute install or uninstall for all chosen agents and integrations, printing results."""
    print()
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            result = integ.apply(agent, mode)
            if result is None:
                print(f"    {_DIM}– {integ.id}: not supported{_RESET}")
                continue
            ok = result.action in ("created", "updated", "removed", "unchanged")
            detail = _ACTION_DETAIL.get(result.action, "")
            suffix = f" — {detail}" if detail else ""
            print(f"    {_tick(ok)} {integ.id} ({result.action}){suffix} → {result.path}")
        print()


def run(mode: Mode) -> None:
    """Interactively install or uninstall semble across coding agents."""
    install = mode == "install"
    print(f"\n  {_BOLD}{'Semble Installer' if install else 'Semble Uninstaller'}{_RESET}\n")

    # Pre-check detected agents on install.
    agent_items = [
        (f"{a.display_name}{'  (detected)' if (detected := is_detected(a)) else ''}", a, detected and install)
        for a in AGENTS
    ]
    chosen_agents = _checkbox(
        f"Select agents to {'configure' if install else 'remove configuration from'}:", agent_items
    ) or _exit("Nothing selected. Exiting.")

    integ_items = [(f"{i.label}  —  {i.desc}", i, True) for i in _INTEGRATIONS]
    chosen_integrations = _checkbox(
        f"Select integrations to {'enable' if install else 'remove'}:", integ_items
    ) or _exit("Nothing selected. Exiting.")

    _print_plan(chosen_agents, chosen_integrations)

    question = "Proceed?" if install else "Remove semble configuration?"
    if not questionary.confirm(question, default=install).ask():
        _exit("Cancelled.")

    _apply(mode, chosen_agents, chosen_integrations)
    footer = " Restart your agents to pick up the changes." if install else ""
    print(f"  {_GREEN}Done!{_RESET}{footer}\n")
