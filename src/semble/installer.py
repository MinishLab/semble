from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Callable, Literal, NoReturn, Sequence, TypeVar, cast

from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, download, get_parser

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

For CLI fallback or sub-agents without MCP access, use:

```bash
semble search "authentication flow" .
semble search "deployment guide" . --content docs
semble find-related src/auth.py 42 .
```

The index is built on first run and cached automatically. If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble`.
{SEMBLE_END}
"""

Action = Literal["created", "updated", "unchanged", "not-found", "removed", "error"]
Mode = Literal["install", "uninstall"]

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
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
    mcp_format: Literal["json", "toml"] = "json"  # "toml" = Codex-style config.toml


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
    AgentTarget(
        id="codex",
        display_name="Codex",
        binary="codex",
        config_dir=_HOME / ".codex",
        mcp_path=_HOME / ".codex" / "config.toml",
        mcp_key="mcp_servers",  # unused for TOML (entry written by _merge_toml_block)
        mcp_entry=_STDIO_ENTRY,  # unused for TOML
        instructions_path=_HOME / ".codex" / "AGENTS.md",
        mcp_format="toml",
    ),
    AgentTarget(
        id="vscode",
        display_name="VS Code",
        binary="code",
        config_dir=None,
        mcp_path=None,  # resolved dynamically via _vscode_mcp_path()
        mcp_key="servers",
        mcp_entry=_STDIO_ENTRY,
        instructions_path=None,
    ),
    AgentTarget(
        id="windsurf",
        display_name="Windsurf",
        binary="windsurf",
        config_dir=_HOME / ".codeium" / "windsurf",
        mcp_path=_HOME / ".codeium" / "windsurf" / "mcp_config.json",
        mcp_key="mcpServers",
        mcp_entry=_BARE_STDIO_ENTRY,
        instructions_path=None,
    ),
    AgentTarget(
        id="zed",
        display_name="Zed",
        binary="zed",
        config_dir=_HOME / ".config" / "zed",
        mcp_path=_HOME / ".config" / "zed" / "settings.json",
        mcp_key="context_servers",
        mcp_entry=_ZED_ENTRY,
        instructions_path=None,
    ),
]


def _mcp_path(agent: AgentTarget) -> Path | None:
    """Resolve the agent's MCP config path, or None if MCP is unsupported."""
    if agent.id == "opencode":
        return _opencode_mcp_path()
    if agent.id == "vscode":
        return _vscode_mcp_path()
    return agent.mcp_path


@lru_cache(maxsize=1)
def _json5_parser() -> Parser | None:
    """Cached tree-sitter JSON5 parser (handles comments + trailing commas), or None if unavailable.

    "json5" ships in tree-sitter-language-pack but isn't in its typed language list, hence the cast.
    """
    try:
        return get_parser(cast(SupportedLanguage, "json5"))
    except Exception:  # grammar not present in this language-pack build
        return None


def _json5_object(text: str) -> tuple[Node, bytes] | None:
    """Parse text as JSON5; return (top-level object node, source bytes), or None if not a clean object."""
    parser = _json5_parser()
    if parser is None:
        return None
    src = text.encode("utf-8")
    root = parser.parse(src).root_node
    if root.has_error:
        return None
    objects = [c for c in root.named_children if c.type == "object"]
    return (objects[0], src) if objects else None


def _member(obj: Node, src: bytes, key: str) -> Node | None:
    """Return the member of object `obj` whose key equals `key`, or None."""
    for m in obj.named_children:
        if m.type != "member":
            continue
        parts = [c for c in m.named_children if c.type != "comment"]
        if parts and src[parts[0].start_byte : parts[0].end_byte].decode("utf-8").strip("\"'") == key:
            return m
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
    j = end
    while j < len(src) and src[j : j + 1] in (b" ", b"\t"):
        j += 1
    if j < len(src) and src[j : j + 1] == b",":  # prefer a trailing comma
        end = j + 1
    else:
        i = start
        while i > 0 and src[i - 1 : i] in (b" ", b"\t"):
            i -= 1
        if i > 0 and src[i - 1 : i] == b",":
            start = i - 1
    while start > 0 and src[start - 1 : start] in (b" ", b"\t"):
        start -= 1
    if start > 0 and src[start - 1 : start] == b"\n":
        start -= 1  # drop the now-empty line
    return src[:start] + src[end:]


def _reparse_ok(text: str) -> bool:
    """True if text still parses as error-free JSON5 — the guard run before every write."""
    parser = _json5_parser()
    return parser is not None and not parser.parse(text.encode("utf-8")).root_node.has_error


def merge_mcp(agent: AgentTarget) -> WriteResult:
    """Add the semble MCP entry to the agent's config, preserving comments and formatting."""
    path = _mcp_path(agent)
    assert path is not None
    existed = path.exists()
    text = path.read_text(encoding="utf-8") if existed else ""

    if not text.strip():  # missing or empty: write a clean fresh file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({agent.mcp_key: {"semble": agent.mcp_entry}}, indent=2) + "\n", encoding="utf-8")
        return WriteResult(path=path, action="updated" if existed else "created")

    located = _json5_object(text)
    if located is None:
        return WriteResult(path=path, action="error")  # don't clobber what we can't parse
    obj, src = located
    entry = json.dumps(agent.mcp_entry)

    section = _member(obj, src, agent.mcp_key)
    if section is None:
        new_src = _insert_first_member(src, obj, f'"{agent.mcp_key}": {{"semble": {entry}}}')
    elif _value_of(section).type != "object":
        return WriteResult(path=path, action="error")
    elif (existing := _member(_value_of(section), src, "semble")) is not None:
        value = _value_of(existing)
        new_src = src[: value.start_byte] + entry.encode("utf-8") + src[value.end_byte :]
    else:
        new_src = _insert_first_member(src, _value_of(section), f'"semble": {entry}')

    new_text = new_src.decode("utf-8")
    if new_text == text:
        return WriteResult(path=path, action="unchanged")
    if not _reparse_ok(new_text):
        return WriteResult(path=path, action="error")
    path.write_text(new_text, encoding="utf-8")
    return WriteResult(path=path, action="updated" if existed else "created")


def remove_mcp(agent: AgentTarget) -> WriteResult:
    """Remove the semble MCP entry from the agent's config, leaving everything else intact."""
    path = _mcp_path(agent)
    assert path is not None
    if not path.exists():
        return WriteResult(path=path, action="not-found")

    located = _json5_object(path.read_text(encoding="utf-8"))
    if located is None:
        return WriteResult(path=path, action="error")
    obj, src = located

    section = _member(obj, src, agent.mcp_key)
    if section is None or _value_of(section).type != "object":
        return WriteResult(path=path, action="not-found")
    semble = _member(_value_of(section), src, "semble")
    if semble is None:
        return WriteResult(path=path, action="not-found")

    new_text = _delete_member(src, semble).decode("utf-8")
    if not _reparse_ok(new_text):
        return WriteResult(path=path, action="error")
    path.write_text(new_text, encoding="utf-8")
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


def _strip_toml_section(text: str, header: str) -> str:
    """Drop the TOML table beginning at `header` (a [table] line) up to the next table or EOF."""
    result, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == header:
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
    path = _mcp_path(agent)
    if path is None:
        return None
    if agent.mcp_format == "toml":
        return WriteResult(path, _merge_toml_block(path) if mode == "install" else _remove_toml_block(path))
    return merge_mcp(agent) if mode == "install" else remove_mcp(agent)


def _apply_instructions(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    p = agent.instructions_path
    if p is None:
        return None
    action = replace_or_append_marked(p, _INSTRUCTIONS) if mode == "install" else remove_marked(p)
    return WriteResult(p, action)


def _apply_subagent(agent: AgentTarget, mode: Mode) -> WriteResult | None:
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
    dest.write_text(files("semble").joinpath(f"agents/{agent.id}.md").read_text(encoding="utf-8"), encoding="utf-8")
    return WriteResult(dest, "updated" if existed else "created")


@dataclass(frozen=True)
class _Integration:
    id: str
    label: str
    desc: str
    apply: Callable[[AgentTarget, Mode], WriteResult | None]
    plan_path: Callable[[AgentTarget], Path | None]


INTEGRATIONS: list[_Integration] = [
    _Integration("mcp", "MCP server", "registers semble as a tool in the agent", _apply_mcp, _mcp_path),
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
    return f"{_GREEN}✓{_RESET}" if ok else f"{_DIM}–{_RESET}"


def _exit(message: str) -> NoReturn:
    print(message)
    sys.exit(0)


_T = TypeVar("_T")


def _checkbox(prompt: str, items: Sequence[tuple[str, _T, bool]]) -> list[_T] | None:
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


def _apply(mode: Mode, agents: list[AgentTarget], integrations: list[_Integration]) -> None:
    print()
    for agent in agents:
        print(f"  {_BOLD}{agent.display_name}{_RESET}")
        for integ in integrations:
            result = integ.apply(agent, mode)
            if result is None:
                print(f"    {_DIM}– {integ.id}: not supported{_RESET}")
                continue
            ok = result.action in ("created", "updated", "removed")
            print(f"    {_tick(ok)} {integ.id} ({result.action}) → {result.path}")
        print()


def _ensure_json5_grammar() -> None:
    """Download the json5 tree-sitter grammar if not already cached."""
    try:
        download(["json5"])
    except Exception as e:
        print(
            f"  {_YELLOW}Warning:{_RESET} Could not download the json5 grammar ({e}).\n"
            f"  Config files that use JSON5/JSONC (e.g. VS Code, OpenCode) will be skipped.\n"
            f"  You can add the MCP entry manually (see semble README).\n"
        )


def run(mode: Mode) -> None:
    """Interactively install or uninstall semble across coding agents."""
    import questionary

    install = mode == "install"
    print(f"\n  {_BOLD}{'Semble Installer' if install else 'Semble Uninstaller'}{_RESET}\n")
    if install:
        _ensure_json5_grammar()

    # Pre-check detected agents on install.
    agent_items = [
        (f"{a.display_name}{'  (detected)' if (d := is_detected(a)) else ''}", a, d and install) for a in AGENTS
    ]
    chosen_agents = _checkbox(
        f"Select agents to {'configure' if install else 'remove configuration from'}:", agent_items
    ) or _exit("Nothing selected. Exiting.")

    integ_items = [(f"{i.label}  —  {i.desc}", i, True) for i in INTEGRATIONS]
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
