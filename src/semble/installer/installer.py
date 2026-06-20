from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, NoReturn, Sequence, TypeVar

import questionary

from semble.installer.agents import (
    AGENTS,
    INSTRUCTIONS,
    AgentTarget,
    Mode,
    WriteResult,
    is_detected,
)
from semble.installer.config import (
    merge_json_member,
    merge_toml_block,
    remove_json_member,
    remove_marked,
    remove_toml_block,
    replace_or_append_marked,
)

_T = TypeVar("_T")

_GREEN = "\033[32m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_ACTION_DETAIL: dict[str, str] = {
    "skipped": "JSON5 grammar unavailable — add manually (see README)",
    "error": "could not parse or edit config",
}


@dataclass(frozen=True)
class _Integration:
    """Descriptor for one installer integration (MCP server, instructions, sub-agent)."""

    id: str
    label: str
    desc: str
    apply: Callable[[AgentTarget, Mode], WriteResult | None]
    plan_path: Callable[[AgentTarget], Path | None]


def merge_mcp(agent: AgentTarget) -> WriteResult:
    """Add the semble MCP entry to the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.path
    return WriteResult(path, merge_json_member(path, agent.mcp.key, "semble", agent.mcp.entry))


def remove_mcp(agent: AgentTarget) -> WriteResult:
    """Remove the semble MCP entry from the agent's config."""
    assert agent.mcp is not None
    path = agent.mcp.path
    return WriteResult(path, remove_json_member(path, agent.mcp.key, "semble"))


def _apply_mcp(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the MCP server integration for one agent."""
    if agent.mcp is None:
        return None
    path = agent.mcp.path
    if agent.mcp.format == "toml":
        return WriteResult(path, merge_toml_block(path) if mode == "install" else remove_toml_block(path))
    return merge_mcp(agent) if mode == "install" else remove_mcp(agent)


def _apply_instructions(agent: AgentTarget, mode: Mode) -> WriteResult | None:
    """Apply or remove the instructions block integration for one agent."""
    path = agent.instructions_path
    if path is None:
        return None
    action = replace_or_append_marked(path, INSTRUCTIONS) if mode == "install" else remove_marked(path)
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
        src = files("semble").joinpath(f"agents/{agent.id}{dest.suffix}").read_text(encoding="utf-8")
        dest.write_text(src, encoding="utf-8")
    except Exception:
        return WriteResult(dest, "error")
    return WriteResult(dest, "updated" if existed else "created")


_INTEGRATIONS: list[_Integration] = [
    _Integration(
        "mcp", "MCP server", "lets the agent call semble directly as a tool", _apply_mcp, AgentTarget.resolved_mcp_path
    ),
    _Integration(
        "instructions",
        "Instructions",
        "adds CLI usage guidance to AGENTS.md / CLAUDE.md",
        _apply_instructions,
        lambda a: a.instructions_path,
    ),
    _Integration(
        "subagent",
        "Sub-agent",
        "installs a dedicated semble-search sub-agent",
        _apply_subagent,
        lambda a: a.subagent_path,
    ),
]


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

    agent_items = [
        (f"{a.display_name}{'  (detected)' if (detected := is_detected(a)) else ''}", a, detected and install)
        for a in sorted(AGENTS, key=lambda a: not is_detected(a))
    ]
    chosen_agents = _checkbox(
        f"Select agents to {'configure' if install else 'remove configuration from'}:", agent_items
    ) or _exit("Nothing selected. Exiting.")

    max_label = max(len(i.label) for i in _INTEGRATIONS)
    integ_items = [(f"{i.label:<{max_label}}  —  {i.desc}", i, True) for i in _INTEGRATIONS]
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
