"""Tests for the installer module's file-manipulation helpers."""

import json
import sys
from dataclasses import replace

import pytest

from semble.installer import (
    _CODEX_MCP_HEADER,
    _STDIO_ENTRY,
    AGENTS,
    SEMBLE_END,
    SEMBLE_START,
    _merge_toml_block,
    _remove_toml_block,
    _vscode_mcp_path,
    merge_mcp,
    remove_marked,
    remove_mcp,
    replace_or_append_marked,
)

_BLOCK = f"{SEMBLE_START}\n## Semble\nsome instructions\n{SEMBLE_END}\n"
_BLOCK_V2 = f"{SEMBLE_START}\n## Semble\nupdated instructions\n{SEMBLE_END}\n"


@pytest.fixture
def claude_agent(tmp_path):
    """A Claude agent target with MCP/instructions paths rooted in tmp_path."""
    a = next(a for a in AGENTS if a.id == "claude")
    return replace(
        a,
        config_dir=tmp_path / ".claude",
        mcp_path=tmp_path / ".claude.json",
        instructions_path=tmp_path / ".claude" / "CLAUDE.md",
    )


def test_merge_mcp_creates_fresh_file(claude_agent):
    """merge_mcp writes a clean new config file when none exists."""
    assert merge_mcp(claude_agent).action == "created"
    data = json.loads(claude_agent.mcp_path.read_text())
    assert data["mcpServers"]["semble"] == _STDIO_ENTRY


def test_merge_mcp_preserves_comments_and_other_entries(claude_agent):
    """merge_mcp adds semble while leaving existing comments and entries byte-intact."""
    claude_agent.mcp_path.write_text('{\n  // my servers\n  "mcpServers": {\n    "other": {"command": "x"}\n  }\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    text = claude_agent.mcp_path.read_text()
    assert "// my servers" in text  # comment preserved
    assert '"other"' in text  # existing entry preserved
    assert '"semble"' in text  # semble added


def test_merge_mcp_adds_section_when_absent(claude_agent):
    """merge_mcp creates the mcpServers section if missing, keeping other keys and comments."""
    claude_agent.mcp_path.write_text('{\n  // keep me\n  "theme": "dark"\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    text = claude_agent.mcp_path.read_text()
    assert "// keep me" in text
    assert '"theme"' in text
    assert '"mcpServers"' in text
    assert '"semble"' in text


@pytest.mark.parametrize(
    "initial",
    ['{\n  "mcpServers": {}\n}\n', '{"mcpServers": {}}\n', "{}"],
)
def test_merge_mcp_into_empty_object_produces_valid_json(claude_agent, initial):
    """Inserting into an empty strict-JSON object must not produce a trailing comma."""
    claude_agent.mcp_path.write_text(initial)
    assert merge_mcp(claude_agent).action == "updated"
    json.loads(claude_agent.mcp_path.read_text())  # raises if invalid


def test_merge_mcp_idempotent(claude_agent):
    """Running merge twice adds semble once and reports unchanged the second time."""
    claude_agent.mcp_path.write_text('{\n  "mcpServers": {}\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    assert merge_mcp(claude_agent).action == "unchanged"
    assert claude_agent.mcp_path.read_text().count('"semble":') == 1  # the member key, once


def test_merge_mcp_errors_on_malformed_file(claude_agent):
    """merge_mcp reports an error and leaves a genuinely unparseable file untouched."""
    original = "this is not json {{{{ "
    claude_agent.mcp_path.write_text(original)
    assert merge_mcp(claude_agent).action == "error"
    assert claude_agent.mcp_path.read_text() == original


@pytest.mark.parametrize(
    ("agent_id", "key"), [("zed", "context_servers"), ("windsurf", "mcpServers"), ("copilot", "mcpServers")]
)
def test_merge_mcp_writes_under_agent_key(tmp_path, agent_id, key):
    """merge_mcp writes the semble entry under each agent's own top-level MCP key."""
    src = next(a for a in AGENTS if a.id == agent_id)
    agent = replace(src, mcp_path=tmp_path / "cfg.json")
    merge_mcp(agent)
    assert "semble" in json.loads((tmp_path / "cfg.json").read_text())[key]


def test_remove_mcp_preserves_comments(claude_agent):
    """remove_mcp deletes only semble, keeping comments and sibling entries intact."""
    claude_agent.mcp_path.write_text(
        "{\n"
        "  // my servers\n"
        '  "mcpServers": {\n'
        '    "semble": {"command": "uvx"},\n'
        '    "other": {"command": "x"}\n'
        "  }\n"
        "}\n"
    )
    assert remove_mcp(claude_agent).action == "removed"
    text = claude_agent.mcp_path.read_text()
    assert "// my servers" in text
    assert '"other"' in text
    assert '"semble"' not in text


@pytest.mark.parametrize("setup", [None, '{\n  "mcpServers": {"other": {}}\n}\n'])
def test_remove_mcp_not_found(claude_agent, setup):
    """remove_mcp reports not-found when the file is missing or has no semble entry."""
    if setup is not None:
        claude_agent.mcp_path.write_text(setup)
    assert remove_mcp(claude_agent).action == "not-found"


def test_codex_toml_merge_and_remove(tmp_path):
    """The Codex TOML helpers add/remove [mcp_servers.semble] while preserving other tables and keys."""
    f = tmp_path / "config.toml"
    f.write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n')
    assert _merge_toml_block(f) == "updated"
    text = f.read_text()
    assert _CODEX_MCP_HEADER in text
    assert 'model = "gpt-5"' in text
    assert "[mcp_servers.other]" in text
    assert _merge_toml_block(f) == "unchanged"  # idempotent

    assert _remove_toml_block(f) == "removed"
    text = f.read_text()
    assert _CODEX_MCP_HEADER not in text
    assert "[mcp_servers.other]" in text  # only the semble table is removed


def test_vscode_mcp_path_is_user_mcp_json():
    """_vscode_mcp_path resolves to the user-profile mcp.json (…/Code/User/mcp.json)."""
    p = _vscode_mcp_path()
    assert p.name == "mcp.json"
    assert p.parent.name == "User"


@pytest.mark.parametrize(
    ("initial", "block", "expected", "present", "absent"),
    [
        (None, _BLOCK, "created", [SEMBLE_START], []),
        ("# Existing\n", _BLOCK, "updated", ["# Existing", SEMBLE_START], []),
        (
            f"# Before\n\n{_BLOCK}\n# After\n",
            _BLOCK_V2,
            "updated",
            ["updated instructions", "# Before", "# After"],
            ["some instructions"],
        ),
        (_BLOCK, _BLOCK, "unchanged", [SEMBLE_START], []),
    ],
)
def test_replace_or_append_marked(tmp_path, initial, block, expected, present, absent):
    """replace_or_append_marked creates, appends, or replaces the marked block and reports the action."""
    f = tmp_path / "CLAUDE.md"
    if initial is not None:
        f.write_text(initial)
    assert replace_or_append_marked(f, block) == expected
    text = f.read_text()
    assert all(s in text for s in present)
    assert all(s not in text for s in absent)


def test_remove_marked_strips_block_and_deletes_empty_file(tmp_path):
    """remove_marked strips the block (keeping surrounding text), and deletes the file if nothing remains."""
    f = tmp_path / "CLAUDE.md"
    f.write_text(f"# Before\n\n{_BLOCK}\n# After\n")
    assert remove_marked(f) == "removed"
    text = f.read_text()
    assert SEMBLE_START not in text
    assert "# Before" in text
    assert "# After" in text

    f.write_text(_BLOCK)
    remove_marked(f)
    assert not f.exists()


@pytest.mark.parametrize("initial", [None, "# No semble section here\n"])
def test_remove_marked_not_found(tmp_path, initial):
    """remove_marked reports not-found for a missing file or one without markers."""
    f = tmp_path / "CLAUDE.md"
    if initial is not None:
        f.write_text(initial)
    assert remove_marked(f) == "not-found"


@pytest.mark.parametrize("command", ["install", "uninstall"])
def test_cli_dispatches_to_installer_run(monkeypatch, command):
    """`semble install` / `semble uninstall` route to installer.run with the command name."""
    import semble.cli as cli

    calls = []
    monkeypatch.setattr("semble.installer.run", lambda mode: calls.append(mode))
    monkeypatch.setattr(sys, "argv", ["semble", command])
    cli.main()
    assert calls == [command]
