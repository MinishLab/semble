"""Tests for the installer module's file-manipulation helpers."""

import json
import sys

import pytest

from semble.installer import (
    _STDIO_ENTRY,
    AGENTS,
    SEMBLE_END,
    SEMBLE_START,
    AgentTarget,
    _read_json,
    merge_mcp,
    remove_marked,
    remove_mcp,
    replace_or_append_marked,
)

_BLOCK = f"{SEMBLE_START}\n## Semble\nsome instructions\n{SEMBLE_END}\n"
_BLOCK_V2 = f"{SEMBLE_START}\n## Semble\nupdated instructions\n{SEMBLE_END}\n"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, {}),  # missing file
        ("", {}),  # empty file
        ('{"mcpServers": {"other": {}}}', {"mcpServers": {"other": {}}}),
        ('{"url": "https://localhost:8080/x"}', {"url": "https://localhost:8080/x"}),  # // inside a string kept
        ('{\n  // a comment\n  "key": "value"\n}\n', {"key": "value"}),  # JSONC line comment
        ('{\n  // c\n  "url": "https://x"\n}\n', {"url": "https://x"}),  # JSONC comment + // inside a string
    ],
)
def test_read_json(tmp_path, content, expected):
    """_read_json parses JSON/JSONC, keeps // inside strings, and returns {} when missing or empty."""
    f = tmp_path / "cfg.json"
    if content is not None:
        f.write_text(content)
    assert _read_json(f) == expected


def test_read_json_raises_on_unparseable(tmp_path):
    """_read_json raises rather than silently returning {} for content it cannot parse."""
    f = tmp_path / "bad.json"
    f.write_text("not json at all {{{{")
    with pytest.raises(ValueError):
        _read_json(f)


@pytest.fixture
def claude_agent(tmp_path):
    """A Claude agent target with MCP/instructions paths rooted in tmp_path."""
    a = next(a for a in AGENTS if a.id == "claude")
    return AgentTarget(
        id=a.id,
        display_name=a.display_name,
        binary=a.binary,
        config_dir=tmp_path / ".claude",
        mcp_path=tmp_path / ".claude.json",
        mcp_key=a.mcp_key,
        mcp_entry=a.mcp_entry,
        instructions_path=tmp_path / ".claude" / "CLAUDE.md",
    )


def test_merge_mcp_creates_file_with_semble(claude_agent):
    """merge_mcp creates the config file and adds the semble entry."""
    assert merge_mcp(claude_agent).path.exists()
    assert json.loads(claude_agent.mcp_path.read_text())["mcpServers"]["semble"] == _STDIO_ENTRY


def test_merge_mcp_preserves_others_and_is_idempotent(claude_agent):
    """merge_mcp keeps existing entries and adds semble once when run repeatedly."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"other": {"command": "foo"}}}))
    merge_mcp(claude_agent)
    merge_mcp(claude_agent)
    servers = json.loads(claude_agent.mcp_path.read_text())["mcpServers"]
    assert servers["semble"] == _STDIO_ENTRY
    assert servers["other"] == {"command": "foo"}


def test_merge_mcp_preserves_unparseable_file(claude_agent):
    """merge_mcp reports an error and leaves an unparseable config untouched instead of clobbering it."""
    original = '{ broken, "url": "https://x"  '
    claude_agent.mcp_path.write_text(original)
    assert merge_mcp(claude_agent).action == "error"
    assert claude_agent.mcp_path.read_text() == original


def test_remove_mcp_deletes_semble_and_drops_empty_section(claude_agent):
    """remove_mcp removes only the semble entry, dropping the section once it is empty."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"semble": _STDIO_ENTRY, "other": {}}}))
    assert remove_mcp(claude_agent).action == "removed"
    assert json.loads(claude_agent.mcp_path.read_text())["mcpServers"] == {"other": {}}

    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"semble": _STDIO_ENTRY}}))
    remove_mcp(claude_agent)
    assert "mcpServers" not in json.loads(claude_agent.mcp_path.read_text())


@pytest.mark.parametrize("setup", [None, {"mcpServers": {"other": {}}}])
def test_remove_mcp_not_found(claude_agent, setup):
    """remove_mcp reports not-found when the file is missing or has no semble entry."""
    if setup is not None:
        claude_agent.mcp_path.write_text(json.dumps(setup))
    assert remove_mcp(claude_agent).action == "not-found"


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


def test_replace_marked_unchanged_when_identical(tmp_path):
    """Re-applying an identical block reports unchanged."""
    f = tmp_path / "CLAUDE.md"
    replace_or_append_marked(f, _BLOCK)
    assert replace_or_append_marked(f, _BLOCK) == "unchanged"


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
