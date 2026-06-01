"""Tests for the installer module's file-manipulation helpers."""

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# _read_json
# ---------------------------------------------------------------------------


def test_read_json_missing_returns_empty(tmp_path: Path) -> None:
    """A missing file reads as an empty dict."""
    assert _read_json(tmp_path / "missing.json") == {}


def test_read_json_parses_valid(tmp_path: Path) -> None:
    """Valid JSON is parsed into a dict."""
    f = tmp_path / "cfg.json"
    f.write_text('{"mcpServers": {"other": {}}}')
    assert _read_json(f) == {"mcpServers": {"other": {}}}


def test_read_json_strips_line_comments(tmp_path: Path) -> None:
    """JSONC // line comments are stripped before parsing."""
    f = tmp_path / "cfg.jsonc"
    f.write_text('{\n  // a comment\n  "key": "value"\n}\n')
    assert _read_json(f) == {"key": "value"}


def test_read_json_invalid_returns_empty(tmp_path: Path) -> None:
    """Unparseable content reads as an empty dict."""
    f = tmp_path / "bad.json"
    f.write_text("not json at all {{{{")
    assert _read_json(f) == {}


# ---------------------------------------------------------------------------
# merge_mcp / remove_mcp
# ---------------------------------------------------------------------------


@pytest.fixture()
def claude_agent(tmp_path: Path) -> AgentTarget:
    """A Claude agent target with paths rooted in tmp_path."""
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


def test_merge_mcp_creates_new_file(claude_agent: AgentTarget, tmp_path: Path) -> None:
    """Merging into a missing file creates it with the semble entry."""
    result = merge_mcp(claude_agent)
    assert result.path.exists()
    data = json.loads(result.path.read_text())
    assert data["mcpServers"]["semble"] == _STDIO_ENTRY


def test_merge_mcp_preserves_existing_entries(claude_agent: AgentTarget, tmp_path: Path) -> None:
    """Merging keeps existing MCP entries alongside semble."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"other": {"command": "foo"}}}))
    merge_mcp(claude_agent)
    data = json.loads(claude_agent.mcp_path.read_text())
    assert "other" in data["mcpServers"]
    assert "semble" in data["mcpServers"]


def test_merge_mcp_idempotent(claude_agent: AgentTarget) -> None:
    """Merging twice leaves a single semble entry."""
    merge_mcp(claude_agent)
    merge_mcp(claude_agent)
    data = json.loads(claude_agent.mcp_path.read_text())
    assert list(data["mcpServers"].keys()) == ["semble"]


def test_remove_mcp_removes_only_semble(claude_agent: AgentTarget) -> None:
    """Removal deletes the semble entry but leaves others intact."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"semble": _STDIO_ENTRY, "other": {}}}))
    result = remove_mcp(claude_agent)
    assert result.action == "removed"
    data = json.loads(claude_agent.mcp_path.read_text())
    assert "semble" not in data["mcpServers"]
    assert "other" in data["mcpServers"]


def test_remove_mcp_empty_section_cleaned_up(claude_agent: AgentTarget) -> None:
    """Removing the last entry drops the now-empty mcpServers section."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"semble": _STDIO_ENTRY}}))
    remove_mcp(claude_agent)
    data = json.loads(claude_agent.mcp_path.read_text())
    assert "mcpServers" not in data


def test_remove_mcp_missing_file(claude_agent: AgentTarget) -> None:
    """Removing from a missing file reports not-found."""
    result = remove_mcp(claude_agent)
    assert result.action == "not-found"


def test_remove_mcp_no_semble_entry(claude_agent: AgentTarget) -> None:
    """Removing when no semble entry exists reports not-found."""
    claude_agent.mcp_path.write_text(json.dumps({"mcpServers": {"other": {}}}))
    result = remove_mcp(claude_agent)
    assert result.action == "not-found"


# ---------------------------------------------------------------------------
# replace_or_append_marked / remove_marked
# ---------------------------------------------------------------------------


def test_append_marked_to_empty_file(tmp_path: Path) -> None:
    """Appending to a missing file creates it with the marked block."""
    f = tmp_path / "CLAUDE.md"
    action = replace_or_append_marked(f, _BLOCK)
    assert action == "created"
    assert SEMBLE_START in f.read_text()


def test_append_marked_to_existing_file(tmp_path: Path) -> None:
    """Appending preserves existing content and adds the marked block."""
    f = tmp_path / "CLAUDE.md"
    f.write_text("# Existing content\n")
    action = replace_or_append_marked(f, _BLOCK)
    assert action == "updated"
    text = f.read_text()
    assert "# Existing content" in text
    assert SEMBLE_START in text


def test_replace_marked_section(tmp_path: Path) -> None:
    """An existing marked section is replaced in place, leaving surrounding text."""
    f = tmp_path / "CLAUDE.md"
    f.write_text(f"# Before\n\n{_BLOCK}\n# After\n")
    action = replace_or_append_marked(f, _BLOCK_V2)
    assert action == "updated"
    text = f.read_text()
    assert "updated instructions" in text
    assert "some instructions" not in text
    assert "# Before" in text
    assert "# After" in text


def test_replace_marked_unchanged_when_identical(tmp_path: Path) -> None:
    """Re-applying an identical block reports unchanged."""
    f = tmp_path / "CLAUDE.md"
    replace_or_append_marked(f, _BLOCK)
    action = replace_or_append_marked(f, _BLOCK)
    assert action == "unchanged"


def test_remove_marked_removes_section(tmp_path: Path) -> None:
    """Removing the marked section leaves surrounding text intact."""
    f = tmp_path / "CLAUDE.md"
    f.write_text(f"# Before\n\n{_BLOCK}\n# After\n")
    action = remove_marked(f)
    assert action == "removed"
    text = f.read_text()
    assert SEMBLE_START not in text
    assert "# Before" in text
    assert "# After" in text


def test_remove_marked_deletes_file_when_only_content(tmp_path: Path) -> None:
    """A file containing only the marked block is deleted on removal."""
    f = tmp_path / "CLAUDE.md"
    replace_or_append_marked(f, _BLOCK)
    remove_marked(f)
    assert not f.exists()


def test_remove_marked_missing_file(tmp_path: Path) -> None:
    """Removing from a missing file reports not-found."""
    action = remove_marked(tmp_path / "missing.md")
    assert action == "not-found"


def test_remove_marked_no_markers(tmp_path: Path) -> None:
    """Removing from a file without markers reports not-found."""
    f = tmp_path / "CLAUDE.md"
    f.write_text("# No semble section here\n")
    action = remove_marked(f)
    assert action == "not-found"
