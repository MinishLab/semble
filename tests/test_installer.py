import json
import sys
from dataclasses import replace

import pytest

from semble.installer import run
from semble.installer.agents import (
    _STDIO_SERVER_CONFIG,
    AGENTS,
    SEMBLE_END,
    SEMBLE_START,
    _opencode_mcp_path,
    _vscode_mcp_path,
    is_detected,
)
from semble.installer.config import (
    _CODEX_MCP_HEADER,
    merge_toml_block,
    remove_marked,
    remove_toml_block,
    replace_or_append_marked,
)
from semble.installer.installer import (
    _INTEGRATIONS,
    _apply_instructions,
    _apply_mcp,
    _apply_subagent,
    _checkbox,
    _print_plan,
    merge_mcp,
    remove_mcp,
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
        mcp=replace(a.mcp, path=tmp_path / ".claude.json"),
        instructions_path=tmp_path / ".claude" / "CLAUDE.md",
        subagent_path=tmp_path / ".claude" / "agents" / "semble-search.md",
    )


@pytest.fixture
def run_setup(monkeypatch, tmp_path, claude_agent):
    """Patches AGENTS with claude+cursor in tmp_path; stubs _checkbox to select all."""
    cursor = next(a for a in AGENTS if a.id == "cursor")
    cursor = replace(
        cursor,
        mcp=replace(cursor.mcp, path=tmp_path / "cursor.json"),
        subagent_path=tmp_path / "cursor_subagent.md",
    )
    monkeypatch.setattr("semble.installer.installer.AGENTS", [claude_agent, cursor])
    monkeypatch.setattr("semble.installer.installer._checkbox", lambda _p, items: [v for _, v, _ in items])
    return monkeypatch


def test_merge_mcp_creates_fresh_file(claude_agent):
    """merge_mcp writes a clean new config file when none exists."""
    assert merge_mcp(claude_agent).action == "created"
    data = json.loads(claude_agent.mcp.path.read_text())
    assert data["mcpServers"]["semble"] == _STDIO_SERVER_CONFIG


def test_merge_mcp_preserves_comments_and_other_entries(claude_agent):
    """merge_mcp adds semble while leaving existing comments and entries byte-intact."""
    claude_agent.mcp.path.write_text('{\n  // my servers\n  "mcpServers": {\n    "other": {"command": "x"}\n  }\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    text = claude_agent.mcp.path.read_text()
    assert "// my servers" in text  # comment preserved
    assert '"other"' in text  # existing entry preserved
    assert '"semble"' in text  # semble added


def test_merge_mcp_adds_section_when_absent(claude_agent):
    """merge_mcp creates the mcpServers section if missing, keeping other keys and comments."""
    claude_agent.mcp.path.write_text('{\n  // keep me\n  "theme": "dark"\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    text = claude_agent.mcp.path.read_text()
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
    claude_agent.mcp.path.write_text(initial)
    assert merge_mcp(claude_agent).action == "updated"
    json.loads(claude_agent.mcp.path.read_text())  # raises if invalid


def test_merge_mcp_idempotent(claude_agent):
    """Running merge twice adds semble once and reports unchanged the second time."""
    claude_agent.mcp.path.write_text('{\n  "mcpServers": {}\n}\n')
    assert merge_mcp(claude_agent).action == "updated"
    assert merge_mcp(claude_agent).action == "unchanged"
    assert claude_agent.mcp.path.read_text().count('"semble":') == 1  # the member key, once


@pytest.mark.parametrize(
    "content",
    [
        "this is not json {{{{ ",
        '{ "mcpServers": "not-an-object" }',
    ],
)
def test_merge_mcp_errors(claude_agent, content):
    """merge_mcp reports error for an unparseable or structurally invalid config."""
    claude_agent.mcp.path.write_text(content)
    assert merge_mcp(claude_agent).action == "error"


@pytest.mark.parametrize(
    ("agent_id", "key"),
    [
        ("zed", "context_servers"),
        ("windsurf", "mcpServers"),
        ("copilot", "mcpServers"),
        ("reasonix", "mcpServers"),
        ("pi", "mcpServers"),
        ("commandcode", "mcpServers"),
        ("antigravity", "mcpServers"),
    ],
)
def test_merge_mcp_writes_under_agent_key(tmp_path, agent_id, key):
    """merge_mcp writes the semble entry under each agent's own top-level MCP key."""
    src = next(a for a in AGENTS if a.id == agent_id)
    agent = replace(src, mcp=replace(src.mcp, path=tmp_path / "cfg.json"))
    merge_mcp(agent)
    assert "semble" in json.loads((tmp_path / "cfg.json").read_text())[key]


def test_mcp_skipped_when_grammar_unavailable(claude_agent, monkeypatch):
    """When the JSON5 grammar cannot be downloaded, merge/remove return 'skipped'."""
    claude_agent.mcp.path.write_text('{ "mcpServers": {} }')
    monkeypatch.setattr("semble.installer.config.download", lambda _: 1 / 0)
    monkeypatch.setattr("semble.installer.config._json5_parser_cache", False)
    assert merge_mcp(claude_agent).action == "skipped"
    assert remove_mcp(claude_agent).action == "skipped"


def test_merge_mcp_reparse_guard(claude_agent, monkeypatch):
    """merge_mcp reports error when the edited JSON5 fails reparse validation."""
    claude_agent.mcp.path.write_text('{\n  "mcpServers": {}\n}\n')
    monkeypatch.setattr("semble.installer.config._reparse_ok", lambda _: False)
    assert merge_mcp(claude_agent).action == "error"


def test_remove_mcp_preserves_comments(claude_agent):
    """remove_mcp deletes only semble, keeping comments and sibling entries intact."""
    claude_agent.mcp.path.write_text(
        "{\n"
        "  // my servers\n"
        '  "mcpServers": {\n'
        '    "semble": {"command": "uvx"},\n'
        '    "other": {"command": "x"}\n'
        "  }\n"
        "}\n"
    )
    assert remove_mcp(claude_agent).action == "removed"
    text = claude_agent.mcp.path.read_text()
    assert "// my servers" in text
    assert '"other"' in text
    assert '"semble"' not in text


@pytest.mark.parametrize(
    "initial",
    [
        '{\n  "mcpServers": {\n    "other": {},\n    "semble": {}\n  }\n}\n',
        '{\n  "mcpServers": {\n    "semble": {}  ,\n    "other": {}\n  }\n}\n',
        '{\n  "mcpServers": {\n    "other": {},  \n    "semble": {}\n  }\n}\n',
    ],
)
def test_remove_mcp_no_trailing_comma(claude_agent, initial):
    """Removing semble must not leave a trailing comma regardless of its position or whitespace."""
    claude_agent.mcp.path.write_text(initial)
    assert remove_mcp(claude_agent).action == "removed"
    json.loads(claude_agent.mcp.path.read_text())  # raises if trailing comma or invalid


def test_remove_mcp_reparse_guard(claude_agent, monkeypatch):
    """remove_mcp reports error when the result fails reparse validation."""
    claude_agent.mcp.path.write_text('{\n  "mcpServers": {\n    "semble": {}\n  }\n}\n')
    monkeypatch.setattr("semble.installer.config._reparse_ok", lambda _: False)
    assert remove_mcp(claude_agent).action == "error"


@pytest.mark.parametrize(
    "setup",
    [None, '{\n  "mcpServers": {"other": {}}\n}\n', '{"other": "stuff"}'],
)
def test_remove_mcp_not_found(claude_agent, setup):
    """remove_mcp reports not-found when the file is missing, has no semble entry, or no mcpServers key."""
    if setup is not None:
        claude_agent.mcp.path.write_text(setup)
    assert remove_mcp(claude_agent).action == "not-found"


def test_codex_toml_merge_and_remove(tmp_path):
    """The Codex TOML helpers add/remove [mcp_servers.semble] while preserving other tables and keys."""
    f = tmp_path / "config.toml"
    f.write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n')
    assert merge_toml_block(f) == "updated"
    text = f.read_text()
    assert _CODEX_MCP_HEADER in text
    assert 'model = "gpt-5"' in text
    assert "[mcp_servers.other]" in text
    assert merge_toml_block(f) == "unchanged"  # idempotent

    assert remove_toml_block(f) == "removed"
    text = f.read_text()
    assert _CODEX_MCP_HEADER not in text
    assert "[mcp_servers.other]" in text  # only the semble table is removed


def test_codex_toml_merge_replaces_section_with_inline_comment(tmp_path):
    """_merge_toml_block replaces an existing semble table even when the header has a trailing comment."""
    f = tmp_path / "config.toml"
    f.write_text('[mcp_servers.semble] # added manually\ncommand = "old"\n')
    assert merge_toml_block(f) == "updated"
    text = f.read_text()
    assert text.count("[mcp_servers.semble]") == 1


@pytest.mark.parametrize(
    ("setup", "expected"),
    [(None, "not-found"), ("model = 'gpt-5'\n", "not-found")],
)
def test_remove_toml_not_found(tmp_path, setup, expected):
    """_remove_toml_block reports not-found when the file is absent or has no semble header."""
    f = tmp_path / "config.toml"
    if setup is not None:
        f.write_text(setup)
    assert remove_toml_block(f) == expected


def test_remove_toml_deletes_file_when_only_semble(tmp_path):
    """_remove_toml_block unlinks the file when removing semble leaves it empty."""
    f = tmp_path / "config.toml"
    merge_toml_block(f)
    remove_toml_block(f)
    assert not f.exists()


_SUB_AFTER = (
    '[mcp_servers.semble]\ncommand = "uvx"\n\n'
    '[mcp_servers.semble.tools.search]\napproval_mode = "approve"\n\n'
    '[other]\nkey = "val"\n'
)
_SUB_BEFORE = (
    '[mcp_servers.semble.tools.search]\napproval_mode = "approve"\n\n'
    '[mcp_servers.semble]\ncommand = "uvx"\n\n'
    '[other]\nkey = "val"\n'
)


@pytest.mark.parametrize("content", [_SUB_AFTER, _SUB_BEFORE])
def test_remove_toml_strips_sub_tables(tmp_path, content):
    """_remove_toml_block removes sub-tables like [mcp_servers.semble.tools.search], before or after the main header."""
    f = tmp_path / "config.toml"
    f.write_text(content)
    assert remove_toml_block(f) == "removed"
    text = f.read_text()
    assert "[mcp_servers.semble]" not in text
    assert "[mcp_servers.semble.tools.search]" not in text
    assert "[other]" in text


@pytest.mark.parametrize(
    ("platform", "env_vars"),
    [
        ("darwin", {}),
        ("win32", {"APPDATA": "/appdata"}),
        ("linux", {"XDG_CONFIG_HOME": "/xdg"}),
    ],
)
def test_vscode_mcp_path(monkeypatch, platform, env_vars):
    """_vscode_mcp_path returns a Code/User/mcp.json path for each supported OS."""
    monkeypatch.setattr("sys.platform", platform)
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    p = _vscode_mcp_path()
    assert p.name == "mcp.json"
    assert "Code" in str(p)


def test_opencode_mcp_path(monkeypatch, tmp_path):
    """_opencode_mcp_path respects XDG_CONFIG_HOME and prefers .jsonc over .json."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert _opencode_mcp_path().parent == tmp_path / "opencode"
    assert _opencode_mcp_path().name == "opencode.jsonc"  # fallback when neither exists

    (tmp_path / "opencode").mkdir()
    json_ = tmp_path / "opencode" / "opencode.json"
    json_.touch()
    assert _opencode_mcp_path() == json_  # json when no jsonc

    jsonc = tmp_path / "opencode" / "opencode.jsonc"
    jsonc.touch()
    assert _opencode_mcp_path() == jsonc  # jsonc preferred


def test_apply_mcp(tmp_path):
    """_apply_mcp returns None for mcp=None agents and uses the TOML path for codex."""
    no_mcp = replace(next(a for a in AGENTS if a.id == "claude"), mcp=None)
    assert _apply_mcp(no_mcp, "install") is None

    codex = next(a for a in AGENTS if a.id == "codex")
    codex = replace(codex, mcp=replace(codex.mcp, path=tmp_path / "config.toml"))
    assert _apply_mcp(codex, "install").action in ("created", "updated")


def test_apply_instructions_none():
    """_apply_instructions returns None for agents with no instructions_path."""
    cursor = next(a for a in AGENTS if a.id == "cursor")
    assert _apply_instructions(cursor, "install") is None


def test_apply_subagent(tmp_path):
    """_apply_subagent installs/uninstalls the sub-agent file; returns error for missing resource."""
    dest = tmp_path / "agents" / "semble-search.md"
    agent = replace(next(a for a in AGENTS if a.id == "claude"), subagent_path=dest)

    assert _apply_subagent(agent, "install").action == "created"
    assert dest.exists()
    assert _apply_subagent(agent, "install").action == "updated"
    assert _apply_subagent(agent, "uninstall").action == "removed"
    assert not dest.exists()
    assert _apply_subagent(agent, "uninstall").action == "not-found"
    assert _apply_subagent(replace(agent, subagent_path=None), "install") is None
    assert _apply_subagent(replace(agent, id="zzz"), "install").action == "error"


def test_apply_subagent_codex_toml(tmp_path):
    """_apply_subagent writes a .toml file for codex (deriving source extension from destination)."""
    dest = tmp_path / "agents" / "semble-search.toml"
    codex = next(a for a in AGENTS if a.id == "codex")
    agent = replace(codex, subagent_path=dest)

    assert _apply_subagent(agent, "install").action == "created"
    assert dest.exists()
    text = dest.read_text()
    assert 'name = "semble_search"' in text
    assert "developer_instructions" in text
    assert _apply_subagent(agent, "uninstall").action == "removed"
    assert not dest.exists()


def test_is_detected(monkeypatch, tmp_path):
    """is_detected returns True when binary is on PATH or config dir exists."""
    agent = next(a for a in AGENTS if a.id == "claude")
    monkeypatch.setattr("semble.installer.agents.shutil.which", lambda _: "/usr/bin/claude")
    assert is_detected(agent)

    agent_no_bin = replace(agent, binary=None, config_dir=tmp_path)
    assert is_detected(agent_no_bin)


def test_checkbox(monkeypatch):
    """_checkbox wraps questionary.checkbox and returns the selected values."""

    class _Fake:
        def ask(self):
            return ["a"]

    monkeypatch.setattr("semble.installer.installer.questionary.checkbox", lambda *_, **__: _Fake())
    assert _checkbox("Pick:", [("Option A", "a", False)]) == ["a"]


def test_print_plan(capsys, claude_agent):
    """_print_plan prints each agent, integration, and resolved path (or 'not supported')."""
    no_mcp = replace(claude_agent, display_name="No MCP Agent", mcp=None)
    _print_plan([claude_agent, no_mcp], _INTEGRATIONS)
    out = capsys.readouterr().out
    assert "Claude Code" in out
    assert "not supported" in out  # no_mcp has no MCP


def test_run_completes(run_setup, monkeypatch, capsys):
    """run('install') completes a full interactive install and prints Done."""

    class _Yes:
        def ask(self):
            return True

    monkeypatch.setattr("semble.installer.installer.questionary.confirm", lambda *_, **__: _Yes())
    run("install")
    assert "Done!" in capsys.readouterr().out


def test_run_cancels(run_setup, monkeypatch):
    """Run exits when the user declines the confirmation prompt."""

    class _No:
        def ask(self):
            return False

    monkeypatch.setattr("semble.installer.installer.questionary.confirm", lambda *_, **__: _No())
    with pytest.raises(SystemExit):
        run("install")


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
