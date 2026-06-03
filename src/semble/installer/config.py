from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar, cast

from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, download, get_parser

from semble.installer.agents import SEMBLE_END, SEMBLE_START, Action

JsonObjectResult = tuple[Node, bytes] | Literal["skipped", "error"]
_T = TypeVar("_T")

_CODEX_MCP_HEADER = "[mcp_servers.semble]"
_CODEX_MCP_BLOCK = '[mcp_servers.semble]\ncommand = "uvx"\nargs = ["--from", "semble[mcp]", "semble"]\n'

_json5_parser_cache: Parser | None | bool = False  # False = not yet attempted


def _json5_parser() -> Parser | None:
    """Return a tree-sitter JSON5 parser, downloading the grammar if needed.

    "json5" ships in tree-sitter-language-pack but isn't in its typed language list, hence the cast.
    """
    global _json5_parser_cache
    if _json5_parser_cache is not False:
        return _json5_parser_cache  # type: ignore[return-value]
    try:
        download(["json5"])
        _json5_parser_cache = get_parser(cast(SupportedLanguage, "json5"))
    except Exception:
        _json5_parser_cache = None
    return _json5_parser_cache  # type: ignore[return-value]


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
    """Drop all TOML tables matching `header` or any of its sub-tables, in any order."""
    prefix = header.strip()[1:-1]  # "[mcp_servers.semble]" → "mcp_servers.semble"
    result, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        table_key = stripped.split("#")[0].strip()
        if table_key.startswith("[") and table_key.endswith("]"):
            table_name = table_key[1:-1]
            if table_name == prefix or table_name.startswith(prefix + "."):
                skipping = True
                continue
            skipping = False
        if skipping:
            continue
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
