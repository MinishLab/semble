"""Code chunking with tree-sitter AST parsing and line-based fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from semble._types import Chunk, SymbolKind

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}

# tree-sitter node types that represent top-level definitions
_TS_SYMBOL_NODES: dict[str, dict[str, SymbolKind]] = {
    "python": {
        "function_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
    },
    "javascript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
    },
    "typescript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
    },
    "go": {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.CLASS,
    },
    "rust": {
        "function_item": SymbolKind.FUNCTION,
        "impl_item": SymbolKind.CLASS,
        "struct_item": SymbolKind.CLASS,
        "enum_item": SymbolKind.CLASS,
    },
    "java": {
        "method_declaration": SymbolKind.METHOD,
        "class_declaration": SymbolKind.CLASS,
        "interface_declaration": SymbolKind.CLASS,
    },
}


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_symbol_name(node: Any, source_bytes: bytes) -> str | None:
    """Extract the name identifier from a tree-sitter node."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def chunk_with_treesitter(source: str, file_path: str, language: str) -> list[Chunk]:
    """Chunk source code using tree-sitter AST parsing.

    Extracts top-level functions and classes as individual chunks.
    Code between symbols is captured as CHUNK-type entries to avoid gaps.
    Falls back to line-based chunking if tree-sitter is unavailable.
    """
    try:
        import tree_sitter
    except ImportError:
        return chunk_by_lines(source, file_path, language)

    import importlib

    lang_module_name = f"tree_sitter_{language}"
    try:
        lang_module = importlib.import_module(lang_module_name)
        ts_language = tree_sitter.Language(lang_module.language())
    except (ImportError, AttributeError):
        return chunk_by_lines(source, file_path, language)

    parser = tree_sitter.Parser(ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbol_nodes = _TS_SYMBOL_NODES.get(language, {})
    if not symbol_nodes:
        return chunk_by_lines(source, file_path, language)

    chunks: list[Chunk] = []
    lines = source.splitlines(keepends=True)
    covered_ranges: list[tuple[int, int]] = []

    def _walk(node: Any, depth: int = 0) -> None:
        if node.type in symbol_nodes and depth < 3:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            content = "".join(lines[start_line : end_line + 1])
            if content.strip():
                chunks.append(
                    Chunk(
                        content=content,
                        file_path=file_path,
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                        symbol_name=_get_symbol_name(node, source_bytes),
                        symbol_kind=symbol_nodes[node.type],
                        language=language,
                        content_hash=_content_hash(content),
                    )
                )
                covered_ranges.append((start_line, end_line))
        for child in node.children:
            _walk(child, depth + 1)

    _walk(tree.root_node)

    # Capture gaps between symbols (imports, module-level code, etc.)
    if chunks:
        covered_ranges.sort()
        covered_lines: set[int] = set()
        for start, end in covered_ranges:
            covered_lines.update(range(start, end + 1))
        gap_lines = sorted(set(range(len(lines))) - covered_lines)

        if gap_lines:
            groups: list[list[int]] = []
            current_group = [gap_lines[0]]
            for line_num in gap_lines[1:]:
                if line_num == current_group[-1] + 1:
                    current_group.append(line_num)
                else:
                    groups.append(current_group)
                    current_group = [line_num]
            groups.append(current_group)

            for group in groups:
                content = "".join(lines[group[0] : group[-1] + 1])
                if content.strip() and len(content.strip()) > 20:
                    chunks.append(
                        Chunk(
                            content=content,
                            file_path=file_path,
                            start_line=group[0] + 1,
                            end_line=group[-1] + 1,
                            symbol_name=None,
                            symbol_kind=SymbolKind.CHUNK,
                            language=language,
                            content_hash=_content_hash(content),
                        )
                    )

    chunks.sort(key=lambda c: c.start_line)
    return chunks if chunks else chunk_by_lines(source, file_path, language)


def chunk_by_lines(
    source: str,
    file_path: str,
    language: str | None = None,
    max_lines: int = 50,
    overlap_lines: int = 5,
) -> list[Chunk]:
    """Fallback chunker: split by line count with overlap."""
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        content = "".join(lines[start:end])
        if content.strip():
            chunks.append(
                Chunk(
                    content=content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    symbol_name=None,
                    symbol_kind=SymbolKind.CHUNK,
                    language=language,
                    content_hash=_content_hash(content),
                )
            )
        start = end - overlap_lines if end < len(lines) else end

    return chunks


def chunk_file(file_path: Path) -> list[Chunk]:
    """Chunk a single file, choosing the best available strategy."""
    suffix = file_path.suffix.lower()
    language = EXTENSION_MAP.get(suffix)

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not source.strip():
        return []

    if language in _TS_SYMBOL_NODES:
        return chunk_with_treesitter(source, str(file_path), language)

    return chunk_by_lines(source, str(file_path), language)
