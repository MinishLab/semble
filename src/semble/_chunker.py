"""Code chunking with AST parsing and line-based fallback."""

from __future__ import annotations

import ast
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


_MAX_CHUNK_LINES = 60
_CHUNK_OVERLAP_LINES = 5


def chunk_with_ast(source: str, file_path: str) -> list[Chunk]:
    """Chunk Python source using the built-in ast module.

    Extracts top-level and class-level function/class definitions as individual
    chunks with proper symbol names. Symbols larger than _MAX_CHUNK_LINES are
    split into overlapping line-windows; the first window keeps the symbol name
    so symbol search still finds the definition. Falls back to line-based if
    parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_by_lines(source, file_path, "python")

    lines = source.splitlines(keepends=True)
    chunks: list[Chunk] = []
    covered_lines: set[int] = set()

    _AST_KIND: dict[type, SymbolKind] = {
        ast.FunctionDef: SymbolKind.FUNCTION,
        ast.AsyncFunctionDef: SymbolKind.FUNCTION,
        ast.ClassDef: SymbolKind.CLASS,
    }

    def _add(node: ast.stmt, kind: SymbolKind) -> None:
        start = node.lineno - 1  # 0-based
        end = node.end_lineno or start
        n_lines = end - start
        name: str = node.name  # type: ignore[attr-defined]

        if n_lines <= _MAX_CHUNK_LINES:
            content = "".join(lines[start:end])
            if content.strip():
                chunks.append(
                    Chunk(
                        content=content,
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                        symbol_name=name,
                        symbol_kind=kind,
                        language="python",
                        content_hash=_content_hash(content),
                    )
                )
                covered_lines.update(range(start, end))
        else:
            # Sub-chunk with overlap; first window keeps the symbol name
            first = True
            win_start = start
            while win_start < end:
                win_end = min(win_start + _MAX_CHUNK_LINES, end)
                content = "".join(lines[win_start:win_end])
                if content.strip():
                    chunks.append(
                        Chunk(
                            content=content,
                            file_path=file_path,
                            start_line=win_start + 1,
                            end_line=win_end,
                            symbol_name=name if first else None,
                            symbol_kind=kind if first else SymbolKind.CHUNK,
                            language="python",
                            content_hash=_content_hash(content),
                        )
                    )
                    covered_lines.update(range(win_start, win_end))
                    first = False
                if win_end >= end:
                    break
                win_start = win_end - _CHUNK_OVERLAP_LINES

    # Only iterate module-level and class-level directly — no recursive walk
    for node in tree.body:
        kind = _AST_KIND.get(type(node))
        if kind is not None:
            _add(node, kind)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                child_kind = _AST_KIND.get(type(child))
                if child_kind is not None:
                    _add(child, child_kind)

    # Capture gap lines (imports, module-level assignments, etc.)
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
                        language="python",
                        content_hash=_content_hash(content),
                    )
                )

    chunks.sort(key=lambda c: c.start_line)
    return chunks if chunks else chunk_by_lines(source, file_path, "python")


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


def chunk_with_chonkie(source: str, file_path: str, language: str) -> list[Chunk]:
    """Chunk source code using Chonkie's CodeChunker (requires chonkie[code]).

    Falls back to the AST / line-based strategy if chonkie is not installed or
    the language is unsupported. Does not set symbol_name (chonkie doesn't
    expose that information), so symbol search will return no results for
    files chunked this way.
    """
    try:
        import chonkie as _chonkie

        CodeChunker = _chonkie.CodeChunker  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return (
            chunk_with_ast(source, file_path)
            if language == "python"
            else chunk_by_lines(source, file_path, language)
        )

    try:
        cc = CodeChunker(language=language, chunk_size=1500)
        raw = cc.chunk(source)
    except Exception:
        return (
            chunk_with_ast(source, file_path)
            if language == "python"
            else chunk_by_lines(source, file_path, language)
        )

    if not raw:
        return chunk_by_lines(source, file_path, language)

    # Convert character offsets to line numbers
    line_starts = [0]
    for ch in source:
        if ch == "\n":
            line_starts.append(line_starts[-1] + 1)
        else:
            line_starts[-1] += 1
    # Rebuild as cumulative character positions per line
    cum: list[int] = [0]
    for line in source.splitlines(keepends=True):
        cum.append(cum[-1] + len(line))

    def _char_to_line(offset: int) -> int:
        # Binary search for the line containing this character offset
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cum[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-based

    chunks: list[Chunk] = []
    for c in raw:
        text = c.text
        if not text.strip():
            continue
        start_line = _char_to_line(c.start_index)
        end_line = _char_to_line(max(c.end_index - 1, c.start_index))
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=None,
                symbol_kind=SymbolKind.CHUNK,
                language=language,
                content_hash=_content_hash(text),
            )
        )
    return chunks if chunks else chunk_by_lines(source, file_path, language)


def chunk_file(file_path: Path, use_chonkie: bool = False) -> list[Chunk]:
    """Chunk a single file, choosing the best available strategy.

    :param file_path: Path to the source file.
    :param use_chonkie: If True, use Chonkie CodeChunker instead of the
        built-in AST / tree-sitter strategy (requires ``chonkie[code]``).
    :returns: List of chunks extracted from the file.
    """
    suffix = file_path.suffix.lower()
    language = EXTENSION_MAP.get(suffix)

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not source.strip():
        return []

    if use_chonkie and language:
        return chunk_with_chonkie(source, str(file_path), language)

    if language == "python":
        return chunk_with_ast(source, str(file_path))

    if language in _TS_SYMBOL_NODES:
        return chunk_with_treesitter(source, str(file_path), language)

    return chunk_by_lines(source, str(file_path), language)
