from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from logging import getLogger

from tree_sitter import Node, Parser
from tree_sitter_language_pack import LanguageNotFoundError, SupportedLanguage, get_parser

from semble.index.files import ALL_LANGUAGES

logger = getLogger(__name__)

_RECURSION_DEPTH = 500
_MIN_CHUNK_SIZE = 50


def is_supported_language(language: str) -> bool:
    """Check if the language is supported by tree-sitter."""
    return language in ALL_LANGUAGES


@dataclass
class ChunkBoundary:
    """The output of the internal chunking algorithm."""

    start: int
    end: int


@cache
def _cached_get_parser(language: SupportedLanguage) -> Parser | None:
    """Gets a parser from tree_sitter."""
    try:
        return get_parser(language)
    except LanguageNotFoundError:
        logger.warning("Language %s not found, falling back to line chunking", language)
    except Exception:
        logger.error("Uncaught exception in _cached_get_parser", exc_info=True)
    return None


def _merge_adjacent_chunks(
    chunks: list[ChunkBoundary],
    desired_length: int,
) -> list[ChunkBoundary]:
    """Merge adjacent chunks up to the desired length."""
    merged = []

    current_start = chunks[0].start
    current_end = chunks[0].end
    current_length = current_end - current_start

    for group in chunks[1:]:
        start, end = group.start, group.end
        length = end - start

        if current_length + length > desired_length:
            merged.append(ChunkBoundary(start=current_start, end=current_end))
            current_start = start
            current_end = end
            current_length = length
            continue

        current_end = end
        current_length += length

    merged.append(ChunkBoundary(start=current_start, end=current_end))

    return merged


def _node_start_byte(node: Node) -> int:
    """Get the start byte offset of a node, compatible with tree-sitter 0.25+."""
    val = node.start_byte
    return val() if callable(val) else val


def _node_end_byte(node: Node) -> int:
    """Get the end byte offset of a node, compatible with tree-sitter 0.25+."""
    val = node.end_byte
    return val() if callable(val) else val


def _node_child_count(node: Node) -> int:
    """Get the number of children of a node, compatible with tree-sitter 0.25+."""
    # tree-sitter <0.24: node.children is a list
    if hasattr(node, "children") and isinstance(node.children, list):
        return len(node.children)
    # tree-sitter >=0.25: node.child_count() is a method
    val = node.child_count
    return val() if callable(val) else val


def _node_child(node: Node, index: int) -> Node:
    """Get the i-th child of a node, compatible with tree-sitter 0.25+."""
    # tree-sitter <0.24: node.children is a list
    if hasattr(node, "children") and isinstance(node.children, list):
        return node.children[index]
    # tree-sitter >=0.25: node.child(i) is a method
    return node.child(index)


def _node_children(node: Node) -> list[Node]:
    """Get all children of a node as a list, compatible with tree-sitter 0.25+."""
    # tree-sitter <0.24: node.children is a list
    if hasattr(node, "children") and isinstance(node.children, list):
        return node.children
    # tree-sitter >=0.25: use child(i) + child_count()
    count = _node_child_count(node)
    return [node.child(i) for i in range(count)]


def _merge_node_inner(node: Node, desired_length: int, i: int) -> list[ChunkBoundary]:
    """Recursively merge and split nodes."""
    child_count = _node_child_count(node)
    # If there are no child nodes, the only thing we can do is return the current node.
    if child_count == 0:
        return [ChunkBoundary(_node_start_byte(node), _node_end_byte(node))]

    start_byte = _node_start_byte(node)
    end_byte = _node_end_byte(node)
    length = end_byte - start_byte
    # Prevent recursion issues. A depth of > 500 is unlikely
    if i > _RECURSION_DEPTH:
        logger.warning("Recursion depth exceeded in chunk.")
        return [ChunkBoundary(start_byte, end_byte)]
    # Prevent recursing into short chunks.
    if length < _MIN_CHUNK_SIZE:
        return [ChunkBoundary(start_byte, end_byte)]

    groups: list[ChunkBoundary] = []
    children = _node_children(node)
    index = 0

    while index < len(children):
        child = children[index]
        start = _node_start_byte(child)
        end = _node_end_byte(child)
        length = end - start

        # Increment the pointer, as we accessed a child node.
        index += 1
        # If this single chunk is longer than the desired length
        # we try to split it again.
        if length > desired_length:
            groups.extend(_merge_node_inner(child, desired_length, i + 1))
            continue

        while index < len(children):
            # Extend the current group with or more children, if they fit.
            child = children[index]
            child_start = _node_start_byte(child)
            child_end = _node_end_byte(child)
            child_length = child_end - child_start

            if length + child_length > desired_length:
                break

            end = child_end
            length += child_length
            index += 1

        groups.append(ChunkBoundary(start, end))

    return groups


def _merge_node(node: Node, desired_length: int) -> list[ChunkBoundary]:
    """Recursively turn nodes into chunks, then merge adjacent chunks."""
    raw_chunks = _merge_node_inner(node, desired_length, 0)
    return _merge_adjacent_chunks(raw_chunks, desired_length)


def chunk_lines(text: str, desired_length: int) -> list[ChunkBoundary]:
    """Chunk source code by line."""
    if not text.strip():
        return []
    lines_as_groups = []
    index = 0
    for line in text.splitlines(keepends=True):
        lines_as_groups.append(ChunkBoundary(start=index, end=index + len(line)))
        index += len(line)

    return _merge_adjacent_chunks(lines_as_groups, desired_length)


def chunk(text: str, language: str, desired_length: int) -> list[ChunkBoundary] | None:
    """Chunk source code."""
    if not text.strip():
        return []

    as_bytes = text.encode("utf-8")
    parser = _cached_get_parser(language)
    if parser is None:
        return None
    tree = parser.parse(text)
    # tree-sitter >=0.25: root_node is a method; <0.24: it is a property.
    root = tree.root_node() if callable(tree.root_node) else tree.root_node

    chunks = []
    for chunk_boundary in _merge_node(root, desired_length):
        start_char = len(as_bytes[: chunk_boundary.start].decode("utf-8"))
        end_char = len(as_bytes[: chunk_boundary.end].decode("utf-8"))
        chunks.append(ChunkBoundary(start=start_char, end=end_char))

    return chunks
