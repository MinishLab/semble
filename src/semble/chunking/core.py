from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from logging import getLogger

from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, get_parser, manifest_languages

logger = getLogger(__name__)


_TREE_SITTER_LANGUAGES: frozenset[str] = frozenset(manifest_languages())


def is_supported_language(language: str) -> bool:
    """Check if the language is supported by tree-sitter."""
    return language in _TREE_SITTER_LANGUAGES


@dataclass
class ChunkBoundary:
    """The output of the internal chunking algorithm."""

    start: int
    end: int


@cache
def _cached_get_parser(language: SupportedLanguage) -> Parser:
    """Gets a parser from tree_sitter."""
    return get_parser(language)


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


def _merge_node_inner(node: Node, desired_length: int) -> list[ChunkBoundary]:
    """Recursively merge and split nodes."""
    # If there are no child nodes, the only thing we can do is return the current node.
    if not node.children:
        return [ChunkBoundary(node.start_byte, node.end_byte)]

    groups: list[ChunkBoundary] = []
    children = node.children
    index = 0

    while index < len(children):
        child = children[index]
        start = child.start_byte
        end = child.end_byte
        length = child.end_byte - child.start_byte

        # Increment the pointer, as we accessed a child node.
        index += 1
        # If this single chunk is longer than the desired length
        # we try to split it again.
        if length > desired_length:
            groups.extend(_merge_node_inner(child, desired_length))
            continue

        while index < len(children):
            # Extend the current group with or more children, if they fit.
            child = children[index]
            child_length = child.end_byte - child.start_byte

            if length + child_length > desired_length:
                break

            end = child.end_byte
            length += child_length
            index += 1

        groups.append(ChunkBoundary(start, end))

    return groups


def _merge_node(node: Node, desired_length: int) -> list[ChunkBoundary]:
    """Recursively turn nodes into chunks, then merge adjacent chunks."""
    raw_chunks = _merge_node_inner(node, desired_length)
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


def chunk(text: str, language: str, desired_length: int) -> list[ChunkBoundary]:
    """Chunk source code."""
    if not text.strip():
        return []

    as_bytes = text.encode("utf-8")
    parser = _cached_get_parser(language)
    root = parser.parse(as_bytes).root_node

    chunks = []
    for chunk_boundary in _merge_node(root, desired_length):
        start_char = len(as_bytes[: chunk_boundary.start].decode("utf-8"))
        end_char = len(as_bytes[: chunk_boundary.end].decode("utf-8"))
        chunks.append(ChunkBoundary(start=start_char, end=end_char))

    return chunks
