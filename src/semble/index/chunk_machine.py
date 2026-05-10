from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from logging import getLogger

from tree_sitter import Node, Parser
from tree_sitter_language_pack import SupportedLanguage, get_parser

logger = getLogger(__name__)


@dataclass
class ChunkBoundary:
    start: int
    end: int


@dataclass
class Group:
    start: int
    end: int
    length: int


@cache
def _cached_get_parser(language: SupportedLanguage) -> Parser:
    """Gets a parser from tree_sitter."""
    return get_parser(language)


def _merge_adjacent_groups(
    groups: list[Group],
    desired_length: int,
) -> list[tuple[int, int]]:
    """Merge adjacent groups up to desired_length.

    Input groups are (start_byte, end_byte, length).
    Output groups are (start_byte, end_byte).
    """
    merged: list[tuple[int, int]] = []

    current_start = groups[0].start
    current_end = groups[0].end
    current_length = groups[0].length

    for group in groups[1:]:
        start, end, length = group.start, group.end, group.length

        if current_length + length > desired_length:
            merged.append((current_start, current_end))
            current_start = start
            current_end = end
            current_length = length
            continue

        current_end = end
        current_length += length

    merged.append((current_start, current_end))

    return merged


def _group_node_raw(node: Node, desired_length: int) -> list[Group]:
    """Recursively split oversized syntax nodes.

    Returns groups as Group(start_byte, end_byte, length).
    """
    if not node.children:
        length = node.end_byte - node.start_byte
        return [Group(node.start_byte, node.end_byte, length)]

    groups: list[Group] = []
    children = node.children
    index = 0

    while index < len(children):
        child = children[index]
        length = child.end_byte - child.start_byte

        if length > desired_length:
            groups.extend(_group_node_raw(child, desired_length))
            index += 1
            continue

        current_start = child.start_byte
        current_end = child.end_byte
        current_length = length
        index += 1

        while index < len(children):
            child = children[index]
            length = child.end_byte - child.start_byte

            if length > desired_length:
                break

            if current_length + length > desired_length:
                break

            current_end = child.end_byte
            current_length += length
            index += 1

        groups.append(Group(current_start, current_end, current_length))

    return groups


def _group_node(node: Node, desired_length: int) -> list[tuple[int, int]]:
    """Recursively group a node, then merge adjacent groups."""
    raw_groups = _group_node_raw(node, desired_length)
    return _merge_adjacent_groups(raw_groups, desired_length)


class Chunker:
    def __init__(
        self,
        language: str,
        desired_size: int,
    ) -> None:
        """Initialize the chunker."""
        self.language = language
        self.desired_size = desired_size

    def chunk(self, text: str) -> list[ChunkBoundary]:
        """Chunk source code."""
        if not text.strip():
            return []

        as_bytes = text.encode("utf-8")
        parser = _cached_get_parser(self.language)
        root = parser.parse(as_bytes).root_node

        chunks = []
        for start, end in _group_node(root, self.desired_size):
            if text.isascii():
                start_char = start
                end_char = end
            else:
                start_char = len(as_bytes[:start].decode("utf-8"))
                end_char = len(as_bytes[:end].decode("utf-8"))
            chunks.append(ChunkBoundary(start=start_char, end=end_char))

        return chunks
