from pathlib import Path
from typing import Any

from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from semble.index import SembleIndex
from semble.types import SearchResult


def _text(content: str) -> list[mcp_types.TextContent]:
    """Wrap a string in a single-element TextContent list."""
    return [mcp_types.TextContent(type="text", text=content)]


def _format_results(header: str, results: list[SearchResult]) -> list[mcp_types.TextContent]:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        lines.append("```")
        lines.append(r.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return _text("\n".join(lines))


_TOOLS: list[mcp_types.Tool] = [
    mcp_types.Tool(
        name="search",
        description=(
            "Search the indexed codebase with a natural-language or code query. "
            "Returns the most relevant code chunks with file paths and line numbers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code query."},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "bm25"],
                    "default": "hybrid",
                    "description": "Search mode. 'hybrid' is best for most queries.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of results to return.",
                },
            },
            "required": ["query"],
        },
    ),
    mcp_types.Tool(
        name="find_related",
        description=(
            "Find code chunks semantically similar to a specific location in a file. "
            "Useful for discovering related logic elsewhere in the codebase."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file."},
                "line": {"type": "integer", "description": "Line number (1-indexed)."},
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of similar chunks to return.",
                },
            },
            "required": ["file_path", "line"],
        },
    ),
]


def create_server(index: SembleIndex) -> Server:
    """Build and return a configured MCP Server for the given index.

    :param index: A SembleIndex that has already been indexed.
    :return: Configured MCP Server.
    """
    server: Server = Server("semble")

    @server.list_tools()  # type: ignore[misc]
    async def list_tools() -> list[mcp_types.Tool]:
        return _TOOLS

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        if name == "search":
            query: str = arguments["query"]
            mode: str = arguments.get("mode", "hybrid")
            top_k: int = int(arguments.get("top_k", 5))
            results = index.search(query, top_k=top_k, mode=mode)
            if not results:
                return _text("No results found.")
            return _format_results(f"Search results for: {query!r} (mode={mode})", results)

        if name == "find_related":
            file_path: str = arguments["file_path"]
            line: int = int(arguments["line"])
            top_k = int(arguments.get("top_k", 5))
            results = index.find_related(file_path, line, top_k=top_k)
            if not results:
                return _text(
                    f"No related chunks found for {file_path}:{line}. "
                    "Make sure the file is indexed and the line number is within a known chunk."
                )
            return _format_results(f"Chunks related to {file_path}:{line}", results)

        raise ValueError(f"Unknown tool: {name!r}")

    return server


async def serve(path: str) -> None:
    """Index path and start an MCP stdio server.

    :param path: Directory to index and serve.
    """
    index = SembleIndex()
    index.index(str(Path(path).resolve()))

    server = create_server(index)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
