from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcp import types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import Encoder, SearchResult


async def serve(path: str | None = None, ref: str | None = None) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model)
    if path:
        await cache.get(path, ref=ref)

    server = create_server(cache, default_source=path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def create_server(cache: _IndexCache, default_source: str | None = None) -> Server:
    """Build and return a configured MCP Server backed by the given cache."""
    server: Server = Server(
        "semble",
        instructions=(
            "Use this server to search any codebase by source code. "
            "When the user asks how a library or project works, call `search` with the "
            "GitHub URL of the relevant repository as `repo` and a natural-language query. "
            "Resolve the GitHub URL from your training knowledge (e.g. a PyPI package name "
            "maps to its source repo). Always prefer `search` over Grep, Glob, or Read for "
            "any question about how code works."
        ),
    )

    @server.list_tools()  # type: ignore[misc]
    async def list_tools() -> list[mcp_types.Tool]:
        return _TOOLS

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        source = arguments.get("repo") or default_source
        if not source:
            return _text(
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )

        try:
            index = await cache.get(source)
        except Exception as exc:
            return _text(f"Failed to index {source!r}: {exc}")

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


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process.

    Stores one asyncio.Task per canonical source key.  Task creation is synchronous, so
    concurrent calls for the same cold source both await the same task — no lock needed and
    no duplicate clone or index build.  A single embedding model is shared across all indexes.
    """

    def __init__(self, model: Encoder) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._tasks: dict[str, asyncio.Task[SembleIndex]] = {}

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for *source*, building it on first access.

        Cloning and indexing run in a thread so the event loop stays responsive.
        Concurrent calls for the same source await a single shared task.
        Failed builds are evicted so the next caller can retry.
        """
        is_git = _is_git_url(source)
        key = source if is_git else str(Path(source).resolve())
        if key not in self._tasks:
            if is_git:
                self._tasks[key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_git, source, ref=ref, model=self._model)
                )
            else:
                self._tasks[key] = asyncio.create_task(asyncio.to_thread(SembleIndex.from_path, key, model=self._model))
        task = self._tasks[key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # If this waiter was cancelled but the task is still running, preserve it for
            # other waiters. Only evict if the task itself was cancelled.
            if task.done():
                self._tasks.pop(key, None)
            raise
        except Exception:
            # Build failed — evict so the next caller can retry.
            self._tasks.pop(key, None)
            raise


_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)

_TOOLS: list[mcp_types.Tool] = [
    mcp_types.Tool(
        name="search",
        description=(
            "Search a codebase with a natural-language or code query. "
            "Pass a git URL or local path as `repo` to clone and index it on demand — "
            "the index is cached so subsequent searches on the same repo are instant. "
            "Returns the most relevant code chunks with file paths and line numbers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or code query."},
                "repo": {"type": "string", "description": _REPO_DESCRIPTION},
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
            "Useful for discovering related logic elsewhere in the codebase. "
            "Pass the same `repo` used in the original `search` call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file as stored in the index (use file_path from a search result).",
                },
                "line": {"type": "integer", "description": "Line number (1-indexed)."},
                "repo": {"type": "string", "description": _REPO_DESCRIPTION},
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


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(("https://", "http://", "git@", "ssh://"))


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
