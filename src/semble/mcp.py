from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import Encoder, SearchResult

_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
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

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        mode: Annotated[
            Literal["hybrid", "semantic", "bm25"],
            Field(description="Search mode. 'hybrid' is best for most queries."),
        ] = "hybrid",
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1, le=20)] = 5,
    ) -> str:
        """Search a codebase with a natural-language or code query.

        Pass a git URL or local path as `repo` to clone and index it on demand.
        The index is cached so subsequent searches on the same repo are instant.
        Returns the most relevant code chunks with file paths and line numbers.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        results = index.search(query, top_k=top_k, mode=mode)
        if not results:
            return "No results found."
        return _format_results(f"Search results for: {query!r} (mode={mode})", results)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1, le=10)] = 5,
    ) -> str:
        """Find code chunks semantically similar to a specific location in a file.

        Useful for discovering related logic elsewhere in the codebase.
        Pass the same `repo` used in the original `search` call.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        results = index.find_related(file_path, line, top_k=top_k)
        if not results:
            return (
                f"No related chunks found for {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        return _format_results(f"Chunks related to {file_path}:{line}", results)

    return server


async def serve(path: str | None = None, ref: str | None = None) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model)
    if path:
        await cache.get(path, ref=ref)

    server = create_server(cache, default_source=path)
    await server.run_stdio_async()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self, model: Encoder) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._tasks: dict[str, asyncio.Task[SembleIndex]] = {}

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        is_git = _is_git_url(source)
        cache_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

        if cache_key not in self._tasks:
            if is_git:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_git, source, ref=ref, model=self._model)
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_path, cache_key, model=self._model)
                )
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # If this waiter was cancelled but the task is still running, preserve it for
            # other waiters. Only evict if the task itself was cancelled.
            if task.done():
                self._tasks.pop(cache_key, None)
            raise
        except Exception:
            # Build failed: evict so the next caller can retry.
            self._tasks.pop(cache_key, None)
            raise


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(("https://", "http://", "git@", "ssh://"))


def _format_results(header: str, results: list[SearchResult]) -> str:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        lines.append("```")
        lines.append(r.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
