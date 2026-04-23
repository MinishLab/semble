from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import Chunk, Encoder, SearchResult

_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
# scp-like syntax: [user@]host:path, where host has no '/' before the ':'.
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or GitHub repository. "
            "Call `search` to find relevant code; call `find_related` on a result to discover similar code elsewhere. "
            "For questions about a library (e.g. a PyPI/npm package), resolve the GitHub URL from your training "
            "knowledge and pass it as `repo`. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
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
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
    ) -> str:
        """Search a codebase with a natural-language or code query.

        Pass a git URL or local path as `repo` to index it on demand; indexes are cached for the session.
        Use this to find where something is implemented, understand a library, or locate related code.
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
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
    ) -> str:
        """Find code chunks semantically similar to a specific location in a file.

        Use after `search` to explore related implementations or callers.
        Pass file_path and line from a prior search result.
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
        chunk = _resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k)
        if not results:
            return f"No related chunks found for {file_path}:{line}."
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
        except asyncio.CancelledError:  # pragma: no cover
            # If this waiter was cancelled but the task is still running, preserve it for
            # other waiters. Only evict if the task itself was cancelled.
            if task.done():
                self._tasks.pop(cache_key, None)
            raise
        except Exception:
            # Build failed: evict so the next caller can retry.
            self._tasks.pop(cache_key, None)
            raise


def _resolve_chunk(chunks: list[Chunk], file_path: str, line: int) -> Chunk | None:
    """Return the chunk that contains *line* in *file_path*, or None.

    MCP tool arguments are JSON primitives (strings and ints), so the agent
    passes file_path + line rather than a Chunk object. This function
    reconstructs the Chunk at the MCP boundary before calling into the library.

    :param chunks: All indexed chunks to search.
    :param file_path: File path as stored in the index.
    :param line: 1-indexed line number to resolve.
    :return: The best-matching Chunk, or None if not found.
    """
    fallback = None
    for chunk in chunks:
        if chunk.file_path == file_path and chunk.start_line <= line <= chunk.end_line:
            if line < chunk.end_line:
                return chunk
            if fallback is None:  # line == end_line: boundary; keep as fallback for end-of-file chunks
                fallback = chunk
    return fallback


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


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


def main() -> None:
    """Entry point for the semble command-line tool."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    args = parser.parse_args()
    asyncio.run(serve(args.path, ref=args.ref))
