from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Literal

import watchfiles
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import Encoder
from semble.utils import _format_results, _is_git_url, _resolve_chunk

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "https:// or http:// git URL (e.g. https://github.com/org/repo) or local directory path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)

_CACHE_MAX_SIZE = 10  # Max number of cached indexes to keep in memory


async def _get_index(
    repo: str | None,
    default_source: str | None,
    cache: _IndexCache,
) -> SembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes."""
    if repo is not None and _is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    source = repo or default_source
    if not source:
        raise ValueError(
            "No repo specified and no default index. "
            "Pass an https:// or http:// git URL or local directory path as `repo`."
        )
    try:
        return await cache.get(source)
    except Exception as exc:
        raise ValueError(f"Failed to index {source!r}: {exc}") from exc


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or remote git repository. "
            "Call `search` to find relevant code; call `find_related` on a result to discover similar code elsewhere. "
            "When working in a local project, pass the project root as `repo`. "
            "For remote repos, pass an explicit https:// URL. Never guess or infer URLs. "
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
        try:
            index = await _get_index(repo, default_source, cache)
        except ValueError as exc:
            return str(exc)
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
        try:
            index = await _get_index(repo, default_source, cache)
        except ValueError as exc:
            return str(exc)
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


async def serve(path: str | None = None, ref: str | None = None, include_text_files: bool = False) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model, include_text_files=include_text_files)
    if path:
        await cache.get(path, ref=ref)
        if not _is_git_url(path):
            await cache.start_watcher(path)

    server = create_server(cache, default_source=path)
    await server.run_stdio_async()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self, model: Encoder, include_text_files: bool = False) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._include_text_files = include_text_files
        self._tasks: OrderedDict[str, asyncio.Task[SembleIndex]] = OrderedDict()  # ordered for LRU eviction
        self._watcher_task: asyncio.Task[None] | None = None

    def _compute_cache_key(self, source: str, ref: str | None = None) -> str:
        """Compute the canonical cache key for a source."""
        is_git = _is_git_url(source)
        return (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

    def evict(self, source: str) -> None:
        self._tasks.pop(self._compute_cache_key(source), None)

    async def start_watcher(self, path: str) -> None:
        """Start a background task that re-indexes the path whenever files change."""
        self._watcher_task = asyncio.create_task(self._watch_loop(path))

    async def _watch_loop(self, path: str) -> None:
        """Watch the given path for changes and evict the cache entry on changes."""
        try:
            async for _ in watchfiles.awatch(path):
                self.evict(path)
                try:
                    await self.get(path)
                except Exception:
                    logger.warning("Failed to rebuild index for %r after file change", path, exc_info=True)
        except Exception:
            pass

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        cache_key = self._compute_cache_key(source, ref)

        if cache_key in self._tasks:
            self._tasks.move_to_end(cache_key)
        else:
            if len(self._tasks) >= _CACHE_MAX_SIZE:
                self._tasks.popitem(last=False)
            if _is_git_url(source):
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(
                        SembleIndex.from_git,
                        source,
                        ref=ref,
                        model=self._model,
                        include_text_files=self._include_text_files,
                    )
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(
                        SembleIndex.from_path, cache_key, model=self._model, include_text_files=self._include_text_files
                    )
                )
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self.evict(source)
            raise
        except Exception:
            # Only evict if this task hasn't already been replaced by evict()+get().
            if self._tasks.get(cache_key) is task:
                self.evict(source)
            raise
