from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.cache import save_index_to_cache
from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import ContentType
from semble.utils import format_results, is_git_url, resolve_chunk

logger = logging.getLogger(__name__)

_REPO_DESCRIPTION = (
    "A local directory path or https:// or http:// git URL (e.g. https://github.com/org/repo) to index and "
    "search. The index is cached after the first call, so repeat queries are fast."
)

_CACHE_MAX_SIZE = 10  # Max number of cached indexes to keep in memory


async def _get_index(
    repo: str,
    cache: _IndexCache,
) -> SembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes."""
    if is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    try:
        return await cache.get(repo)
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc


def create_server(cache: _IndexCache) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search for any local or remote git repository. "
            "Call `search` once with a focused query, it returns the file path and exact line. "
            "Navigate directly to that file at the given line; do not grep for the same content. "
            "Use `find_related` to discover similar code elsewhere in the same repo. "
            "When working in a local project, pass the project root as `repo`. "
            "For remote repos, pass an explicit https:// URL. Never guess or infer URLs."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source to include per result. "
                    "Default (10): function/class signature + first body lines, enough to confirm the location. "
                    "0: file path and line range only. None: full chunk (~10-20 lines). "
                    "If the snippet does not contain enough context to confirm you have the right location, "
                    "call again with max_snippet_lines=None."
                ),
                ge=0,
            ),
        ] = 10,
    ) -> str:
        """Search once with a focused query describing what the code does or its name.

        Write queries using function/class names or behavior descriptions, not error messages.
        Returns file paths and line numbers — navigate directly there, do not repeat the search.
        Pass a git URL or local path as `repo`; indexes are cached for the session.
        """
        try:
            index = await _get_index(repo, cache)
        except ValueError as exc:
            return str(exc)
        results = index.search(query, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": "No results found."})
        return json.dumps(format_results(query, results, max_snippet_lines))

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str, Field(description=_REPO_DESCRIPTION)],
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
        max_snippet_lines: Annotated[
            int | None,
            Field(
                description=(
                    "Lines of source per result. "
                    "Default 10 = signature + first body lines. 0 = location only. None = full chunk."
                ),
                ge=0,
            ),
        ] = 10,
    ) -> str:
        """Find code similar to a known location.

        Useful for discovering all implementations of an interface, all callers of a function,
        or all tests for a class. Use after `search` when you need related code beyond the primary result.
        Pass `file_path` and `line` from a prior search result.
        """
        try:
            index = await _get_index(repo, cache)
        except ValueError as exc:
            return str(exc)
        chunk = resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k, max_snippet_lines=max_snippet_lines)
        if not results:
            return json.dumps({"error": f"No related chunks found for {file_path}:{line}."})
        label = f"Chunks related to {file_path}:{line}"
        return json.dumps(format_results(label, results, max_snippet_lines))

    return server


async def serve(
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> None:
    """Start an MCP stdio server."""
    cache = _IndexCache(content=content)

    async def _load_and_prewarm() -> None:
        """Pre-load the embedding model in parallel with starting the server."""
        try:
            _, cache._model_path = await asyncio.to_thread(load_model)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            cache._model_error = exc
            return
        finally:
            cache._model_ready.set()

    init_task = asyncio.create_task(_load_and_prewarm())
    server = create_server(cache)
    try:
        await server.run_stdio_async()
    finally:
        if not init_task.done():
            init_task.cancel()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self, content: Sequence[ContentType] = (ContentType.CODE,)) -> None:
        """Initialise an empty cache."""
        self._model_path: str | None = None
        self._model_error: BaseException | None = None
        self._model_ready = asyncio.Event()
        self._content = content
        self._tasks: OrderedDict[str, asyncio.Task[SembleIndex]] = OrderedDict()  # ordered for LRU eviction

    async def _await_model(self) -> str:
        """Block until the model is installed; re-raise the load error if it failed."""
        await self._model_ready.wait()
        if self._model_error is not None:
            raise self._model_error
        assert self._model_path is not None
        return self._model_path

    def _compute_cache_key(self, source: str, ref: str | None = None) -> str:
        """Compute the canonical cache key for a source."""
        is_git = is_git_url(source)
        return (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

    def _build_and_cache_index(self, source: str, ref: str | None, model_path: str, cache_key: str) -> SembleIndex:
        """Build an index for the given source and cache it."""
        index = (
            SembleIndex.from_git(source, ref=ref, model_path=model_path, content=self._content)
            if is_git_url(source)
            else SembleIndex.from_path(cache_key, model_path=model_path, content=self._content)
        )
        try:
            save_index_to_cache(index, cache_key)
        except Exception:
            logger.warning("Failed to save index cache for %r", cache_key, exc_info=True)
        return index

    def evict(self, source: str) -> None:
        self._tasks.pop(self._compute_cache_key(source), None)

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        cache_key = self._compute_cache_key(source, ref)

        if cache_key not in self._tasks:
            model_path = await self._await_model()
            # Re-check after the await: another caller may have populated the entry.
            if cache_key not in self._tasks:
                if len(self._tasks) >= _CACHE_MAX_SIZE:
                    self._tasks.popitem(last=False)
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(self._build_and_cache_index, source, ref, model_path, cache_key)
                )
        self._tasks.move_to_end(cache_key)
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
