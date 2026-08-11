from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.cache import get_validated_cache, save_index_to_cache
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
_MIN_REVALIDATE_FACTOR = 3  # Don't recheck staleness sooner than this many times the last build's duration
ContentSelection = Literal["code", "docs", "config", "all"]
_CacheKey = tuple[str, tuple[ContentType, ...]]


async def _get_index(repo: str, cache: _IndexCache, content: Sequence[ContentType]) -> SembleIndex:
    """Return a cached index for a repo, rejecting unsafe git transport schemes."""
    if is_git_url(repo) and not repo.startswith(("https://", "http://")):
        raise ValueError(f"Only https://, http://, or local directory paths are accepted as `repo`. Got: {repo!r}")
    try:
        return await cache.get(repo, content=content)
    except Exception as exc:
        raise ValueError(f"Failed to index {repo!r}: {exc}") from exc


def _resolve_content_selection(
    content: ContentSelection | None, default_content: Sequence[ContentType]
) -> tuple[ContentType, ...]:
    """Resolve an MCP content selection to exact index content types."""
    if content is None:
        return tuple(default_content)
    if content == "all":
        return tuple(ContentType)
    return (ContentType(content),)


def create_server(cache: _IndexCache, default_content: Sequence[ContentType] = (ContentType.CODE,)) -> FastMCP:
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
        content: Annotated[
            ContentSelection | None,
            Field(description="Content to search. Defaults to the MCP server's configured content."),
        ] = None,
    ) -> str:
        """Search once with a focused query describing what the code does or its name.

        Write queries using function/class names or behavior descriptions, not error messages.
        Returns file paths and line numbers — navigate directly there, do not repeat the search.
        Pass a git URL or local path as `repo`; indexes are cached for the session.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            index = await _get_index(repo, cache, selected_content)
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
        content: Annotated[
            ContentSelection | None,
            Field(description="Content containing the related file. Defaults to the MCP server configuration."),
        ] = None,
    ) -> str:
        """Find code similar to a known location.

        Useful for discovering all implementations of an interface, all callers of a function,
        or all tests for a class. Use after `search` when you need related code beyond the primary result.
        Pass `file_path` and `line` from a prior search result.
        """
        selected_content = _resolve_content_selection(content, default_content)
        try:
            index = await _get_index(repo, cache, selected_content)
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
    cache = _IndexCache()

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
    server = create_server(cache, default_content=content)
    try:
        await server.run_stdio_async()
    finally:
        if not init_task.done():
            init_task.cancel()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self) -> None:
        """Initialise an empty cache."""
        self._model_path: str | None = None
        self._model_error: BaseException | None = None
        self._model_ready = asyncio.Event()
        self._tasks: OrderedDict[_CacheKey, asyncio.Task[SembleIndex]] = OrderedDict()  # ordered for LRU eviction
        self._revalidate_after: dict[_CacheKey, float] = {}

    async def _await_model(self) -> str:
        """Block until the model is installed; re-raise the load error if it failed."""
        await self._model_ready.wait()
        if self._model_error is not None:
            raise self._model_error
        assert self._model_path is not None
        return self._model_path

    def _compute_cache_key(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> _CacheKey:
        """Compute the canonical key for an exact index variant."""
        is_git = is_git_url(source)
        source_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())
        normalized = tuple(content_type for content_type in ContentType if content_type in content)
        return source_key, normalized

    def _build_index(self, source: str, ref: str | None, model_path: str, cache_key: _CacheKey) -> SembleIndex:
        """Build an index for the given source and cache it."""
        source_key, content = cache_key
        index = (
            SembleIndex.from_git(source, ref=ref, model_path=model_path, content=content)
            if is_git_url(source)
            else SembleIndex.from_path(source_key, model_path=model_path, content=content)
        )
        try:
            save_index_to_cache(index, source_key)
        except Exception:
            logger.warning("Failed to save index cache for %r", source_key, exc_info=True)
        return index

    async def _build_tracked(self, source: str, ref: str | None, model_path: str, cache_key: _CacheKey) -> SembleIndex:
        """Build an index and, for local paths, record when its staleness cooldown ends.

        The cooldown write happens after the await, i.e. back on the event loop thread,
        regardless of which thread `_build_index` itself ran on.
        """
        start = time.monotonic()
        index = await asyncio.to_thread(self._build_index, source, ref, model_path, cache_key)
        if not is_git_url(source):
            finished = time.monotonic()
            self._revalidate_after[cache_key] = finished + (finished - start) * _MIN_REVALIDATE_FACTOR
        return index

    def evict(self, cache_key: _CacheKey) -> None:
        """Evict one exact index variant from memory."""
        self._tasks.pop(cache_key, None)
        self._revalidate_after.pop(cache_key, None)

    async def _evict_if_stale(self, cache_key: _CacheKey) -> None:
        """Evict a cached local-path entry whose on-disk cache no longer matches its files.

        Skipped while inside the cooldown window so repos that are slow to build aren't
        rebuilt faster than they can be served.
        """
        cached = self._tasks.get(cache_key)
        if (
            cached is None
            or is_git_url(cache_key[0])
            or not cached.done()
            or cached.cancelled()
            or cached.exception() is not None
        ):
            return
        if time.monotonic() < self._revalidate_after.get(cache_key, 0.0):
            return
        validated = await asyncio.to_thread(get_validated_cache, cache_key[0], self._model_path, cache_key[1])
        # Only evict if this entry hasn't already been replaced by a concurrent caller.
        if validated is None and self._tasks.get(cache_key) is cached:
            self.evict(cache_key)

    async def get(
        self,
        source: str,
        ref: str | None = None,
        content: Sequence[ContentType] = (ContentType.CODE,),
    ) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access.

        Local paths are revalidated against the on-disk cache on every call (subject to a
        cooldown scaled by build time), so an entry is rebuilt once its files change.
        """
        cache_key = self._compute_cache_key(source, ref, content)
        await self._evict_if_stale(cache_key)

        if cache_key not in self._tasks:
            model_path = await self._await_model()
            # Re-check after the await: another caller may have populated the entry.
            if cache_key not in self._tasks:
                if len(self._tasks) >= _CACHE_MAX_SIZE:
                    evicted_key, _ = self._tasks.popitem(last=False)
                    self._revalidate_after.pop(evicted_key, None)
                self._tasks[cache_key] = asyncio.create_task(self._build_tracked(source, ref, model_path, cache_key))
        self._tasks.move_to_end(cache_key)
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self.evict(cache_key)
            raise
        except Exception:
            # Only evict if this task hasn't already been replaced by evict()+get().
            if self._tasks.get(cache_key) is task:
                self.evict(cache_key)
            raise
