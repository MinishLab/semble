from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import watchfiles
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import ContentType
from semble.utils import format_results, is_git_url, resolve_chunk

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
    if repo is not None and is_git_url(repo) and not repo.startswith(("https://", "http://")):
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
        results = index.search(query, top_k=top_k)
        if not results:
            return "No results found."
        return format_results(f"Search results for: {query!r}", results)

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
        chunk = resolve_chunk(index.chunks, file_path, line)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k)
        if not results:
            return f"No related chunks found for {file_path}:{line}."
        return format_results(f"Chunks related to {file_path}:{line}", results)

    return server


async def serve(
    path: str | None = None,
    ref: str | None = None,
    content: Sequence[ContentType] = (ContentType.CODE,),
) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    cache = _IndexCache(content=content)

    async def _load_and_prewarm() -> None:
        """Pre-load the model and optionally pre-index the default source in parallel with starting the server."""
        try:
            _, cache._model_path = await asyncio.to_thread(load_model)
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            cache._model_error = exc
            return
        finally:
            cache._model_ready.set()
        if path:
            try:
                await cache.get(path, ref=ref)
            except Exception:
                logger.warning("Failed to pre-index %r at startup", path, exc_info=True)
            if not is_git_url(path):
                await cache.start_watcher(path)

    init_task = asyncio.create_task(_load_and_prewarm())
    server = create_server(cache, default_source=path)
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
        self._watcher_task: asyncio.Task[None] | None = None

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

        if cache_key not in self._tasks:
            model_path = await self._await_model()
            # Re-check after the await: another caller may have populated the entry.
            if cache_key not in self._tasks:
                if len(self._tasks) >= _CACHE_MAX_SIZE:
                    self._tasks.popitem(last=False)
                if is_git_url(source):
                    self._tasks[cache_key] = asyncio.create_task(
                        asyncio.to_thread(
                            SembleIndex.from_git,
                            source,
                            ref=ref,
                            model_path=model_path,
                            content=self._content,
                        )
                    )
                else:
                    self._tasks[cache_key] = asyncio.create_task(
                        asyncio.to_thread(
                            SembleIndex.from_path,
                            cache_key,
                            model_path=model_path,
                            content=self._content,
                        )
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
