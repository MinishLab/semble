from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Generator
from threading import BoundedSemaphore
from typing import TypeVar

_R = TypeVar("_R")


def _configured_int(name: str, default: int, minimum: int) -> int:
    configured = os.environ.get(name)
    if configured:
        with contextlib.suppress(ValueError):
            return max(minimum, int(configured))
    return default


_INDEX_WORKER_QUOTA = _configured_int("SEMBLE_INDEX_WORKER_QUOTA", max(1, os.cpu_count() or 1), 1)
_INDEX_WORKER_SEMAPHORE = BoundedSemaphore(_INDEX_WORKER_QUOTA)


def index_worker_quota() -> int:
    """Return the shared worker quota for one indexing process."""
    return _INDEX_WORKER_QUOTA


def git_metadata_worker_count() -> int:
    """Return worker count for git metadata work within the shared quota."""
    configured = os.environ.get("SEMBLE_GIT_METADATA_WORKERS")
    if configured:
        with contextlib.suppress(ValueError):
            return max(1, min(_INDEX_WORKER_QUOTA, int(configured)))
    return min(4, max(1, _INDEX_WORKER_QUOTA // 4), _INDEX_WORKER_QUOTA)


def foreground_worker_count(background_workers: int = 0, maximum: int | None = None) -> int:
    """Return foreground workers left after reserving background quota."""
    available = max(1, _INDEX_WORKER_QUOTA - max(0, background_workers))
    if maximum is not None:
        return max(1, min(maximum, available))
    return available


@contextlib.contextmanager
def reserve_index_workers(count: int) -> Generator[None, None, None]:
    """Reserve shared indexing worker slots for blocking CPU/IO work."""
    normalized = max(1, min(_INDEX_WORKER_QUOTA, count))
    for _ in range(normalized):
        _INDEX_WORKER_SEMAPHORE.acquire()
    try:
        yield
    finally:
        for _ in range(normalized):
            _INDEX_WORKER_SEMAPHORE.release()


def run_with_index_worker(fn: Callable[..., _R], *args: object) -> _R:
    """Run a callable while holding one shared indexing worker slot."""
    with reserve_index_workers(1):
        return fn(*args)
