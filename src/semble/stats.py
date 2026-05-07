from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from semble.types import SearchResult

_STATS_FILE = Path.home() / ".semble" / "savings.jsonl"


def log_search_stats(
    results: list[SearchResult],
    call_type: str,
    file_sizes: dict[str, int] | None = None,
) -> None:
    """Append token-savings stats for one search/find_related call. Failures are silently ignored."""
    try:
        snippet_chars = sum(len(result.chunk.content) for result in results)
        if file_sizes:
            file_chars = sum(file_sizes.get(path, 0) for path in {result.chunk.file_path for result in results})
        else:
            file_chars = 0

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "call": call_type,
            "results": len(results),
            "snippet_chars": snippet_chars,
            "file_chars": file_chars,
        }
        _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _STATS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
