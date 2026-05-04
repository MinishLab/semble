from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from semble.types import SearchResult

_STATS_FILE = Path.home() / ".semble" / "stats.jsonl"


def log_search_stats(
    results: list[SearchResult],
    call_type: str,
    root_path: Path | None = None,
) -> None:
    """Append token-savings stats for one search/find_related call. Failures are silently ignored."""
    try:
        snippet_chars = sum(len(r.chunk.content) for r in results)

        file_chars = 0
        if root_path is not None:
            seen: set[str] = set()
            for r in results:
                fp = r.chunk.file_path
                if fp in seen:
                    continue
                seen.add(fp)
                try:
                    file_chars += len((root_path / fp).read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass

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
