import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from semble.types import CallType, SearchResult

logger = logging.getLogger(__name__)

_STATS_FILE = Path.home() / ".semble" / "savings.jsonl"


@dataclass
class BucketStats:
    calls: int = 0
    snippet_chars: int = 0
    file_chars: int = 0

    def add(self, snippet_chars: int, file_chars: int) -> None:
        """Update stats with a call and its character counts."""
        self.calls += 1
        self.snippet_chars += snippet_chars
        self.file_chars += file_chars


@dataclass
class SavingsSummary:
    buckets: dict[str, BucketStats]
    call_type_counts: dict[str, int]


def save_search_stats(
    results: list[SearchResult],
    call_type: CallType,
    file_sizes: dict[str, int],
) -> None:
    """Save stats about a search or find_related call to the stats file."""
    try:
        snippet_chars = sum(len(result.chunk.content) for result in results)
        file_chars = sum(
            file_sizes[path] for path in {result.chunk.file_path for result in results} if path in file_sizes
        )

        record = {
            "ts": datetime.now(timezone.utc).timestamp(),
            "call": call_type,
            "results": len(results),
            "snippet_chars": snippet_chars,
            "file_chars": file_chars,
        }
        _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _STATS_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def build_savings_summary(path: Path = _STATS_FILE) -> SavingsSummary:
    """Read savings.jsonl and return a SavingsSummary."""
    now = datetime.now(timezone.utc)
    today = now.date()
    seven_days_ago = (now - timedelta(days=7)).date()

    buckets = {
        "Today": BucketStats(),
        "Last 7 days": BucketStats(),
        "All time": BucketStats(),
    }
    call_type_counts: defaultdict[str, int] = defaultdict(int)

    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in stats file")
                continue
            snippet_chars = record["snippet_chars"]
            file_chars = record["file_chars"]
            call_type = record["call"]
            call_type_counts[call_type] += 1
            dt = datetime.fromtimestamp(record["ts"], tz=timezone.utc)
            in_today = dt.date() == today
            in_last_7 = dt.date() > seven_days_ago
            buckets["All time"].add(snippet_chars, file_chars)
            if in_last_7:
                buckets["Last 7 days"].add(snippet_chars, file_chars)
            if in_today:
                buckets["Today"].add(snippet_chars, file_chars)

    return SavingsSummary(buckets=buckets, call_type_counts=dict(call_type_counts))


def format_savings_report(path: Path | None = None, *, verbose: bool = False) -> str:
    """Return a formatted token-savings report."""
    if path is None:
        path = _STATS_FILE
    if not path.exists():
        return "No stats yet. Run a search first."

    summary = build_savings_summary(path)
    bar_width = 16
    heavy_line = "  " + "═" * 64
    light_line = "  " + "─" * 64

    lines = [
        "",
        "  Semble Token Savings",
        heavy_line,
        f"  {'Period':<12}  {'Calls':<6}  Savings",
        light_line,
    ]
    for label, bucket in summary.buckets.items():
        saved_chars = max(0, bucket.file_chars - bucket.snippet_chars)
        saved_tokens = saved_chars // 4  # standard ~4 chars/token approximation
        if saved_tokens >= 1_000_000:
            saved_str = f"~{saved_tokens / 1_000_000:.1f}M"
        elif saved_tokens >= 1000:
            saved_str = f"~{saved_tokens / 1000:.1f}k"
        else:
            saved_str = f"~{saved_tokens}"
        calls_str = f"{bucket.calls / 1000:.1f}k" if bucket.calls >= 1000 else str(bucket.calls)
        if bucket.file_chars > 0:
            ratio = saved_chars / bucket.file_chars
            filled = round(ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = round(ratio * 100)
            lines.append(f"  {label:<12}  {calls_str:<6}  [{bar}]  {saved_str} tokens ({pct}%)")
        else:
            lines.append(f"  {label:<12}  {calls_str:<6}  [{'░' * bar_width}]  {saved_str} tokens")
    if verbose and summary.call_type_counts:
        lines += ["", "  Usage Breakdown", light_line, f"  {'Call type':<16}  Calls"]
        for call_type, count in sorted(summary.call_type_counts.items()):
            count_str = f"{count / 1000:.1f}k" if count >= 1000 else str(count)
            lines.append(f"  {call_type:<16}  {count_str}")
        lines.append(heavy_line)
    lines.append("")
    return "\n".join(lines)
