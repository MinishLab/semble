import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType

from semble.cache import resolve_cache_folder
from semble.types import CallType, SearchResult

logger = logging.getLogger(__name__)


def _get_stats_file() -> Path:
    """Safely create a stats file."""
    return resolve_cache_folder() / "savings.jsonl"


def _use_color() -> bool:
    """Return whether ANSI colors should be emitted."""
    return "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb" and sys.stdout.isatty()


def _color(code: str, text: str, enabled: bool) -> str:
    """Apply an ANSI color code when enabled."""
    return f"\033[{code}m{text}\033[0m" if enabled else text


@dataclass
class BucketStats:
    calls: int = 0
    snippet_chars: int = 0
    file_chars: int = 0
    saved_chars: int = 0

    def add(self, snippet_chars: int, file_chars: int) -> None:
        """Update stats with a call and its character counts."""
        self.calls += 1
        self.snippet_chars += snippet_chars
        self.file_chars += file_chars
        self.saved_chars += max(0, file_chars - snippet_chars)


@dataclass
class SavingsSummary:
    buckets: dict[str, BucketStats]
    call_type_counts: dict[str, int]


@cache
def _import_fcntl() -> ModuleType | None:
    """Return fcntl when available, otherwise None."""
    try:
        return import_module("fcntl")
    except ImportError:  # pragma: no cover
        return None


def save_search_stats(
    results: list[SearchResult],
    call_type: CallType,
    file_sizes: dict[str, int],
    max_snippet_lines: int | None = None,
) -> None:
    """Save stats about a search or find_related call to the stats file."""
    try:
        snippet_chars = sum(
            len("\n".join(result.chunk.content.splitlines()[:max_snippet_lines]))
            if max_snippet_lines and max_snippet_lines > 0
            else 0
            if max_snippet_lines == 0
            else len(result.chunk.content)
            for result in results
        )
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
        stats_file = _get_stats_file()
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("a") as f:
            fcntl = _import_fcntl()
            try:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:  # pragma: no cover
                return  # another process holds the lock; skip this record
            except OSError:  # pragma: no cover
                return  # lock contention or unsupported filesystem; skip
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def build_savings_summary(path: Path | None = None) -> SavingsSummary:
    """Read savings.jsonl and return a SavingsSummary."""
    if path is None:
        path = _get_stats_file()
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


def _format_token_count(tokens: int) -> str:
    """Format a token count with k/M suffix, keeping the ~ prefix for estimates."""
    if tokens >= 1_000_000:
        return f"~{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"~{tokens / 1_000:.1f}k"
    return f"~{tokens}"


def _format_calls(calls: int) -> str:
    """Format a call count with k suffix for thousands."""
    return f"{calls / 1_000:.1f}k" if calls >= 1_000 else str(calls)


def _color_ratio(pct: int, enabled: bool) -> str:
    """Color a savings percentage according to its value."""
    code = "32" if pct >= 80 else "33" if pct >= 50 else "31"
    return _color(code, f"{pct}%", enabled)


def format_savings_report(path: Path | None = None) -> str:
    """Return a formatted token-savings report."""
    if path is None:
        path = _get_stats_file()
    if not path.exists():
        return "No stats yet. Run a search first."

    summary = build_savings_summary(path)
    color = _use_color()
    bar_width = 24
    border_width = 72
    heavy_line = "  " + _color("38;5;244", "═" * border_width, color)
    light_line = "  " + _color("38;5;244", "─" * border_width, color)

    all_time = summary.buckets["All time"]
    total_saved_tokens = all_time.saved_chars // 4
    overall_pct = round(all_time.saved_chars / all_time.file_chars * 100) if all_time.file_chars else 0
    efficiency_filled = round(overall_pct / 100 * bar_width)
    efficiency_bar = _color("32", "█" * efficiency_filled, color)
    efficiency_bar += _color("38;5;244", "░" * (bar_width - efficiency_filled), color)

    lines = [
        "",
        "  " + _color("1;36", "Semble Token Savings", color),
        heavy_line,
        "",
        f"  {_color('1', 'Total saved:', color)}  "
        f"{_color('1;33', _format_token_count(total_saved_tokens) + ' tokens', color)}  "
        f"({_color_ratio(overall_pct, color)})",
        f"  {_color('1', 'Total calls:', color)}  {_color('1;33', _format_calls(all_time.calls), color)}",
        f"  {_color('1', 'Efficiency:', color)}  {efficiency_bar}  {_color_ratio(overall_pct, color)}",
        "",
        "  " + _color("1", "By Period", color),
        light_line,
        f"  {'Period':<14}  {'Calls':>8}  {'Saved':>14}  Ratio",
        light_line,
    ]
    for label, bucket in summary.buckets.items():
        saved_tokens = bucket.saved_chars // 4
        saved_str = _format_token_count(saved_tokens) + " tokens"
        calls_str = _format_calls(bucket.calls)
        if bucket.file_chars > 0:
            ratio = bucket.saved_chars / bucket.file_chars
            filled = round(ratio * bar_width)
            row_bar = _color("32", "█" * filled, color) + _color("38;5;244", "░" * (bar_width - filled), color)
            ratio_str = _color_ratio(round(ratio * 100), color)
        else:
            row_bar = _color("38;5;244", "░" * bar_width, color)
            ratio_str = _color("38;5;244", "–", color)
        lines.append(
            f"  {_color('1', f'{label:<14}', color)}  {_color('1;33', f'{calls_str:>8}', color)}  "
            f"{_color('1;33', f'{saved_str:>14}', color)}  {row_bar}  {ratio_str}"
        )

    if summary.call_type_counts:
        lines += [
            "",
            "  " + _color("1", "By Call Type", color),
            light_line,
            f"  {'#':<4}  {'Call type':<16}  {'Calls':>8}  Share",
            light_line,
        ]
        top = sorted(summary.call_type_counts.items(), key=lambda kv: -kv[1])
        total = sum(summary.call_type_counts.values())
        for i, (call_type, count) in enumerate(top, start=1):
            share = count / total
            filled = max(1, round(share * 16))
            bar = _color("32", "█" * filled, color) + _color("38;5;244", "░" * (16 - filled), color)
            rank = f"{i}."
            lines.append(
                f"  {_color('38;5;244', f'{rank:<4}', color)}  {call_type:<16}  "
                f"{_color('1;33', f'{_format_calls(count):>8}', color)}  {bar}  "
                f"{_color('38;5;244', f'{share * 100:>4.0f}%', color)}"
            )

    lines.append(heavy_line)
    lines.append("")
    return "\n".join(lines)
