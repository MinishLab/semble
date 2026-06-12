import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from semble.cache import resolve_cache_folder
from semble.types import CallType, SearchResult

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _get_stats_file() -> Path:
    """Safely create a stats file."""
    return resolve_cache_folder() / "savings.jsonl"


def _use_color() -> bool:
    """Return True when ANSI color codes should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def _vis_len(s: str) -> int:
    """Visible length of a string, ignoring ANSI escape sequences."""
    return len(_ANSI_RE.sub("", s))


def _align_left(s: str, width: int) -> str:
    """Pad `s` on the right so its visible width matches `width` (left-aligned)."""
    pad = max(0, width - _vis_len(s))
    return s + " " * pad


def _align_right(s: str, width: int) -> str:
    """Pad `s` on the left so its visible width matches `width` (right-aligned)."""
    pad = max(0, width - _vis_len(s))
    return " " * pad + s


class _C:
    """ANSI color helpers; no-op when color is disabled."""

    __slots__ = ("enabled",)

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def title(self, text: str) -> str:
        return self._wrap("1;36", text)

    def dim(self, text: str) -> str:
        return self._wrap("38;5;244", text)

    def label(self, text: str) -> str:
        return self._wrap("1", text)

    def num(self, text: str) -> str:
        return self._wrap("1;33", text)

    def good(self, text: str) -> str:
        return self._wrap("32", text)

    def bad(self, text: str) -> str:
        return self._wrap("31", text)

    def mid(self, text: str) -> str:
        return self._wrap("33", text)


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
        stats_file = _get_stats_file()
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("a") as f:
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


def _ratio_color(pct: int, c: _C) -> str:
    """Pick a color for a savings ratio percentage."""
    if pct >= 80:
        return c.good(f"{pct}%")
    if pct >= 50:
        return c.mid(f"{pct}%")
    return c.bad(f"{pct}%")


def _row(c: _C, cols: list[tuple[str, int, str]]) -> str:
    """Build a table row with 2-space gutters between columns."""
    gutter = "  "
    parts: list[str] = []
    for i, (align, width, text) in enumerate(cols):
        if i > 0:
            parts.append(gutter)
        if align == "left":
            parts.append(_align_left(text, width))
        else:
            parts.append(_align_right(text, width))
    return "".join(parts)


def format_savings_report(path: Path | None = None) -> str:
    """Return a formatted token-savings report."""
    if path is None:
        path = _get_stats_file()
    if not path.exists():
        return "No stats yet. Run a search first."

    summary = build_savings_summary(path)
    c = _C(_use_color())
    bar_width = 24
    border_w = 64
    heavy_line = "  " + c.dim("═" * border_w)
    light_line = "  " + c.dim("─" * border_w)

    all_time = summary.buckets["All time"]
    total_saved_tokens = all_time.saved_chars // 4
    overall_pct = round(all_time.saved_chars / all_time.file_chars * 100) if all_time.file_chars else 0

    lines: list[str] = ["", "  " + c.title("Semble Token Savings"), heavy_line, ""]

    total_label = c.label("Total saved:")
    total_value = c.num(_format_token_count(total_saved_tokens) + " tokens")
    pct_value = _ratio_color(overall_pct, c)
    lines.append(f"  {total_label}  {total_value}  {c.dim('(')}{pct_value}{c.dim(')')}")

    calls_label = c.label("Total calls:")
    calls_value = c.num(_format_calls(all_time.calls))
    lines.append(f"  {calls_label}  {calls_value}")

    eff_label = c.label("Efficiency:")
    eff_filled = round(overall_pct / 100 * bar_width)
    eff_bar = c.good("█" * eff_filled) + c.dim("░" * (bar_width - eff_filled))
    lines.append(f"  {eff_label}  {eff_bar}  {pct_value}")
    lines.append("")

    lines.append("  " + c.label("By Period"))
    lines.append(light_line)
    period_cols = [("left", 14, "Period"), ("right", 8, "Calls"), ("right", 14, "Saved")]
    lines.append("  " + _row(c, period_cols) + "  " + c.dim("Ratio"))
    lines.append(light_line)
    for label, bucket in summary.buckets.items():
        saved_tokens = bucket.saved_chars // 4
        saved_str = c.num(_format_token_count(saved_tokens) + " tokens")
        calls_str = c.num(_format_calls(bucket.calls))
        if bucket.file_chars > 0:
            ratio = bucket.saved_chars / bucket.file_chars
            filled = round(ratio * bar_width)
            row_bar = c.good("█" * filled) + c.dim("░" * (bar_width - filled))
            pct = round(ratio * 100)
            pct_str = _ratio_color(pct, c)
        else:
            row_bar = c.dim("░" * bar_width)
            pct_str = c.dim("–")
        data_cols = [("left", 14, c.label(label)), ("right", 8, calls_str), ("right", 14, saved_str)]
        lines.append("  " + _row(c, data_cols) + "  " + row_bar + "  " + pct_str)

    if summary.call_type_counts:
        lines.append("")
        lines.append("  " + c.label("By Call Type"))
        lines.append(light_line)
        call_cols = [("left", 4, "#"), ("left", 16, "Call type"), ("right", 8, "Calls")]
        lines.append("  " + _row(c, call_cols) + "  " + c.dim("Share"))
        lines.append(light_line)
        top = sorted(summary.call_type_counts.items(), key=lambda kv: -kv[1])
        total = max(1, sum(summary.call_type_counts.values()))
        max_bar = 16
        for i, (call_type, count) in enumerate(top, start=1):
            share = count / total
            filled = max(1, round(share * max_bar)) if share > 0 else 0
            bar = c.good("█" * filled) + c.dim("░" * (max_bar - filled))
            calls_str = c.num(_format_calls(count))
            share_str = c.dim(f"{share * 100:>4.0f}%")
            data_cols = [("left", 4, c.dim(f"{i}.")), ("left", 16, call_type), ("right", 8, calls_str)]
            lines.append("  " + _row(c, data_cols) + "  " + bar + "  " + share_str)

    lines.append(heavy_line)
    lines.append("")
    return "\n".join(lines)
