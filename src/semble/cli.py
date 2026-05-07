import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from model2vec.utils import get_package_extras

from semble.index import SembleIndex
from semble.stats import _STATS_FILE
from semble.utils import _format_results, _is_git_url, _resolve_chunk

_CLAUDE_FILE_PATH = Path(".claude") / "agents" / "semble-search.md"
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "init", "savings", "-h", "--help"})


def main() -> None:
    """Entry point for the semble command-line tool."""
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_DISPATCH_ARGS:
        _cli_main()
    else:
        _mcp_main()


def _mcp_main() -> None:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    asyncio.run(serve(args.path, ref=args.ref))


def _run_init(*, force: bool = False) -> None:
    """Write the Claude Code sub-agent file into the current project."""
    dest = _CLAUDE_FILE_PATH
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath("agents/semble-search.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


def _add_to_bucket(bucket: dict[str, int], snippet_chars: int, file_chars: int) -> None:
    bucket["calls"] += 1
    bucket["snippet_chars"] += snippet_chars
    bucket["file_chars"] += file_chars


def _parse_stats() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Read savings.jsonl and return (period_buckets, call_type_counts)."""
    now = datetime.now(timezone.utc)
    today = now.date()
    seven_days_ago = (now - timedelta(days=7)).date()

    buckets: dict[str, dict[str, int]] = {
        "Today": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
        "Last 7 days": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
        "All time": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
    }
    call_type_counts: defaultdict[str, int] = defaultdict(int)

    with _STATS_FILE.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            snippet_chars = record.get("snippet_chars", 0)
            file_chars = record.get("file_chars", 0)
            call_type = record.get("call", "search")
            call_type_counts[call_type] += 1
            try:
                record_date = datetime.fromisoformat(record.get("ts", "")).date()
                in_today = record_date == today
                in_last_7 = record_date > seven_days_ago
            except ValueError:
                in_today = in_last_7 = True
            _add_to_bucket(buckets["All time"], snippet_chars, file_chars)
            if in_last_7:
                _add_to_bucket(buckets["Last 7 days"], snippet_chars, file_chars)
            if in_today:
                _add_to_bucket(buckets["Today"], snippet_chars, file_chars)

    return buckets, call_type_counts


def _run_stats(*, verbose: bool = False) -> None:
    """Print a summary of semble usage and token savings from ~/.semble/savings.jsonl."""
    if not _STATS_FILE.exists():
        print("No stats yet. Run a search first.")
        return

    buckets, call_type_counts = _parse_stats()

    bar_width = 16
    heavy_line = "  " + "═" * 64
    light_line = "  " + "─" * 64
    print()
    print("  Semble Token Savings")
    print(heavy_line)
    print(f"  {'Period':<12}  {'Calls':<6}  Savings")
    print(light_line)
    for label, bucket in buckets.items():
        saved_chars = max(0, bucket["file_chars"] - bucket["snippet_chars"])
        saved_tokens = saved_chars // 4
        saved_str = f"~{saved_tokens / 1000:.1f}k" if saved_tokens >= 1000 else f"~{saved_tokens}"
        if bucket["file_chars"] > 0:
            ratio = saved_chars / bucket["file_chars"]
            filled = round(ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = round(ratio * 100)
            print(f"  {label:<12}  {bucket['calls']:<6}  [{bar}]  {saved_str} tokens ({pct}%)")
        else:
            print(f"  {label:<12}  {bucket['calls']:<6}  [{'░' * bar_width}]  {saved_str} tokens")
    if verbose and call_type_counts:
        print()
        print("  Usage Breakdown")
        print(light_line)
        print(f"  {'Call type':<16}  Calls")
        for call_type, count in sorted(call_type_counts.items()):
            print(f"  {call_type:<16}  {count}")
        print(heavy_line)
    print()


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument(
        "-m", "--mode", default="hybrid", choices=["hybrid", "semantic", "bm25"], help="Search mode (default: hybrid)."
    )

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")

    init_p = sub.add_parser("init", help="Write .claude/agents/semble-search.md for Claude Code sub-agent support.")
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(force=args.force)
        return

    if args.command == "savings":
        _run_stats(verbose=args.verbose)
        return

    index = SembleIndex.from_git(args.path) if _is_git_url(args.path) else SembleIndex.from_path(args.path)

    if args.command == "search":
        results = index.search(args.query, top_k=args.top_k, mode=args.mode)
        if not results:
            print("No results found.")
        else:
            print(_format_results(f"Search results for: {args.query!r} (mode={args.mode})", results))

    elif args.command == "find-related":
        chunk = _resolve_chunk(index.chunks, args.file_path, args.line)
        if chunk is None:
            print(f"No chunk found at {args.file_path}:{args.line}.", file=sys.stderr)
            sys.exit(1)
        results = index.find_related(chunk, top_k=args.top_k)
        if not results:
            print(f"No related chunks found for {args.file_path}:{args.line}.")
        else:
            print(_format_results(f"Chunks related to {args.file_path}:{args.line}", results))
