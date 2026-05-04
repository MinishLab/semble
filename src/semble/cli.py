import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from semble.index import SembleIndex
from semble.stats import _STATS_FILE
from semble.utils import _format_results, _is_git_url, _resolve_chunk

_CLAUDE_FILE_PATH = Path(".claude") / "agents" / "semble-search.md"
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "init", "stats", "-h", "--help"})


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


def _parse_stats() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Read stats.jsonl and return (period_buckets, call_type_counts)."""
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    this_week = now.isocalendar()[:2]

    buckets: dict[str, dict[str, int]] = {
        "Today": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
        "This week": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
        "All time": {"calls": 0, "snippet_chars": 0, "file_chars": 0},
    }
    call_type_counts: dict[str, int] = {}

    with _STATS_FILE.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sc, fc = r.get("snippet_chars", 0), r.get("file_chars", 0)
            ct = r.get("call", "search")
            call_type_counts[ct] = call_type_counts.get(ct, 0) + 1
            for b in buckets.values():
                b["calls"] += 1
                b["snippet_chars"] += sc
                b["file_chars"] += fc
            try:
                dt = datetime.fromisoformat(r.get("ts", ""))
                if dt.date().isoformat() != today:
                    buckets["Today"]["calls"] -= 1
                    buckets["Today"]["snippet_chars"] -= sc
                    buckets["Today"]["file_chars"] -= fc
                if dt.isocalendar()[:2] != this_week:
                    buckets["This week"]["calls"] -= 1
                    buckets["This week"]["snippet_chars"] -= sc
                    buckets["This week"]["file_chars"] -= fc
            except ValueError:
                pass

    return buckets, call_type_counts


def _run_stats() -> None:
    """Print a summary of semble usage and token savings from ~/.semble/stats.jsonl."""
    if not _STATS_FILE.exists():
        print("No stats yet. Run a search first.")
        return

    buckets, call_type_counts = _parse_stats()

    print("\nSemble stats")
    print("─" * 48)
    print(f"  {'Period':<12}  {'Calls':>6}  {'Tokens saved':>14}")
    print(f"  {'──────':<12}  {'─────':>6}  {'────────────':>14}")
    for label, b in buckets.items():
        t = max(0, b["file_chars"] - b["snippet_chars"]) // 4
        saved = f"~{t / 1000:.0f}k" if t >= 1000 else f"~{t}"
        print(f"  {label:<12}  {b['calls']:>6}  {saved:>14}")
    print()
    if call_type_counts:
        print("  Call breakdown:")
        for ct, n in sorted(call_type_counts.items()):
            print(f"    {ct:<16} {n}")
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

    sub.add_parser("stats", help="Show token savings and usage stats.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(force=args.force)
        return

    if args.command == "stats":
        _run_stats()
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
