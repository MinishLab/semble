import argparse
import asyncio
import sys
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from model2vec.utils import get_package_extras

from semble.index import SembleIndex
from semble.index.file_walker import FILE_TYPES
from semble.stats import format_savings_report
from semble.utils import _format_results, _is_git_url, _resolve_chunk

_CLAUDE_FILE_PATH = Path(".claude") / "agents" / "semble-search.md"
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "init", "savings", "-h", "--help"})


def _parse_extensions(value: str) -> frozenset[str]:
    """Split a comma-separated string of extensions into a frozenset."""
    return frozenset(ext.strip() for ext in value.split(",") if ext.strip())


def _resolve_extensions(args: argparse.Namespace) -> frozenset[str] | None:
    """Return the effective extension set based on parsed CLI arguments."""
    if args.extensions is not None:
        return _parse_extensions(args.extensions)
    if args.add_extensions is not None:
        return frozenset(FILE_TYPES.keys()) | _parse_extensions(args.add_extensions)
    return None


def _add_extension_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --extensions and --add-extension flags to *parser*."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated list of file extensions to index instead of the default set (e.g. '.py,.js,.ext').",
    )
    group.add_argument(
        "--add-extensions",
        default=None,
        help="Comma-separated list of file extensions to add to the default set (e.g. '.ext,.lua').",
    )


def _add_include_text_files_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --include-text-files flag to *parser*."""
    parser.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )


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
    _add_extension_args(parser)
    _add_include_text_files_arg(parser)
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    asyncio.run(
        serve(
            args.path,
            ref=args.ref,
            include_text_files=args.include_text_files,
            extensions=_resolve_extensions(args),
        )
    )


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
    _add_extension_args(search_p)
    _add_include_text_files_arg(search_p)

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    _add_extension_args(related_p)
    _add_include_text_files_arg(related_p)

    init_p = sub.add_parser("init", help="Write .claude/agents/semble-search.md for Claude Code sub-agent support.")
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(force=args.force)
        return

    if args.command == "savings":
        print(format_savings_report(verbose=args.verbose), end="")
        return

    include_text = args.include_text_files
    extensions = _resolve_extensions(args)
    index = (
        SembleIndex.from_git(args.path, include_text_files=include_text, extensions=extensions)
        if _is_git_url(args.path)
        else SembleIndex.from_path(args.path, include_text_files=include_text, extensions=extensions)
    )

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
