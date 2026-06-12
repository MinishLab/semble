import argparse
import asyncio
import json
import re
import sys
import warnings
from importlib.util import find_spec
from shutil import rmtree
from typing import Literal

from model2vec.utils import get_package_extras

from semble.cache import find_index_from_cache_folder, resolve_cache_folder
from semble.index import SembleIndex
from semble.index.types import PersistencePath
from semble.stats import format_savings_report
from semble.types import ContentType
from semble.utils import format_results, is_git_url, resolve_chunk

_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "install", "uninstall", "savings", "-h", "--help", "clear"})
_CLEAR_CHOICE = Literal["all", "index", "savings"]

_SHA_256_REGEX = re.compile(r"^[a-f0-9]{64}$")


def _build_index(path: str, content: list[ContentType]) -> SembleIndex:
    """Build an index from a local path or git URL."""
    return (
        SembleIndex.from_git(path, content=content)
        if is_git_url(path)
        else SembleIndex.from_path(path, content=content)
    )


def _maybe_save_index(index: SembleIndex, path: str) -> None:
    """Save the index to the cache folder if it was not loaded from disk."""
    if not index.loaded_from_disk:
        try:
            cache_folder = find_index_from_cache_folder(path)
            index.save(cache_folder)
        except Exception as e:
            print(f"Error saving index: {e}", file=sys.stderr)


def _add_content_args(p: argparse.ArgumentParser) -> None:
    """Add --content and deprecated --include-text-files to a subparser."""
    p.add_argument(
        "--content",
        nargs="+",
        default=["code"],
        choices=[ct.value for ct in ContentType] + ["all"],
        metavar="TYPE",
        help="Content types to index (space-separated, e.g. --content code docs). Choices: code, docs, config, all. Default: code.",
    )
    p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Deprecated. Use --content all instead.",
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
    _add_content_args(parser)
    args = parser.parse_args()
    if any(find_spec(dep) is None for dep in get_package_extras("semble", "mcp")):
        print("MCP dependencies are not installed. Run: pip install 'semble[mcp]'", file=sys.stderr)
        raise SystemExit(1)
    from semble.mcp import serve

    content = _resolve_content(args.content, args.include_text_files)
    asyncio.run(serve(args.path, ref=args.ref, content=content))


def _resolve_content(content: list[str], include_text_files: bool) -> list[ContentType]:
    """Resolve --content and the deprecated --include-text-files into a list of ContentType values."""
    if include_text_files:
        warnings.warn(
            "--include-text-files is deprecated and will be removed in a future version. Use --content all instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if include_text_files or "all" in content:
        return [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]
    return [ContentType(c) for c in content]


def _load_index(path: str, content: list[ContentType]) -> SembleIndex:
    """Build an index from a local path or git URL, exiting on FileNotFoundError."""
    try:
        return _build_index(path, content)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def _run_search(path: str, query: str, top_k: int, content: list[ContentType]) -> None:
    """Handle the `search` subcommand."""
    index = _load_index(path, content)
    results = index.search(query, top_k=top_k)
    out = format_results(query, results) if results else {"error": "No results found."}
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _run_find_related(path: str, file_path: str, line: int, top_k: int, content: list[ContentType]) -> None:
    """Handle the `find-related` subcommand."""
    index = _load_index(path, content)
    chunk = resolve_chunk(index.chunks, file_path, line)
    if chunk is None:
        print(f"No chunk found at {file_path}:{line}.", file=sys.stderr)
        sys.exit(1)
    results = index.find_related(chunk, top_k=top_k)
    out = (
        format_results(f"Chunks related to {file_path}:{line}", results)
        if results
        else {"error": f"No related chunks found for {file_path}:{line}."}
    )
    print(json.dumps(out))
    _maybe_save_index(index, path)


def _run_clear(clear_type: _CLEAR_CHOICE) -> None:
    """Run the `clear` subcommand."""
    cache_folder = resolve_cache_folder()
    if clear_type == "index" or clear_type == "all":
        indexes = []
        for path in cache_folder.glob("*/index"):
            if not _SHA_256_REGEX.match(path.parent.name):
                continue
            if PersistencePath.from_path(path).non_existing():
                continue
            indexes.append(path)

        if not indexes:
            print(f"No indexes found to clear in `{cache_folder}`")
        else:
            for path in indexes:
                index_folder = path.parent
                rmtree(index_folder)
                print(f"Cleared index at `{index_folder}`")

    if clear_type == "savings" or clear_type == "all":
        path = cache_folder / "savings.jsonl"
        if not path.exists():
            print(f"No savings file found at `{path}`")
        else:
            path.unlink()
            print(f"Cleared savings at `{path}`")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    _add_content_args(search_p)

    clear_p = sub.add_parser("clear", help="Clear the index cache.")
    clear_p.add_argument("type", choices=["all", "index", "savings"], help="Type of cache to clear.")

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    _add_content_args(related_p)

    sub.add_parser("savings", help="Show token savings and usage stats.")

    sub.add_parser("install", help="Interactively configure semble across coding agents.")
    sub.add_parser("uninstall", help="Interactively remove semble configuration from coding agents.")

    args = parser.parse_args()

    if args.command == "savings":
        print(format_savings_report())
    elif args.command in ("install", "uninstall"):
        from semble.installer import run

        run(args.command)
    elif args.command == "clear":
        _run_clear(args.type)
    elif args.command == "search":
        _run_search(args.path, args.query, args.top_k, _resolve_content(args.content, args.include_text_files))
    elif args.command == "find-related":
        _run_find_related(
            args.path, args.file_path, args.line, args.top_k, _resolve_content(args.content, args.include_text_files)
        )
