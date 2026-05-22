import argparse
import asyncio
import sys
import warnings
from enum import Enum
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

from model2vec.utils import get_package_extras

from semble.index import SembleIndex
from semble.stats import format_savings_report
from semble.types import ContentType
from semble.utils import format_results, is_git_url, resolve_chunk


class Agent(str, Enum):
    CLAUDE = "claude"
    COPILOT = "copilot"
    CURSOR = "cursor"
    GEMINI = "gemini"
    KIRO = "kiro"
    OPENCODE = "opencode"


_DEFAULT_AGENT = Agent.CLAUDE
_CLI_DISPATCH_ARGS = frozenset({"search", "find-related", "init", "savings", "-h", "--help", "index"})


def _agent_path(agent: Agent) -> Path:
    """Return the project-relative path where the semble sub-agent file should be written."""
    base_dir = ".github" if agent is Agent.COPILOT else f".{agent.value}"
    return Path(base_dir) / "agents" / "semble-search.md"


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


def _run_index(*, path: str, include_text_files: bool = False, out: str) -> None:
    """Index and store a codebase."""
    if is_git_url(path):
        index = SembleIndex.from_git(path, include_text_files=include_text_files)
    else:
        index = SembleIndex.from_path(path, include_text_files=include_text_files)
    Path(out).mkdir(parents=True, exist_ok=True)
    index.save(out)


def _run_init(*, agent: Agent = _DEFAULT_AGENT, force: bool = False) -> None:
    """Write the semble sub-agent file for the given coding agent into the current project."""
    dest = _agent_path(agent)
    if dest.exists() and not force:
        print(f"{dest} already exists. Run with --force to overwrite.", file=sys.stderr)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = files("semble").joinpath(f"agents/{agent.value}.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")


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


def _cli_main() -> None:
    parser = argparse.ArgumentParser(prog="semble")
    sub = parser.add_subparsers(dest="command")

    index_p = sub.add_parser("index", help="Index and store a codebase.")
    index_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    index_p.add_argument(
        "--include-text-files",
        action="store_true",
        help="Also index non-code text files (.md, .yaml, .json, etc.).",
    )
    index_p.add_argument("-o", "--out", type=str, required=True, help="The path to write the pre-built index to.")

    search_p = sub.add_parser("search", help="Search a codebase.")
    search_p.add_argument("query", help="Natural language or code query.")
    search_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    search_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    search_p.add_argument("--index", type=str, default=None, help="A path pointing to a pre-built index.")
    _add_content_args(search_p)

    related_p = sub.add_parser("find-related", help="Find code similar to a specific location.")
    related_p.add_argument("file_path", help="File path as shown in search results.")
    related_p.add_argument("line", type=int, help="Line number (1-indexed).")
    related_p.add_argument("path", nargs="?", default=".", help="Local path or git URL (default: current directory).")
    related_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5).")
    related_p.add_argument("--index", type=str, default=None, help="A path pointing to a pre-built index.")
    _add_content_args(related_p)

    init_p = sub.add_parser("init", help="Write a semble sub-agent file for your coding agent.")
    init_p.add_argument(
        "--agent",
        "-a",
        default=_DEFAULT_AGENT.value,
        choices=[a.value for a in Agent],
        help=f"Coding agent to set up (default: {_DEFAULT_AGENT.value}).",
    )
    init_p.add_argument("--force", action="store_true", help="Overwrite if the file already exists.")

    savings_p = sub.add_parser("savings", help="Show token savings and usage stats.")
    savings_p.add_argument("--verbose", action="store_true", help="Also show usage breakdown by call type.")

    args = parser.parse_args()

    if args.command == "init":
        _run_init(agent=Agent(args.agent), force=args.force)
        return

    if args.command == "index":
        _run_index(path=args.path, include_text_files=args.include_text_files, out=args.out)
        return

    if args.command == "savings":
        print(format_savings_report(verbose=args.verbose), end="")
        return

    if args.index:
        index = SembleIndex.load_from_disk(args.index)
    else:
        content = _resolve_content(args.content, args.include_text_files)
        index = (
            SembleIndex.from_git(args.path, content=content)
            if is_git_url(args.path)
            else SembleIndex.from_path(args.path, content=content)
        )

    if args.command == "search":
        results = index.search(args.query, top_k=args.top_k)
        if not results:
            print("No results found.")
        else:
            print(format_results(f"Search results for: {args.query!r}", results))

    elif args.command == "find-related":
        chunk = resolve_chunk(index.chunks, args.file_path, args.line)
        if chunk is None:
            print(f"No chunk found at {args.file_path}:{args.line}.", file=sys.stderr)
            sys.exit(1)
        results = index.find_related(chunk, top_k=args.top_k)
        if not results:
            print(f"No related chunks found for {args.file_path}:{args.line}.")
        else:
            print(format_results(f"Chunks related to {args.file_path}:{args.line}", results))
